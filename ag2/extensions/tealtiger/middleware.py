# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
# SPDX-License-Identifier: Apache-2.0

"""TealTiger deterministic governance middleware.

Implements MiddlewareFactory pattern: TealTigerMiddleware is the factory (holds
long-lived state like frozen agents, decisions, cumulative cost) and creates
per-turn _TealTigerPerTurn instances that share a reference to the factory state.

No external dependencies beyond AG2 and the standard library.
"""

import fnmatch
import hashlib
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ag2.annotations import Context
    from ag2.events import BaseEvent, ToolCallEvent
    from ag2.middleware.base import ToolExecution

from ag2.events import ToolErrorEvent, ToolResultEvent
from ag2.events.input_events import DataInput, TextInput
from ag2.extensions.tealtiger.types import (
    ARG_TYPES_BY_NAME,
    DEFAULT_INJECTION_CONFIDENCE_THRESHOLD,
    INJECTION_PATTERNS,
    INJECTION_TECHNIQUES,
    PII_PATTERNS,
    SECRET_PATTERNS,
    GovernanceDecision,
    GovernanceMode,
    GovernancePolicy,
    InjectionFinding,
    OutputAction,
    TEECReceipt,
    most_restrictive,
)
from ag2.middleware import BaseMiddleware
from ag2.middleware.base import ToolResultType
from ag2.utils import AGENT_CONTEXT_DEPENDENCY_KEY

# Injection findings keep a snippet of the match as evidence, not the whole argument.
_MAX_MATCHED_TEXT_CHARS = 100

# Stands in for the argument name in an arg_validation reason code when the call's
# arguments were not a mapping and the hit cannot be attributed to one argument.
_UNNAMED_ARG = "*"


def _string_leaves(value: Any) -> list[str]:
    """Every string inside a structured tool result, in traversal order.

    Non-string scalars are skipped: they cannot be rewritten in place, so matching one
    would mean withholding results over numeric ids that merely look like a phone number.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [leaf for item in value.values() for leaf in _string_leaves(item)]
    if isinstance(value, (list, tuple, set)):
        return [leaf for item in value for leaf in _string_leaves(item)]
    return []


def _rewrite_string_leaves(value: Any, rewrite: Callable[[str], str]) -> Any:
    """`value` with `rewrite` applied to every string `_string_leaves` would collect."""
    if isinstance(value, str):
        return rewrite(value)
    if isinstance(value, Mapping):
        return {key: _rewrite_string_leaves(item, rewrite) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_string_leaves(item, rewrite) for item in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_string_leaves(item, rewrite) for item in value)
    if isinstance(value, set):
        return {_rewrite_string_leaves(item, rewrite) for item in value}
    return value


def _scannable_text(part: Any) -> str | None:
    """The text an output scan reads from one result part, or `None` if it has none.

    A tool that returns a `dict` hands back a `DataInput`, not a `TextInput` — an
    SSN in `{"ssn": "123-45-6789"}` leaks just as readily as one in a sentence, so
    structured parts are scanned through their strings.
    """
    if isinstance(part, TextInput) and isinstance(part.content, str):
        return part.content
    if isinstance(part, DataInput):
        leaves = _string_leaves(part.data)
        return "\n".join(leaves) if leaves else None
    return None


def _rewrite_part(part: Any, rewrite: Callable[[str], str]) -> None:
    """Apply `rewrite` in place to whatever `_scannable_text` reads from `part`."""
    if isinstance(part, TextInput):
        part.content = rewrite(part.content)
    elif isinstance(part, DataInput):
        part.data = _rewrite_string_leaves(part.data, rewrite)


def _reconstructable(exc: BaseException) -> bool:
    """Whether `type(exc)(message)` is safe to construct.

    Custom exceptions can take extra required constructor args; for those we fall
    back to a plain `Exception` rather than risk a TypeError while sanitizing.
    """
    try:
        type(exc)(str(exc))
    except Exception:
        return False
    return True


def _matches_blocked_terms(text: str, spec: dict[str, Any]) -> bool:
    """Whether `text` contains any of the spec's blocked terms, case-insensitively."""
    if "blocked_terms" not in spec:
        return False
    lowered = text.lower()
    return any(term.lower() in lowered for term in spec["blocked_terms"])


def _matches_blocked_patterns(text: str, spec: dict[str, Any]) -> bool:
    """Whether any of the spec's blocked patterns matches `text`.

    Patterns are compiled by `GovernancePolicy.arg_validation`, so nothing is
    compiled here.
    """
    return "blocked_patterns" in spec and any(pattern.search(text) for pattern in spec["blocked_patterns"])


def _check_value(value: Any, spec: dict[str, Any]) -> str | None:
    """Name the first check ``value`` violates, or ``None``.

    Type runs first so a length or membership failure is never reported for a
    value that was the wrong shape to begin with.
    """
    text = value if isinstance(value, str) else str(value)

    if "type" in spec:
        expected = ARG_TYPES_BY_NAME[spec["type"]]
        # bool is a subclass of int, so reject a bool where "int" is required.
        if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            return "type"

    if "allowed_values" in spec and value not in spec["allowed_values"]:
        return "allowed_values"

    if "max_length" in spec and len(text) > spec["max_length"]:
        return "max_length"

    if "min_length" in spec and len(text) < spec["min_length"]:
        return "min_length"

    if _matches_blocked_terms(text, spec):
        return "blocked_terms"

    if _matches_blocked_patterns(text, spec):
        return "blocked_patterns"

    return None


def _scan_unnamed_args(args_str: str, constraints: dict[str, dict[str, Any]]) -> str | None:
    """Check a call whose arguments are not a mapping.

    With no names to read by, length/type/`allowed_values` cannot apply. Rather
    than let the call through, the term and pattern checks run over the whole
    serialized call — fail-closed, so a banned term denies even from an
    argument the policy does not constrain. Reported against ``*``, since no
    single argument can honestly be blamed.
    """
    for spec in constraints.values():
        if _matches_blocked_terms(args_str, spec):
            return f"{_UNNAMED_ARG}:blocked_terms"
        if _matches_blocked_patterns(args_str, spec):
            return f"{_UNNAMED_ARG}:blocked_patterns"
    return None


class TealTigerMiddleware:
    """Deterministic governance middleware factory for AG2.

    Holds long-lived governance state (decisions, receipts, frozen agents, cost)
    across turns. Creates per-turn middleware instances that share this state.

    This is a MiddlewareFactory — pass it directly to the agent's middleware list.

    No external dependencies — all governance evaluation is deterministic and
    runs inline using pattern matching (fnmatch, regex).

    Args:
        policies: List of GovernancePolicy definitions.
        mode: Governance mode (OBSERVE, MONITOR, ENFORCE).
        budget_limit: Per-session cost ceiling in USD (enforced independently).
        cost_per_call: Estimated cost per tool call in USD (default: 0.002).
        on_decision: Optional callback invoked with each GovernanceDecision.
        on_receipt: Optional callback invoked with each TEECReceipt.

    Example:
        from ag2.extensions.tealtiger import TealTigerMiddleware, GovernancePolicy

        governance = TealTigerMiddleware(
            policies=[
                GovernancePolicy.tool_allowlist(["search", "read_*"]),
                GovernancePolicy.pii_block(["ssn", "credit_card"]),
                GovernancePolicy.cost_limit(max_per_session=5.0),
            ],
            mode="ENFORCE",
        )
    """

    def __init__(
        self,
        policies: list[GovernancePolicy] | None = None,
        mode: str | GovernanceMode = GovernanceMode.ENFORCE,
        budget_limit: float = float("inf"),
        cost_per_call: float = 0.002,
        on_decision: Callable[[GovernanceDecision], None] | None = None,
        on_receipt: Callable[[TEECReceipt], None] | None = None,
    ) -> None:
        self.policies = policies or []
        self.mode = GovernanceMode(mode) if isinstance(mode, str) else mode
        self.budget_limit = budget_limit
        self.cost_per_call = cost_per_call
        self.on_decision = on_decision
        self.on_receipt = on_receipt

        # Long-lived state (survives across turns)
        self._decisions: list[GovernanceDecision] = []
        self._receipts: list[TEECReceipt] = []
        self._frozen_agents: set[str] = set()
        self._cumulative_cost: float = 0.0

        # Compute policy digest for receipts
        policy_str = str(sorted((p.type, str(p.config)) for p in self.policies))
        self._policy_digest = hashlib.sha256(policy_str.encode()).hexdigest()[:16]

    def __call__(self, event: "BaseEvent", context: "Context") -> "BaseMiddleware":
        """MiddlewareFactory protocol: create per-turn middleware instance."""
        return _TealTigerPerTurn(event, context, factory=self)

    # ─── Public API (accessible on the factory) ──────────────────────────

    def freeze(self, agent_name: str) -> None:
        """Freeze an agent — blocks all tool calls for this agent."""
        self._frozen_agents.add(agent_name)

    def unfreeze(self, agent_name: str) -> None:
        """Unfreeze an agent — restores normal governance."""
        self._frozen_agents.discard(agent_name)

    def is_frozen(self, agent_name: str) -> bool:
        """Check if an agent is currently frozen."""
        return agent_name in self._frozen_agents

    @property
    def decisions(self) -> list[GovernanceDecision]:
        """All governance decisions made across all turns."""
        return list(self._decisions)

    @property
    def receipts(self) -> list[TEECReceipt]:
        """All TEEC receipts generated across all turns."""
        return list(self._receipts)

    @property
    def total_cost(self) -> float:
        """Cumulative cost tracked across all tool calls."""
        return self._cumulative_cost

    @property
    def deny_count(self) -> int:
        """Number of denied decisions."""
        return sum(1 for d in self._decisions if d.action == "DENY")

    def reset(self) -> None:
        """Reset all state — decisions, receipts, cost, frozen agents."""
        self._decisions.clear()
        self._receipts.clear()
        self._frozen_agents.clear()
        self._cumulative_cost = 0.0


class _TealTigerPerTurn(BaseMiddleware):
    """Per-turn middleware instance — evaluates governance inline."""

    def __init__(
        self,
        event: "BaseEvent",
        context: "Context",
        factory: TealTigerMiddleware,
    ) -> None:
        super().__init__(event, context)
        self._factory = factory
        self._agent_name = self._get_agent_name(context)

    async def on_turn(
        self,
        call_next: Callable[..., Any],
        event: "BaseEvent",
        context: "Context",
    ) -> Any:
        """Kill switch enforcement at the turn level.

        ENFORCE mode: frozen agent's turn is blocked with ToolErrorEvent.
        MONITOR mode: frozen agent is logged but allowed through.
        OBSERVE mode: no evaluation, pass through.
        """
        agent_name = self._agent_name or "unknown"

        # OBSERVE: no evaluation at turn level
        if self._factory.mode == GovernanceMode.OBSERVE:
            return await call_next(event, context)

        # Check kill switch
        if agent_name != "unknown" and self._factory.is_frozen(agent_name):
            decision = GovernanceDecision(
                action="DENY",
                mode=self._factory.mode.value,
                agent_name=agent_name,
                tool_name="*",
                reason_codes=["AGENT_FROZEN"],
                risk_score=100,
            )
            self._factory._decisions.append(decision)
            if self._factory.on_decision:
                self._factory.on_decision(decision)

            if self._factory.mode == GovernanceMode.ENFORCE:
                return ToolErrorEvent.from_call(
                    event,
                    error=Exception(
                        f"[GOVERNANCE DENIED] Agent '{agent_name}' is frozen (kill switch active). All actions blocked."
                    ),
                )
            # MONITOR: record but allow through

        return await call_next(event, context)

    async def on_tool_execution(
        self,
        call_next: "ToolExecution",
        event: "ToolCallEvent",
        context: "Context",
    ) -> "ToolResultType":
        """Govern a tool call on both sides of its execution.

        Policies over the call's *arguments* are evaluated before it runs and can
        deny it; `output_scan` policies then scan the *result* on the way back.
        """
        start_time = time.perf_counter()
        tool_name = event.name
        tool_args = event.serialized_arguments

        # OBSERVE mode: skip policy evaluation, just pass through with audit
        if self._factory.mode == GovernanceMode.OBSERVE:
            self._factory._cumulative_cost += self._factory.cost_per_call
            result = await call_next(event, context)
            decision = GovernanceDecision(
                action="ALLOW",
                mode="OBSERVE",
                agent_name=self._agent_name or "unknown",
                tool_name=tool_name,
                reason_codes=["OBSERVE_PASSTHROUGH"],
            )
            decision.evaluation_time_ms = round((time.perf_counter() - start_time) * 1000, 3)
            decision.cumulative_cost = self._factory._cumulative_cost
            self._factory._decisions.append(decision)
            if self._factory.on_decision:
                self._factory.on_decision(decision)
            outcome = "error" if isinstance(result, ToolErrorEvent) else "executed"
            self._emit_receipt(decision, execution_outcome=outcome)
            return result

        # MONITOR and ENFORCE: evaluate policies
        decision = self._evaluate(tool_name, tool_args)
        decision.evaluation_time_ms = round((time.perf_counter() - start_time) * 1000, 3)

        # Record decision
        self._factory._decisions.append(decision)
        if self._factory.on_decision:
            self._factory.on_decision(decision)

        # Handle DENY in ENFORCE mode
        if self._factory.mode == GovernanceMode.ENFORCE and decision.action == "DENY":
            self._emit_receipt(decision, execution_outcome="blocked")
            reason = ", ".join(decision.reason_codes)
            return ToolErrorEvent.from_call(
                event,
                error=Exception(
                    f"[GOVERNANCE DENIED] Tool '{tool_name}' blocked. "
                    f"Reason: {reason}. Decision ID: {decision.decision_id}"
                ),
            )

        # Track cost for allowed calls
        self._factory._cumulative_cost += self._factory.cost_per_call
        decision.cost_tracked = self._factory.cost_per_call
        decision.cumulative_cost = self._factory._cumulative_cost

        # Execute the tool
        result = await call_next(event, context)

        # Emit receipt for executed tool
        outcome = "error" if isinstance(result, ToolErrorEvent) else "executed"
        self._emit_receipt(decision, execution_outcome=outcome)

        # Post-tool defense: scan the RESULT for PII/secrets before it flows back
        # into agent context. Applies to successful and error results alike, and
        # only if an output_scan policy is configured.
        return self._scan_result(event, tool_name, result)

    def _evaluate(self, tool_name: str, tool_args: Any) -> GovernanceDecision:
        """Evaluate governance policies deterministically."""
        action = "ALLOW"
        reason_codes: list[str] = []
        risk_score = 0
        agent_name = self._agent_name or "unknown"

        if agent_name != "unknown" and self._factory.is_frozen(agent_name):
            return GovernanceDecision(
                action="DENY",
                mode=self._factory.mode.value,
                agent_name=agent_name,
                tool_name=tool_name,
                reason_codes=["AGENT_FROZEN"],
                risk_score=100,
            )

        if self._factory._cumulative_cost >= self._factory.budget_limit:
            return GovernanceDecision(
                action="DENY",
                mode=self._factory.mode.value,
                agent_name=agent_name,
                tool_name=tool_name,
                reason_codes=["BUDGET_EXCEEDED"],
                risk_score=70,
                cumulative_cost=self._factory._cumulative_cost,
            )

        args_str = str(tool_args) if not isinstance(tool_args, str) else tool_args

        for policy in self._factory.policies:
            if policy.type == "prompt_injection_block":
                techniques = policy.config.get("techniques", list(INJECTION_TECHNIQUES))
                threshold = policy.config.get("confidence_threshold", DEFAULT_INJECTION_CONFIDENCE_THRESHOLD)
                injection_findings = self._detect_prompt_injection(args_str, techniques, threshold)
                if injection_findings:
                    top = injection_findings[0]
                    action = "DENY"
                    reason_codes.append(f"PROMPT_INJECTION:{top.technique}/{top.pattern_name}")
                    risk_score = max(risk_score, 95)
                    break
            elif policy.type == "tool_allowlist":
                allowed = policy.config.get("allowed", [])
                if not any(fnmatch.fnmatch(tool_name, p) for p in allowed):
                    action = "DENY"
                    reason_codes.append("TOOL_NOT_ALLOWED")
                    risk_score = max(risk_score, 80)
                    break

            elif policy.type == "tool_blocklist":
                blocked = policy.config.get("blocked", [])
                if any(fnmatch.fnmatch(tool_name, p) for p in blocked):
                    action = "DENY"
                    reason_codes.append("TOOL_BLOCKED")
                    risk_score = max(risk_score, 80)
                    break

            elif policy.type == "arg_validation":
                if fnmatch.fnmatch(tool_name, policy.config.get("tool", "")):
                    violation = self._validate_args(tool_args, args_str, policy.config.get("constraints", {}))
                    if violation is not None:
                        action = "DENY"
                        reason_codes.append(f"ARG_VALIDATION:{violation}")
                        risk_score = max(risk_score, 85)
                        break

            elif policy.type == "pii_block":
                categories = policy.config.get("categories", [])
                pii_found = self._detect_pii(args_str, categories)
                if pii_found:
                    action = "DENY"
                    reason_codes.extend(f"PII_DETECTED:{cat}" for cat in pii_found)
                    risk_score = max(risk_score, 90)
                    break

            elif policy.type == "secret_detection" and self._detect_secrets(args_str):
                action = "DENY"
                reason_codes.append("SECRET_DETECTED")
                risk_score = max(risk_score, 95)
                break

            elif policy.type == "cost_limit":
                limit = policy.config.get("max_per_session", self._factory.budget_limit)
                if self._factory._cumulative_cost >= limit:
                    action = "DENY"
                    reason_codes.append("BUDGET_EXCEEDED")
                    risk_score = max(risk_score, 70)
                    break

        return GovernanceDecision(
            action=action,
            mode=self._factory.mode.value,
            agent_name=agent_name,
            tool_name=tool_name,
            reason_codes=reason_codes or (["POLICY_ALLOW"] if action == "ALLOW" else []),
            risk_score=risk_score,
            cumulative_cost=self._factory._cumulative_cost,
        )

    def _scan_result(self, event: "ToolCallEvent", tool_name: str, result: "ToolResultType") -> "ToolResultType":
        """Scan a tool result for PII/secrets, and redact, withhold, or flag it.

        Applies to both successful results and error results (``ToolErrorEvent``),
        whose exception message and traceback reach the model just like a returned
        value. Each detector resolves its own action across the configured
        ``output_scan`` policies, taking the most restrictive one asked for. An
        error stays an error: a blocked error result is replaced with a sanitized
        governance error, never turned into a success.
        """
        is_error = isinstance(result, ToolErrorEvent)
        # ToolResultEvent covers both successful results and ToolErrorEvent
        # (a subclass): a tool that raises with an SSN or credential in its
        # message leaks it into model context just as a returned value would,
        # so error results are scanned too. Non-result types have nothing to scan.
        if not isinstance(result, ToolResultEvent):
            return result

        policies = [p for p in self._factory.policies if p.type == "output_scan"]
        if not policies:
            return result

        scannable = [part for part in result.result.parts if _scannable_text(part) is not None]
        if not scannable:
            return result

        combined = "\n".join(_scannable_text(part) or "" for part in scannable)

        # Resolve each detector independently: which categories actually hit, and
        # the most restrictive action any policy that scanned them asked for.
        pii_hits: list[str] = []
        pii_actions: list[OutputAction] = []
        secret_actions: list[OutputAction] = []

        for policy in policies:
            cfg = policy.config
            if cfg.get("scan_pii", True):
                found = self._detect_pii(combined, cfg.get("categories", []))
                if found:
                    pii_hits.extend(found)
                    pii_actions.append(OutputAction(cfg.get("pii_action", OutputAction.REDACT)))
            if cfg.get("scan_secrets", True) and self._detect_secrets(combined):
                secret_actions.append(OutputAction(cfg.get("secret_action", OutputAction.BLOCK)))

        pii_categories = list(dict.fromkeys(pii_hits))
        secret_hit = bool(secret_actions)
        if not pii_categories and not secret_hit:
            return result

        pii_action = most_restrictive(pii_actions, default=OutputAction.FLAG)
        secret_action = most_restrictive(secret_actions, default=OutputAction.FLAG)

        reason_codes = [f"OUTPUT_PII_DETECTED:{cat}" for cat in pii_categories]
        if secret_hit:
            reason_codes.append("OUTPUT_SECRET_DETECTED")
        risk_score = 90 if secret_hit else 60
        agent_name = self._agent_name or "unknown"

        enforcing = self._factory.mode == GovernanceMode.ENFORCE
        triggered = [pii_action] if pii_categories else []
        if secret_hit:
            triggered.append(secret_action)

        # BLOCK (ENFORCE only) — withhold the whole result.
        if OutputAction.BLOCK in triggered and enforcing:
            return self._withhold_result(event, tool_name, reason_codes, risk_score, agent_name)

        # A BLOCK outside ENFORCE degrades to redaction so the value still never leaks.
        redact_pii = bool(pii_categories) and pii_action in (OutputAction.REDACT, OutputAction.BLOCK)
        redact_secrets = secret_hit and secret_action in (OutputAction.REDACT, OutputAction.BLOCK)

        action_value = "ALLOW"
        if redact_pii or redact_secrets:
            reason_codes.append("OUTPUT_REDACTED")
            categories = pii_categories if redact_pii else []
            for part in scannable:
                _rewrite_part(part, lambda text: self._redact(text, categories, redact_secrets))
            # An error result also carries the exception itself; `str(error)` reaches
            # the model independently of the parts, so redact it in place too.
            if is_error:
                self._redact_error(result, categories, redact_secrets)

            # Detection runs over the joined parts, redaction over each part alone, so a
            # value straddling two parts can be found but not removed. Rather than let it
            # through, withhold the result (or, outside ENFORCE, the offending parts).
            residual = "\n".join(_scannable_text(part) or "" for part in scannable)
            if self._detect_pii(residual, categories) or (redact_secrets and self._detect_secrets(residual)):
                reason_codes.append("OUTPUT_REDACTION_INCOMPLETE")
                if enforcing:
                    return self._withhold_result(event, tool_name, reason_codes, risk_score, agent_name)
                for part in scannable:
                    _rewrite_part(part, lambda _text: "[REDACTED:unredactable]")

            # `action` carries only ALLOW/DENY/MONITOR; a rewritten-but-delivered result
            # is an ALLOW, and `OUTPUT_REDACTED` in the reason codes is what says it was
            # rewritten. MONITOR keeps its own marker, as it does for a call it let past.
            action_value = "MONITOR" if self._factory.mode == GovernanceMode.MONITOR else "ALLOW"

        self._record_decision(
            action=action_value,
            agent_name=agent_name,
            tool_name=tool_name,
            reason_codes=reason_codes,
            risk_score=risk_score,
            execution_outcome="executed",
        )
        return result

    def _withhold_result(
        self,
        event: "ToolCallEvent",
        tool_name: str,
        reason_codes: list[str],
        risk_score: int,
        agent_name: str,
    ) -> "ToolErrorEvent":
        """Replace a result carrying sensitive data with a governance error."""
        reason_codes = [*reason_codes, "OUTPUT_BLOCKED"]
        decision = self._record_decision(
            action="DENY",
            agent_name=agent_name,
            tool_name=tool_name,
            reason_codes=reason_codes,
            risk_score=risk_score,
            execution_outcome="blocked",
        )
        reason = ", ".join(reason_codes)
        return ToolErrorEvent.from_call(
            event,
            error=Exception(
                f"[GOVERNANCE DENIED] Result of tool '{tool_name}' withheld — "
                f"sensitive data detected. Reason: {reason}. Decision ID: {decision.decision_id}"
            ),
        )

    def _record_decision(
        self,
        action: str,
        agent_name: str,
        tool_name: str,
        reason_codes: list[str],
        risk_score: int,
        execution_outcome: str,
    ) -> GovernanceDecision:
        """Record a decision on the factory, notify `on_decision`, and emit its receipt."""
        decision = GovernanceDecision(
            action=action,
            mode=self._factory.mode.value,
            agent_name=agent_name,
            tool_name=tool_name,
            reason_codes=reason_codes,
            risk_score=risk_score,
            cumulative_cost=self._factory._cumulative_cost,
        )
        self._factory._decisions.append(decision)
        if self._factory.on_decision:
            self._factory.on_decision(decision)
        self._emit_receipt(decision, execution_outcome=execution_outcome)
        return decision

    def _redact_error(self, result: "ToolErrorEvent", categories: list[str], redact_secrets: bool) -> None:
        """Sanitize the exception carried by an error result, in place.

        `ToolErrorEvent.error` is a separate `Exception` whose rendered string
        reaches the model alongside `result.parts`. Redacting the parts alone
        would leave the credential/PII visible via `str(error)`, so the error is
        replaced with one carrying the redacted message.
        """
        original = str(result.error)
        redacted = self._redact(original, categories, redact_secrets)
        if redacted != original:
            result.error = type(result.error)(redacted) if _reconstructable(result.error) else Exception(redacted)

    @staticmethod
    def _redact(text: str, categories: list[str], redact_secrets: bool) -> str:
        """Replace PII (in ``categories``) and, if requested, secrets with markers."""
        redacted = text
        for cat in categories:
            pattern = PII_PATTERNS.get(cat)
            if pattern is not None:
                redacted = pattern.sub(f"[REDACTED:{cat}]", redacted)
        if redact_secrets:
            for pattern in SECRET_PATTERNS:
                redacted = pattern.sub("[REDACTED:secret]", redacted)
        return redacted

    def _emit_receipt(self, decision: GovernanceDecision, execution_outcome: str) -> None:
        """Emit a TEEC receipt for the governance decision."""
        receipt = TEECReceipt(
            decision_id=decision.decision_id,
            agent_name=decision.agent_name,
            tool_name=decision.tool_name,
            action=decision.action,
            execution_outcome=execution_outcome,
            reason_codes=decision.reason_codes,
            risk_score=decision.risk_score,
            policy_digest=self._factory._policy_digest,
        )
        self._factory._receipts.append(receipt)
        if self._factory.on_receipt:
            self._factory.on_receipt(receipt)

    @staticmethod
    def _detect_pii(text: str, categories: list[str]) -> list[str]:
        """Detect PII patterns in text."""
        found = []
        for cat in categories:
            pattern = PII_PATTERNS.get(cat)
            if pattern and pattern.search(text):
                found.append(cat)
        return found

    @staticmethod
    def _detect_secrets(text: str) -> bool:
        """Detect secret patterns in text."""
        return any(p.search(text) for p in SECRET_PATTERNS)

    @staticmethod
    def _validate_args(tool_args: Any, args_str: str, constraints: dict[str, dict[str, Any]]) -> str | None:
        """Name the first violated constraint as ``"{arg}:{check}"``, or ``None``.

        ``args_str`` is the caller's already-serialized form of the call, used
        only by the non-mapping fallback.
        """
        if not isinstance(tool_args, dict):
            return _scan_unnamed_args(args_str, constraints)

        for arg_name, spec in constraints.items():
            if arg_name not in tool_args:
                continue
            violated = _check_value(tool_args[arg_name], spec)
            if violated is not None:
                return f"{arg_name}:{violated}"

        return None

    @staticmethod
    def _detect_prompt_injection(
        text: str, techniques: list[str], confidence_threshold: float
    ) -> list[InjectionFinding]:
        """Detect prompt injection patterns in text.

        Args:
            text: The text to scan (typically serialized tool arguments).
            techniques: Technique categories to check.
            confidence_threshold: Minimum pattern confidence for a pattern to be evaluated.

        Returns:
            At most one finding per matching pattern, sorted by confidence (highest first).
        """
        findings: list[InjectionFinding] = []
        for technique in techniques:
            for injection_pattern in INJECTION_PATTERNS.get(technique, []):
                if injection_pattern.confidence < confidence_threshold:
                    continue
                # The first match is enough — the decision only needs the top finding, so
                # scanning every occurrence of every pattern would be wasted work.
                match = injection_pattern.pattern.search(text)
                if match is None:
                    continue
                findings.append(
                    InjectionFinding(
                        technique=injection_pattern.technique,
                        pattern_name=injection_pattern.name,
                        matched_text=match.group()[:_MAX_MATCHED_TEXT_CHARS],
                        confidence=injection_pattern.confidence,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        findings.sort(key=lambda finding: finding.confidence, reverse=True)
        return findings

    def _get_agent_name(self, context: "Context") -> str | None:
        """Extract agent name from context dependencies."""
        agent = context.dependencies.get(AGENT_CONTEXT_DEPENDENCY_KEY)
        return agent.name if agent is not None else None
