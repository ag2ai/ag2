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
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ag2.annotations import Context
    from ag2.events import BaseEvent, ToolCallEvent
    from ag2.middleware.base import ToolExecution

from ag2.events import ToolErrorEvent
from ag2.extensions.tealtiger.types import (
    GovernanceDecision,
    GovernanceMode,
    GovernancePolicy,
    TEECReceipt,
)
from ag2.middleware import BaseMiddleware
from ag2.middleware.base import ToolResultType
from ag2.utils import AGENT_CONTEXT_DEPENDENCY_KEY

# ── PII and secret patterns ──────────────────────────────────────────────────

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
}

_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(sk-[a-zA-Z0-9]{20,})\b"),
    re.compile(r"\b(ghp_[a-zA-Z0-9]{36,})\b"),
    re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
    re.compile(r"\b(xox[bpors]-[a-zA-Z0-9-]+)\b"),
    re.compile(r"(?i)(?:api[_-]?key|apikey)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"),
]


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
        """Evaluate governance policies before tool execution."""
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

        return result

    # ─── Inline policy evaluation (no external deps) ─────────────────────

    def _evaluate(self, tool_name: str, tool_args: Any) -> GovernanceDecision:
        """Evaluate governance policies deterministically."""
        action = "ALLOW"
        reason_codes: list[str] = []
        risk_score = 0
        agent_name = self._agent_name or "unknown"

        # 1. Kill switch
        if agent_name != "unknown" and self._factory.is_frozen(agent_name):
            return GovernanceDecision(
                action="DENY",
                mode=self._factory.mode.value,
                agent_name=agent_name,
                tool_name=tool_name,
                reason_codes=["AGENT_FROZEN"],
                risk_score=100,
            )

        # 2. Budget limit (factory-level, always enforced)
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

        # 3. Policy evaluation
        args_str = str(tool_args) if not isinstance(tool_args, str) else tool_args

        for policy in self._factory.policies:
            if policy.type == "tool_allowlist":
                allowed = policy.config.get("allowed", [])
                if not any(fnmatch.fnmatch(tool_name, p) for p in allowed):
                    action = "DENY"
                    reason_codes.append("TOOL_NOT_ALLOWED")
                    risk_score = max(risk_score, 80)
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

    # ─── Helpers ─────────────────────────────────────────────────────────

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
            pattern = _PII_PATTERNS.get(cat)
            if pattern and pattern.search(text):
                found.append(cat)
        return found

    @staticmethod
    def _detect_secrets(text: str) -> bool:
        """Detect secret patterns in text."""
        return any(p.search(text) for p in _SECRET_PATTERNS)

    def _get_agent_name(self, context: "Context") -> str | None:
        """Extract agent name from context dependencies."""
        agent = context.dependencies.get(AGENT_CONTEXT_DEPENDENCY_KEY)
        return agent.name if agent is not None else None
