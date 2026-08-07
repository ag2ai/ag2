# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""TealTiger governance middleware for AG2.

Implements MiddlewareFactory pattern: long-lived factory holds shared state
(decisions, receipts, frozen agents, cumulative cost), per-turn instances
get a reference to that shared state.

No external dependencies beyond AG2 and the standard library.

Maintainer: @nagasatish007
"""

import fnmatch
import json
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ag2.annotations import Context
    from ag2.events import BaseEvent, ToolCallEvent
    from ag2.middleware.base import AgentTurn, ToolExecution

from ag2.events import ToolErrorEvent
from ag2.middleware import BaseMiddleware
from ag2.middleware.base import MiddlewareFactory, ToolResultType
from ag2.utils import AGENT_CONTEXT_DEPENDENCY_KEY

from .types import (
    GovernanceDecision,
    GovernanceMode,
    GovernancePolicy,
    PII_PATTERNS,
    SECRET_PATTERNS,
    TEECReceipt,
)


class _GovernanceState:
    """Shared mutable state across all per-turn middleware instances.

    Held by the factory, referenced by each per-turn instance.
    This ensures kill switch, budget, and audit trail persist across turns.
    """

    def __init__(self) -> None:
        self.decisions: list[GovernanceDecision] = []
        self.receipts: list[TEECReceipt] = []
        self.frozen_agents: set[str] = set()
        self.cumulative_cost: float = 0.0


class TealTigerMiddleware(MiddlewareFactory):
    """TealTiger governance middleware factory.

    Long-lived object that holds shared governance state (decisions, receipts,
    frozen agents, cumulative cost). Creates per-turn middleware instances.

    Args:
        mode: Governance mode (ENFORCE, MONITOR, OBSERVE).
        policies: List of governance policies to evaluate.
        budget_limit: Maximum USD spend per session (enforced even without
            a cost_limit policy).
        cost_per_call: Estimated cost per tool call in USD.
        on_decision: Optional callback for each governance decision.
        on_receipt: Optional callback for each TEEC receipt.

    Modes:
        ENFORCE: Blocks denied actions, returns ToolErrorEvent.
        MONITOR: Evaluates policies and records decisions, but allows all
            actions through regardless of the verdict.
        OBSERVE: Does not evaluate policies at all — only records that
            a tool call happened (passthrough with audit).

    Example::

        from ag2.extensions.tealtiger import TealTigerMiddleware, GovernancePolicy

        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            policies=[
                GovernancePolicy.tool_allowlist(["search", "read_file"]),
                GovernancePolicy.pii_block(["ssn", "credit_card"]),
                GovernancePolicy.cost_limit(max_per_session=5.0),
            ],
        )
        # Register with agent via Middleware(gov)
    """

    def __init__(
        self,
        mode: GovernanceMode = GovernanceMode.ENFORCE,
        policies: list[GovernancePolicy] | None = None,
        budget_limit: float = float("inf"),
        cost_per_call: float = 0.002,
        on_decision: Callable[[GovernanceDecision], None] | None = None,
        on_receipt: Callable[[TEECReceipt], None] | None = None,
    ) -> None:
        self.mode = mode
        self.policies = policies or []
        self.budget_limit = budget_limit
        self.cost_per_call = cost_per_call
        self.on_decision = on_decision
        self.on_receipt = on_receipt
        self._state = _GovernanceState()

    @property
    def decisions(self) -> list[GovernanceDecision]:
        """All governance decisions made by this middleware."""
        return self._state.decisions

    @property
    def receipts(self) -> list[TEECReceipt]:
        """All TEEC receipts produced."""
        return self._state.receipts

    def freeze(self, agent_name: str = "*") -> None:
        """Activate kill switch for an agent (or all agents with '*')."""
        self._state.frozen_agents.add(agent_name)

    def unfreeze(self, agent_name: str = "*") -> None:
        """Deactivate kill switch for an agent."""
        self._state.frozen_agents.discard(agent_name)

    def __call__(
        self, event: "BaseEvent", context: "Context"
    ) -> "BaseMiddleware":
        """Create a per-turn middleware instance with shared state."""
        return _TealTigerTurnMiddleware(event, context, self)


class _TealTigerTurnMiddleware(BaseMiddleware):
    """Per-turn middleware instance. References factory's shared state."""

    def __init__(
        self,
        event: "BaseEvent",
        context: "Context",
        factory: TealTigerMiddleware,
    ) -> None:
        super().__init__(event, context)
        self._factory = factory
        self._state = factory._state
        self._agent_name = self._get_agent_name(context)

    async def on_turn(
        self,
        call_next: "AgentTurn",
        event: "BaseEvent",
        context: "Context",
    ) -> "ToolResultType":
        """Kill switch enforcement at the turn level.

        If the agent is frozen:
        - ENFORCE mode: returns a ToolErrorEvent blocking the entire turn.
        - MONITOR mode: records a DENY decision but allows the turn.
        - OBSERVE mode: passes through without evaluation.
        """
        agent_name = self._agent_name or "unknown"

        # OBSERVE mode: no evaluation, pass through
        if self._factory.mode == GovernanceMode.OBSERVE:
            return await call_next(event, context)

        # Check kill switch
        if self._is_frozen(agent_name):
            decision = GovernanceDecision(
                action="DENY",
                reason=f"Agent '{agent_name}' is frozen (kill switch active)",
                reason_codes=["KILL_SWITCH"],
                risk_score=1.0,
                tool_name="*",
                agent_name=agent_name,
                evaluation_time_ms=0.0,
            )
            self._state.decisions.append(decision)
            if self._factory.on_decision:
                self._factory.on_decision(decision)

            if self._factory.mode == GovernanceMode.ENFORCE:
                return ToolErrorEvent.from_call(
                    event,
                    error=Exception(
                        f"[GOVERNANCE DENIED] Agent '{agent_name}' is frozen "
                        f"(kill switch active). All actions blocked."
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
        """Evaluate governance before tool execution.

        Returns ToolErrorEvent on DENY (ENFORCE mode).
        Logs and passes through in MONITOR mode.
        Passes through without evaluation in OBSERVE mode.
        """
        start_time = time.perf_counter()
        tool_name = event.name
        arguments = self._parse_arguments(event)

        # OBSERVE mode: no evaluation, just pass through and record receipt
        if self._factory.mode == GovernanceMode.OBSERVE:
            result = await call_next(event, context)
            self._emit_observe_receipt(tool_name, result)
            return result

        # Evaluate governance (ENFORCE and MONITOR modes)
        decision = self._evaluate(tool_name, arguments, start_time)
        self._state.decisions.append(decision)

        if self._factory.on_decision:
            self._factory.on_decision(decision)

        # Enforce denial (ENFORCE mode only)
        if decision.action == "DENY" and self._factory.mode == GovernanceMode.ENFORCE:
            receipt = TEECReceipt(
                decision_id=decision.decision_id,
                agent_name=self._agent_name or "unknown",
                tool_name=tool_name,
                action="DENY",
                timestamp_ms=time.time() * 1000,
                execution_outcome="blocked",
            )
            self._state.receipts.append(receipt)
            if self._factory.on_receipt:
                self._factory.on_receipt(receipt)

            return ToolErrorEvent.from_call(
                event,
                error=Exception(
                    f"[GOVERNANCE DENIED] Tool '{tool_name}' blocked: "
                    f"{decision.reason}"
                ),
            )

        # Track cost
        self._state.cumulative_cost += self._factory.cost_per_call

        # Execute the tool
        result = await call_next(event, context)

        # Emit receipt with execution outcome
        receipt = TEECReceipt(
            decision_id=decision.decision_id,
            agent_name=self._agent_name or "unknown",
            tool_name=tool_name,
            action=decision.action,
            timestamp_ms=time.time() * 1000,
            execution_outcome="executed",
        )
        self._state.receipts.append(receipt)
        if self._factory.on_receipt:
            self._factory.on_receipt(receipt)

        return result

    def _evaluate(
        self, tool_name: str, arguments: dict[str, Any], start_time: float
    ) -> GovernanceDecision:
        """Evaluate all governance policies for a tool call."""
        agent_name = self._agent_name or "unknown"

        # 1. Kill switch
        if self._is_frozen(agent_name):
            return GovernanceDecision(
                action="DENY",
                reason=f"Agent '{agent_name}' is frozen (kill switch active)",
                reason_codes=["KILL_SWITCH"],
                risk_score=1.0,
                tool_name=tool_name,
                agent_name=agent_name,
                evaluation_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # 2. Budget check (factory-level limit, always enforced)
        if self._state.cumulative_cost >= self._factory.budget_limit:
            return GovernanceDecision(
                action="DENY",
                reason=(
                    f"Budget exceeded: ${self._state.cumulative_cost:.4f} "
                    f">= ${self._factory.budget_limit:.2f}"
                ),
                reason_codes=["BUDGET_EXCEEDED"],
                risk_score=0.8,
                tool_name=tool_name,
                agent_name=agent_name,
                evaluation_time_ms=(time.perf_counter() - start_time) * 1000,
                cumulative_cost=self._state.cumulative_cost,
            )

        # 3. Policy-level cost limit
        for policy in self._factory.policies:
            if policy.type == "cost_limit":
                limit = policy.config.get("max_per_session", self._factory.budget_limit)
                if self._state.cumulative_cost >= limit:
                    return GovernanceDecision(
                        action="DENY",
                        reason=(
                            f"Budget exceeded: ${self._state.cumulative_cost:.4f} "
                            f">= ${limit:.2f}"
                        ),
                        reason_codes=["BUDGET_EXCEEDED"],
                        risk_score=0.8,
                        tool_name=tool_name,
                        agent_name=agent_name,
                        evaluation_time_ms=(time.perf_counter() - start_time) * 1000,
                        cumulative_cost=self._state.cumulative_cost,
                    )

        # 4. Tool allowlist
        for policy in self._factory.policies:
            if policy.type == "tool_allowlist":
                allowed = policy.config.get("allowed", [])
                if not self._matches_patterns(tool_name, allowed):
                    return GovernanceDecision(
                        action="DENY",
                        reason=f"Tool '{tool_name}' not in allowlist",
                        reason_codes=["TOOL_NOT_ALLOWED"],
                        risk_score=0.9,
                        tool_name=tool_name,
                        agent_name=agent_name,
                        evaluation_time_ms=(time.perf_counter() - start_time) * 1000,
                    )

        # 5. PII scan
        args_str = str(arguments)
        for policy in self._factory.policies:
            if policy.type == "pii_block":
                categories = policy.config.get("categories", [])
                findings = self._scan_pii(args_str, categories)
                if findings:
                    return GovernanceDecision(
                        action="DENY",
                        reason=f"PII detected in tool arguments: {len(findings)} finding(s)",
                        reason_codes=["PII_DETECTED"],
                        risk_score=0.85,
                        tool_name=tool_name,
                        agent_name=agent_name,
                        findings=findings,
                        evaluation_time_ms=(time.perf_counter() - start_time) * 1000,
                    )

        # 6. Secret scan
        for policy in self._factory.policies:
            if policy.type == "secret_block":
                if self._scan_secrets(args_str):
                    return GovernanceDecision(
                        action="DENY",
                        reason="Secret detected in tool arguments",
                        reason_codes=["SECRET_DETECTED"],
                        risk_score=0.95,
                        tool_name=tool_name,
                        agent_name=agent_name,
                        evaluation_time_ms=(time.perf_counter() - start_time) * 1000,
                    )

        # All checks passed
        return GovernanceDecision(
            action="ALLOW",
            reason="All governance checks passed",
            tool_name=tool_name,
            agent_name=agent_name,
            evaluation_time_ms=(time.perf_counter() - start_time) * 1000,
            cumulative_cost=self._state.cumulative_cost,
        )

    def _emit_observe_receipt(self, tool_name: str, result: Any) -> None:
        """Emit a minimal receipt in OBSERVE mode (no policy evaluation)."""
        decision = GovernanceDecision(
            action="ALLOW",
            reason="OBSERVE mode — no policy evaluation",
            tool_name=tool_name,
            agent_name=self._agent_name or "unknown",
        )
        self._state.decisions.append(decision)

        receipt = TEECReceipt(
            decision_id=decision.decision_id,
            agent_name=self._agent_name or "unknown",
            tool_name=tool_name,
            action="ALLOW",
            timestamp_ms=time.time() * 1000,
            execution_outcome="executed",
        )
        self._state.receipts.append(receipt)
        if self._factory.on_receipt:
            self._factory.on_receipt(receipt)

    def _is_frozen(self, agent_name: str) -> bool:
        """Check if agent is frozen (kill switch)."""
        return "*" in self._state.frozen_agents or agent_name in self._state.frozen_agents

    @staticmethod
    def _parse_arguments(event: "ToolCallEvent") -> dict[str, Any]:
        """Parse tool arguments from the event.

        ToolCallEvent.arguments is a JSON string; parse it into a dict.
        Falls back to empty dict on parse failure.
        """
        raw = event.arguments
        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _matches_patterns(name: str, patterns: list[str]) -> bool:
        """Check if name matches any glob pattern in the list."""
        return any(fnmatch.fnmatch(name, p) for p in patterns)

    @staticmethod
    def _scan_pii(text: str, categories: list[str]) -> list[dict[str, Any]]:
        """Scan text for PII patterns."""
        findings = []
        for cat, pattern in PII_PATTERNS.items():
            if cat not in categories:
                continue
            for match in pattern.finditer(text):
                findings.append({"type": "pii", "category": cat, "start": match.start()})
        return findings

    @staticmethod
    def _scan_secrets(text: str) -> bool:
        """Check if text contains any secret patterns."""
        return any(pattern.search(text) for pattern in SECRET_PATTERNS)

    def _get_agent_name(self, context: "Context") -> str | None:
        """Get agent name from context dependencies."""
        agent = context.dependencies.get(AGENT_CONTEXT_DEPENDENCY_KEY)
        if agent is not None:
            return agent.name
        return None
