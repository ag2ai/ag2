# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""TealTiger governance middleware for AG2.

Implements MiddlewareFactory pattern: long-lived factory holds shared state
(decisions, receipts, frozen agents, cumulative cost), per-turn instances
get a reference to that shared state.

Maintainer: @nagasatish007
"""

import fnmatch
import time
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ag2.annotations import Context
    from ag2.events import BaseEvent, ToolCallEvent
    from ag2.middleware.base import AgentTurn, LLMCall, ToolExecution, ToolResultType

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
        budget_limit: Maximum USD spend per session.
        on_decision: Optional callback for each governance decision.
        on_receipt: Optional callback for each TEEC receipt.

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
        on_decision: Callable[[GovernanceDecision], None] | None = None,
        on_receipt: Callable[[TEECReceipt], None] | None = None,
    ) -> None:
        self.mode = mode
        self.policies = policies or []
        self.budget_limit = budget_limit
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

    async def on_tool_execution(
        self,
        call_next: "ToolExecution",
        event: "ToolCallEvent",
        context: "Context",
    ) -> "ToolResultType":
        """Evaluate governance before tool execution.

        Returns ToolErrorEvent on DENY (ENFORCE mode).
        Logs and passes through in MONITOR/OBSERVE mode.
        """
        start_time = time.perf_counter()
        tool_name = event.name
        arguments = event.arguments if event.arguments else {}

        # Evaluate governance
        decision = self._evaluate(tool_name, arguments, start_time)
        self._state.decisions.append(decision)

        if self._factory.on_decision:
            self._factory.on_decision(decision)

        # Emit receipt
        receipt = TEECReceipt(
            decision_id=decision.decision_id,
            agent_name=self._agent_name or "unknown",
            tool_name=tool_name,
            action=decision.action,
            timestamp_ms=time.time() * 1000,
        )
        self._state.receipts.append(receipt)
        if self._factory.on_receipt:
            self._factory.on_receipt(receipt)

        # Enforce denial
        if decision.action == "DENY":
            if self._factory.mode == GovernanceMode.ENFORCE:
                return ToolErrorEvent.from_call(
                    event,
                    error=Exception(
                        f"[GOVERNANCE DENIED] Tool '{tool_name}' blocked: "
                        f"{decision.reason}"
                    ),
                )
            # MONITOR/OBSERVE: log but allow through

        # Track cost (simplified: estimate per tool call)
        self._state.cumulative_cost += 0.002

        # Execute the tool
        result = await call_next(event, context)
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

        # 2. Budget check
        for policy in self._factory.policies:
            if policy.type == "cost_limit":
                limit = policy.config.get("max_per_session", self._factory.budget_limit)
                if self._state.cumulative_cost >= limit:
                    return GovernanceDecision(
                        action="DENY",
                        reason=f"Budget exceeded: ${self._state.cumulative_cost:.4f} >= ${limit:.2f}",
                        reason_codes=["BUDGET_EXCEEDED"],
                        risk_score=0.8,
                        tool_name=tool_name,
                        agent_name=agent_name,
                        evaluation_time_ms=(time.perf_counter() - start_time) * 1000,
                        cumulative_cost=self._state.cumulative_cost,
                    )

        # 3. Tool allowlist
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

        # 4. PII scan
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

        # 5. Secret scan
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

    def _is_frozen(self, agent_name: str) -> bool:
        """Check if agent is frozen (kill switch)."""
        return "*" in self._state.frozen_agents or agent_name in self._state.frozen_agents

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
