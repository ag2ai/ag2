# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
# SPDX-License-Identifier: Apache-2.0

"""TealTiger deterministic governance middleware.

Implements MiddlewareFactory pattern: TealTigerMiddleware is the factory (holds
long-lived state like frozen agents, decisions, cumulative cost) and creates
per-turn _TealTigerPerTurn instances that share a reference to the factory state.

This follows the same pattern as MetricsMiddleware/TelemetryMiddleware in
ag2/middleware/builtin/.
"""

import hashlib
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ag2 import Agent
    from ag2.annotations import Context
    from ag2.events import BaseEvent, ToolCallEvent
    from ag2.middleware.base import ToolExecution, ToolResultType

# Import tealtiger for governance evaluation
import tealtiger

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


class TealTigerMiddleware:
    """Deterministic governance middleware factory for AG2.

    Holds long-lived governance state (decisions, receipts, frozen agents, cost)
    across turns. Creates per-turn middleware instances that share this state.

    This is a MiddlewareFactory — pass it directly to the agent's middleware list.

    Args:
        policies: List of GovernancePolicy definitions.
        mode: Governance mode (OBSERVE, MONITOR, ENFORCE).
        cost_per_call: Estimated cost per tool call in USD (default: 0.002).
        on_decision: Optional callback invoked with each GovernanceDecision.
        on_receipt: Optional callback invoked with each TEECReceipt.

    Example:
        from ag2 import Agent
        from ag2.extensions.tealtiger import TealTigerMiddleware, GovernancePolicy

        governance = TealTigerMiddleware(
            policies=[
                GovernancePolicy.tool_allowlist(["search", "read_*"]),
                GovernancePolicy.pii_block(["ssn", "credit_card"]),
                GovernancePolicy.cost_limit(max_per_session=5.0),
            ],
            mode="ENFORCE",
        )

        agent = Agent(
            "assistant",
            config=OpenAIConfig("gpt-4o-mini"),
            tools=[search, read_file],
            middleware=[governance],
        )

        # Long-lived state accessible on the factory
        governance.freeze("assistant")
        print(governance.decisions)
        print(f"Total cost: ${governance.total_cost:.4f}")
    """

    def __init__(
        self,
        policies: list[GovernancePolicy] | None = None,
        mode: str | GovernanceMode = GovernanceMode.OBSERVE,
        cost_per_call: float = 0.002,
        on_decision: Callable[[GovernanceDecision], None] | None = None,
        on_receipt: Callable[[TEECReceipt], None] | None = None,
    ) -> None:
        self.policies = policies or []
        self.mode = GovernanceMode(mode) if isinstance(mode, str) else mode
        self.cost_per_call = cost_per_call
        self.on_decision = on_decision
        self.on_receipt = on_receipt

        # Long-lived state (survives across turns)
        self._decisions: list[GovernanceDecision] = []
        self._receipts: list[TEECReceipt] = []
        self._frozen_agents: set[str] = set()
        self._cumulative_cost: float = 0.0

        # Initialize TealTiger engine for policy evaluation
        self._engine = tealtiger.TealEngine(
            policies=[{"type": p.type, **p.config} for p in self.policies],
            mode=self.mode.value,
        )

        # Compute policy digest for receipts
        policy_str = str(sorted((p.type, str(p.config)) for p in self.policies))
        self._policy_digest = hashlib.sha256(policy_str.encode()).hexdigest()[:16]

    def __call__(self, event: BaseEvent, context: Context) -> BaseMiddleware:
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
    """Per-turn middleware instance — delegates governance to the factory's shared state."""

    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        factory: TealTigerMiddleware,
    ) -> None:
        super().__init__(event, context)
        self._factory = factory
        self._agent_name = self._get_agent_name(context)

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        """Evaluate governance policies before tool execution.

        Returns ToolErrorEvent for denied calls, or passes through to call_next.
        """
        start_time = time.perf_counter()
        tool_name = event.name
        tool_args = event.arguments if hasattr(event, "arguments") else str(event.args)

        # Evaluate governance
        decision = self._evaluate(tool_name, tool_args)
        eval_time_ms = (time.perf_counter() - start_time) * 1000
        decision.evaluation_time_ms = round(eval_time_ms, 3)

        # Record decision
        self._factory._decisions.append(decision)
        if self._factory.on_decision:
            self._factory.on_decision(decision)

        # Handle DENY in ENFORCE mode
        if self._factory.mode == GovernanceMode.ENFORCE and decision.action == "DENY":
            # Emit receipt for blocked execution
            self._emit_receipt(decision, execution_outcome="blocked")

            # Return ToolErrorEvent — the correct return type per the contract
            reason = ", ".join(decision.reason_codes)
            return ToolErrorEvent(
                call_id=event.call_id,
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

    # ─── Private evaluation logic ────────────────────────────────────────

    def _evaluate(self, tool_name: str, tool_args: Any) -> GovernanceDecision:
        """Evaluate governance policies against a tool call."""
        action = "ALLOW"
        reason_codes: list[str] = []
        risk_score = 0

        # Check agent freeze
        if self._agent_name and self._factory.is_frozen(self._agent_name):
            action = "DENY"
            reason_codes.append("AGENT_FROZEN")
            risk_score = 100
        else:
            # Delegate to TealTiger engine for policy evaluation
            engine_decision = self._factory._engine.evaluate(
                tool_name=tool_name,
                tool_args=str(tool_args),
                agent_id=self._agent_name or "unknown",
                cumulative_cost=self._factory._cumulative_cost,
            )
            action = engine_decision.get("action", "ALLOW")
            reason_codes = engine_decision.get("reason_codes", [])
            risk_score = engine_decision.get("risk_score", 0)

        return GovernanceDecision(
            action=action,
            mode=self._factory.mode.value,
            agent_name=self._agent_name or "unknown",
            tool_name=tool_name,
            reason_codes=reason_codes or (["POLICY_ALLOW"] if action == "ALLOW" else []),
            risk_score=risk_score,
            cumulative_cost=self._factory._cumulative_cost,
        )

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

    def _get_agent_name(self, context: Context) -> str | None:
        """Extract agent name from context dependencies."""
        agent: Agent | None = context.dependencies.get(AGENT_CONTEXT_DEPENDENCY_KEY)
        if agent is not None:
            return agent.name
        return None
