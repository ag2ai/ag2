# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Governance middleware -- budget enforcement, circuit breaking, and tool policy.

Tracks per-turn cost from ``ModelResponse.usage``, blocks LLM calls when
budget is exhausted, isolates agents after consecutive failures, and
blocks disallowed tool calls.

Inspired by veronica-core (https://github.com/amabito/veronica-core).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ModelResponse, ToolCall, ToolError
from autogen.beta.middleware.base import (
    BaseMiddleware,
    LLMCall,
    MiddlewareFactory,
    ToolExecution,
    ToolResultType,
)

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[GOVERNANCE]"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class GovernanceConfig:
    """Configuration for GovernanceMiddleware.

    Note on ``allowed_tools``: ``None`` means no restriction (all tools
    allowed). An empty list ``[]`` means *no* tools are allowed -- every
    tool call will be blocked. This distinction is intentional.
    """

    # Budget
    max_cost_usd: float = 10.0
    cost_per_1k_input_tokens: float = 0.00025
    cost_per_1k_output_tokens: float = 0.0005

    # Circuit breaker
    failure_threshold: int = 5
    recovery_timeout_s: float = 60.0

    # Tool policy
    blocked_tools: list[str] = field(default_factory=list)
    allowed_tools: list[str] | None = None

    # Degradation
    degradation_threshold: float = 0.8
    fallback_model: str | None = None
    disable_tools_on_degrade: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal state (shared across turns via the factory)
# ---------------------------------------------------------------------------


class _BudgetTracker:
    def __init__(self, limit_usd: float, cost_1k_in: float, cost_1k_out: float) -> None:
        self._limit = limit_usd
        self._cost_1k_in = cost_1k_in
        self._cost_1k_out = cost_1k_out
        self._spent: float = 0.0
        self._lock = threading.Lock()

    def check(self) -> tuple[bool, str]:
        with self._lock:
            if self._limit > 0 and self._spent >= self._limit:
                return False, f"Budget exhausted: ${self._spent:.4f} / ${self._limit:.4f}"
            return True, ""

    def record(self, usage: dict[str, float]) -> float:
        # Prefer prompt_tokens/completion_tokens (OpenAI), fall back to
        # input_tokens/output_tokens (Gemini). Use `in` check, not `or`,
        # to avoid masking legitimate zero values.
        input_tokens = (
            usage["prompt_tokens"]
            if "prompt_tokens" in usage
            else usage.get("input_tokens", 0)
        )
        output_tokens = (
            usage["completion_tokens"]
            if "completion_tokens" in usage
            else usage.get("output_tokens", 0)
        )
        # Clamp to non-negative; reject None/NaN/Inf.
        if input_tokens is None or not math.isfinite(input_tokens):
            input_tokens = 0
        if output_tokens is None or not math.isfinite(output_tokens):
            output_tokens = 0
        cost = (
            max(input_tokens, 0) / 1000.0 * self._cost_1k_in
            + max(output_tokens, 0) / 1000.0 * self._cost_1k_out
        )
        with self._lock:
            self._spent += cost
        return cost

    def utilization(self) -> float:
        with self._lock:
            if self._limit <= 0:
                return 0.0
            return self._spent / self._limit

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    @property
    def limit(self) -> float:
        return self._limit


class _CircuitBreaker:
    def __init__(self, threshold: int, timeout_s: float) -> None:
        self._threshold = threshold
        self._timeout_s = timeout_s
        self._failures: int = 0
        self._opened_at: float | None = None
        self._probe_in_flight: bool = False
        self._lock = threading.Lock()

    def check_and_claim(self) -> tuple[CircuitState, bool]:
        """Atomically check state and claim probe if HALF_OPEN.

        Returns (state, probe_claimed). If state is HALF_OPEN and probe
        was claimed, probe_claimed is True. Caller MUST call
        record_success() or record_failure() to release the probe.
        """
        with self._lock:
            state = self._state_unlocked()
            if state == CircuitState.HALF_OPEN and not self._probe_in_flight:
                self._probe_in_flight = True
                return state, True
            return state, False

    def release_probe(self) -> None:
        """Release probe slot without recording success or failure."""
        with self._lock:
            self._probe_in_flight = False

    def record_failure(self) -> CircuitState:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                # Refresh opened_at on probe failure so recovery timer restarts.
                self._opened_at = time.monotonic()
            self._probe_in_flight = False
            return self._state_unlocked()

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def get_state(self) -> CircuitState:
        with self._lock:
            return self._state_unlocked()

    def _state_unlocked(self) -> CircuitState:
        if self._failures < self._threshold:
            return CircuitState.CLOSED
        if self._opened_at is None:
            return CircuitState.CLOSED
        if time.monotonic() - self._opened_at >= self._timeout_s:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN


class _ToolPolicy:
    def __init__(self, blocked: list[str], allowed: list[str] | None) -> None:
        self._blocked = frozenset(blocked)
        self._allowed = frozenset(allowed) if allowed is not None else None

    def check(self, tool_name: str) -> tuple[bool, str]:
        if tool_name in self._blocked:
            return False, f"Tool '{tool_name}' is blocked by governance policy"
        if self._allowed is not None and tool_name not in self._allowed:
            return False, f"Tool '{tool_name}' is not in the allowed list"
        return True, ""


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


def _make_tool_error(event: ToolCall, reason: str) -> ToolError:
    """Create a ToolError with explicit content to avoid format_exc() trap."""
    err = ToolError(
        parent_id=getattr(event, "id", ""),
        name=event.name,
        error=RuntimeError(reason),
    )
    err.content = reason
    return err


class GovernanceMiddleware(MiddlewareFactory):
    """Budget enforcement, circuit breaking, and tool policy for AG2 agents.

    Register on an agent via the ``middleware`` parameter::

        from autogen.beta.middleware.builtin.governance import (
            GovernanceConfig,
            GovernanceMiddleware,
        )

        agent = ConversableAgent(
            ...,
            middleware=[GovernanceMiddleware(GovernanceConfig(max_cost_usd=1.0))],
        )

    See: https://github.com/amabito/veronica-core
    """

    def __init__(self, config: GovernanceConfig | None = None) -> None:
        self._config = config or GovernanceConfig()
        self._budget = _BudgetTracker(
            limit_usd=self._config.max_cost_usd,
            cost_1k_in=self._config.cost_per_1k_input_tokens,
            cost_1k_out=self._config.cost_per_1k_output_tokens,
        )
        self._cb = _CircuitBreaker(
            threshold=self._config.failure_threshold,
            timeout_s=self._config.recovery_timeout_s,
        )
        self._policy = _ToolPolicy(
            blocked=self._config.blocked_tools,
            allowed=self._config.allowed_tools,
        )
        self._degraded: bool = False
        self._total_llm_calls: int = 0
        self._total_tool_calls: int = 0
        self._stats_lock = threading.Lock()

    def __call__(self, event: BaseEvent, context: Context) -> BaseMiddleware:
        return _GovernanceMiddlewareInstance(
            event,
            context,
            budget=self._budget,
            cb=self._cb,
            policy=self._policy,
            config=self._config,
            owner=self,
        )

    @property
    def budget(self) -> _BudgetTracker:
        return self._budget

    @property
    def circuit_breaker(self) -> _CircuitBreaker:
        return self._cb


def _blocked_response() -> ModelResponse:
    """Fresh blocked response per call to avoid shared mutable state."""
    return ModelResponse(message=None, usage={})


class _GovernanceMiddlewareInstance(BaseMiddleware):
    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        *,
        budget: _BudgetTracker,
        cb: _CircuitBreaker,
        policy: _ToolPolicy,
        config: GovernanceConfig,
        owner: GovernanceMiddleware,
    ) -> None:
        super().__init__(event, context)
        self._budget = budget
        self._cb = cb
        self._policy = policy
        self._config = config
        self._owner = owner

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        # Circuit breaker -- atomic check + probe claim
        state, probe_claimed = self._cb.check_and_claim()
        if state == CircuitState.OPEN:
            logger.warning("%s Circuit OPEN -- blocking LLM call.", _LOG_PREFIX)
            return _blocked_response()
        if state == CircuitState.HALF_OPEN:
            if not probe_claimed:
                logger.warning("%s Circuit HALF_OPEN -- probe in flight.", _LOG_PREFIX)
                return _blocked_response()
            logger.info("%s Circuit HALF_OPEN -- allowing probe.", _LOG_PREFIX)

        # Budget check -- release probe if budget blocks
        allowed, reason = self._budget.check()
        if not allowed:
            if probe_claimed:
                self._cb.release_probe()
            logger.warning("%s %s", _LOG_PREFIX, reason)
            return _blocked_response()

        # Degradation: log once per factory lifetime
        utilization = self._budget.utilization()
        if (
            self._config.fallback_model
            and utilization >= self._config.degradation_threshold
        ):
            with self._owner._stats_lock:
                already = self._owner._degraded
                self._owner._degraded = True
            if not already:
                logger.info(
                    "%s Budget at %.0f%% -- degraded to %s.",
                    _LOG_PREFIX,
                    utilization * 100,
                    self._config.fallback_model,
                )

        with self._owner._stats_lock:
            self._owner._total_llm_calls += 1

        try:
            response = await call_next(events, context)
        except Exception:
            new_state = self._cb.record_failure()
            if new_state == CircuitState.OPEN:
                logger.warning(
                    "%s Circuit tripped to OPEN after %d failures.",
                    _LOG_PREFIX,
                    self._config.failure_threshold,
                )
            raise

        # Record success + cost
        self._cb.record_success()
        if response.usage:
            cost = self._budget.record(response.usage)
            logger.debug("%s LLM call cost: $%.6f", _LOG_PREFIX, cost)

        return response

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: ToolCall,
        context: Context,
    ) -> ToolResultType:
        tool_name = event.name

        # Policy check
        ok, reason = self._policy.check(tool_name)
        if not ok:
            logger.warning("%s %s", _LOG_PREFIX, reason)
            return _make_tool_error(event, reason)

        # Degradation: disable expensive tools
        utilization = self._budget.utilization()
        if (
            tool_name in self._config.disable_tools_on_degrade
            and utilization >= self._config.degradation_threshold
        ):
            msg = f"Tool '{tool_name}' disabled -- budget at {utilization * 100:.0f}%"
            logger.info("%s %s", _LOG_PREFIX, msg)
            return _make_tool_error(event, msg)

        with self._owner._stats_lock:
            self._owner._total_tool_calls += 1

        return await call_next(event, context)
