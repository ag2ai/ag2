# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Budget middleware -- blocks LLM calls when cumulative cost exceeds a configured limit."""

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ModelResponse
from autogen.beta.middleware.base import BaseMiddleware, LLMCall, MiddlewareFactory

# Token key variants per provider
_INPUT_KEYS = ("prompt_tokens", "prompt_token_count", "input_tokens")
_OUTPUT_KEYS = ("completion_tokens", "candidates_token_count", "output_tokens")


def _safe_float(value: object) -> float:
    """Coerce a token count to a non-negative float, returning 0.0 on failure."""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return max(0.0, result)


@dataclass
class BudgetConfig:
    """Configuration for BudgetMiddleware.

    Args:
        max_cost_usd: Maximum cumulative cost in USD. 0 means unlimited.
        cost_per_1k_input_tokens: Cost in USD per 1000 input tokens.
        cost_per_1k_output_tokens: Cost in USD per 1000 output tokens.
    """

    max_cost_usd: float = 0.0
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_cost_usd) or self.max_cost_usd < 0:
            raise ValueError("max_cost_usd must be a non-negative finite number")
        for name, value in (
            ("cost_per_1k_input_tokens", self.cost_per_1k_input_tokens),
            ("cost_per_1k_output_tokens", self.cost_per_1k_output_tokens),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative finite number")


class _BudgetTracker:
    """Thread-safe cumulative cost tracker."""

    def __init__(self, config: BudgetConfig) -> None:
        self._config = config
        self._spent: float = 0.0
        self._lock = threading.Lock()

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    @property
    def limit(self) -> float:
        return self._config.max_cost_usd

    def check(self) -> bool:
        """Return True if the next call is allowed (budget not exhausted)."""
        if self._config.max_cost_usd == 0:
            return True
        with self._lock:
            return self._spent < self._config.max_cost_usd

    def utilization(self) -> float:
        """Return fraction of budget consumed (0.0 to 1.0+).

        Returns 0.0 when max_cost_usd is 0 (unlimited).
        """
        if self._config.max_cost_usd == 0:
            return 0.0
        with self._lock:
            return self._spent / self._config.max_cost_usd

    def record(self, usage: dict[str, float]) -> None:
        """Accumulate cost from a usage dict returned by the LLM provider."""
        input_tokens = 0.0
        output_tokens = 0.0

        for key in _INPUT_KEYS:
            if key in usage:
                input_tokens = _safe_float(usage[key])
                break

        for key in _OUTPUT_KEYS:
            if key in usage:
                output_tokens = _safe_float(usage[key])
                break

        cost = (input_tokens / 1000.0) * self._config.cost_per_1k_input_tokens + (
            output_tokens / 1000.0
        ) * self._config.cost_per_1k_output_tokens

        with self._lock:
            self._spent += cost


def _blocked_response() -> ModelResponse:
    """Return a minimal ModelResponse indicating a blocked call."""
    return ModelResponse(message=None, usage={})


class BudgetMiddleware(MiddlewareFactory):
    """Factory that creates per-invocation budget middleware instances.

    A single BudgetMiddleware shares one _BudgetTracker across all instances
    it creates, so cost accumulates across the lifetime of the factory.

    Example::

        config = BudgetConfig(
            max_cost_usd=1.0,
            cost_per_1k_input_tokens=0.30,
            cost_per_1k_output_tokens=0.60,
        )
        mw = BudgetMiddleware(config)
        agent = MyAgent(middleware=[mw])
    """

    def __init__(self, config: BudgetConfig | None = None) -> None:
        self._config = config or BudgetConfig()
        self._budget = _BudgetTracker(self._config)

    def __call__(self, event: "BaseEvent", context: "Context") -> "BaseMiddleware":
        return _BudgetMiddlewareInstance(event, context, self._budget)


class _BudgetMiddlewareInstance(BaseMiddleware):
    """Per-invocation middleware instance that enforces the shared budget."""

    def __init__(
        self,
        event: "BaseEvent",
        context: "Context",
        budget: _BudgetTracker,
    ) -> None:
        super().__init__(event, context)
        self._budget = budget

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        if not self._budget.check():
            return _blocked_response()

        response = await call_next(events, context)
        self._budget.record(response.usage)
        return response
