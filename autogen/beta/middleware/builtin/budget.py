# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Budget middleware -- raises when cumulative cost exceeds a configured limit."""

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ModelResponse, Usage
from autogen.beta.middleware.base import BaseMiddleware, LLMCall, MiddlewareFactory

# Token key variants per provider
_INPUT_KEYS = ("prompt_tokens", "prompt_token_count", "input_tokens")
_OUTPUT_KEYS = ("completion_tokens", "candidates_token_count", "output_tokens")
_SPENT_KEY = "autogen.beta.middleware.budget.spent"


class BudgetExceededError(Exception):
    """Raised when a BudgetMiddleware blocks an LLM call because the budget is exhausted.

    Attributes:
        spent: USD already consumed before this call would have run.
        limit: Configured budget cap in USD.
    """

    def __init__(self, spent: float, limit: float) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(f"budget exceeded: spent ${spent:.4f} of ${limit:.2f} USD limit")


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


class _BudgetTracker:
    """Thread-safe cumulative cost tracker."""

    def __init__(self, config: BudgetConfig, initial_spent: float = 0.0) -> None:
        self._config = config
        self._spent: float = _safe_float(initial_spent)
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

    def record(self, usage: Usage | dict[str, object] | object) -> None:
        """Accumulate cost from a provider usage payload."""
        input_tokens = 0.0
        output_tokens = 0.0

        for key in _INPUT_KEYS:
            if isinstance(usage, dict):
                if key not in usage:
                    continue
                input_tokens = _safe_float(usage[key])
                break

            value = getattr(usage, key, None)
            if value is not None:
                input_tokens = _safe_float(value)
                break

        for key in _OUTPUT_KEYS:
            if isinstance(usage, dict):
                if key not in usage:
                    continue
                output_tokens = _safe_float(usage[key])
                break

            value = getattr(usage, key, None)
            if value is not None:
                output_tokens = _safe_float(value)
                break

        cost = (input_tokens / 1000.0) * self._config.cost_per_1k_input_tokens + (
            output_tokens / 1000.0
        ) * self._config.cost_per_1k_output_tokens

        with self._lock:
            self._spent += cost


class BudgetMiddleware(MiddlewareFactory):
    """Factory that tracks budget cumulatively per conversation context.

    Each time this factory is invoked by the agent runtime, it seeds a local
    ``_BudgetTracker`` from ``context.variables[_SPENT_KEY]``. That keeps the
    mutable tracker instance request-local while persisting cumulative spend as
    a serializable float on the conversation context. Separate ``Context``
    objects therefore stay isolated, while repeated ``ask()`` calls that reuse
    the same context continue accumulating spend.

    Example::

        mw = BudgetMiddleware(
            max_cost_usd=1.0,
            cost_per_1k_input_tokens=0.30,
            cost_per_1k_output_tokens=0.60,
        )
        agent = MyAgent(middleware=[mw])

    When the configured budget is exhausted, ``BudgetExceededError`` is
    raised so callers see an explicit failure rather than a silent empty
    response.
    """

    def __init__(
        self,
        max_cost_usd: float = 0.0,
        cost_per_1k_input_tokens: float = 0.0,
        cost_per_1k_output_tokens: float = 0.0,
    ) -> None:
        self._config = BudgetConfig(
            max_cost_usd=max_cost_usd,
            cost_per_1k_input_tokens=cost_per_1k_input_tokens,
            cost_per_1k_output_tokens=cost_per_1k_output_tokens,
        )

    def __call__(self, event: "BaseEvent", context: "Context") -> "BaseMiddleware":
        prior_spent = float(context.variables.get(_SPENT_KEY, 0.0))
        tracker = _BudgetTracker(self._config, initial_spent=prior_spent)
        return _BudgetMiddlewareInstance(event, context, tracker)


class _BudgetMiddlewareInstance(BaseMiddleware):
    """Per-conversation middleware instance that enforces a budget."""

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
            raise BudgetExceededError(spent=self._budget.spent, limit=self._budget.limit)

        response = await call_next(events, context)
        self._budget.record(response.usage)
        context.variables[_SPENT_KEY] = self._budget.spent
        return response
