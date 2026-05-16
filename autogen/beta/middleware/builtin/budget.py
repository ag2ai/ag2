# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Budget middleware -- raises when cumulative cost exceeds a configured limit."""

import math
import threading
import weakref
from collections.abc import Sequence
from dataclasses import dataclass

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ModelResponse, Usage
from autogen.beta.middleware.base import BaseMiddleware, LLMCall, MiddlewareFactory

# Token key variants per provider
_INPUT_KEYS = ("prompt_tokens", "prompt_token_count", "input_tokens")
_OUTPUT_KEYS = ("completion_tokens", "candidates_token_count", "output_tokens")
_SPENT_KEY = "autogen.beta.middleware.budget.spent"
_CONTEXT_LOCK_KEY = f"{_SPENT_KEY}.context_lock_key"

_context_locks: weakref.WeakKeyDictionary[object, threading.Lock] = weakref.WeakKeyDictionary()
_registry_lock = threading.Lock()


class _ContextLockKey:
    """Weak-referenceable identity key for context objects that cannot be weak-keyed directly."""


def _get_context_lock(context: object) -> threading.Lock:
    """Return the lock used to serialize budget updates for a context."""
    with _registry_lock:
        try:
            lock = _context_locks.get(context)
        except TypeError:
            lock_key = _get_context_lock_key(context)
            lock = _context_locks.get(lock_key)
            if lock is None:
                lock = threading.Lock()
                _context_locks[lock_key] = lock
            return lock

        if lock is None:
            lock = threading.Lock()
            _context_locks[context] = lock
        return lock


def _get_context_lock_key(context: object) -> _ContextLockKey:
    dependencies = getattr(context, "dependencies", None)
    if not isinstance(dependencies, dict):
        return _ContextLockKey()

    # Store the private, non-serializable lock key in dependencies rather than
    # variables. Dependencies are shallow-copied to subagents but not synced
    # back, unlike variables which are synced after subagent completion.
    lock_key = dependencies.get(_CONTEXT_LOCK_KEY)
    if not isinstance(lock_key, _ContextLockKey):
        lock_key = _ContextLockKey()
        dependencies[_CONTEXT_LOCK_KEY] = lock_key
    return lock_key


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

    def context_check(self, context: object) -> bool:
        """Return True after syncing the tracker from the shared context.

        The context value is the latest spend known across middleware
        instances for the same conversation. Syncing under the per-context
        lock keeps admission checks from using a stale tracker-local value
        when multiple factories were seeded from the same earlier context
        state.
        """
        lock = _get_context_lock(context)
        with lock:
            variables = getattr(context, "variables", {})
            context_spent = _safe_float(variables.get(_SPENT_KEY, 0.0)) if isinstance(variables, dict) else 0.0
            with self._lock:
                self._spent = max(self._spent, context_spent)
                spent = self._spent

        if self._config.max_cost_usd == 0:
            return True
        return spent < self._config.max_cost_usd

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
        cost = self._compute_cost(usage)

        with self._lock:
            self._spent += cost

    def atomic_update_context(self, usage: Usage | dict[str, object] | object, context: object) -> None:
        """Accumulate cost atomically, writing through to the shared context.

        Two concurrent ask() calls on the same Context could both read the
        same stale ``_SPENT_KEY`` value and clobber each other's writes.
        The per-context lock ensures the full read-delta-write cycle is
        atomic: only the delta (this call's cost) is computed independently;
        the new total is read from the context under the lock so concurrent
        callers see each other's prior writes.

        Within a single serial chain of calls (same instance, different
        context objects passed to on_llm_call), the tracker's ``_spent``
        accumulates locally from 0.  The context writeback uses the max of
        the tracker-local total and any value already present in the context,
        so both patterns are served correctly.
        """
        cost = self._compute_cost(usage)
        lock = _get_context_lock(context)
        with lock:
            context_prior = _safe_float(getattr(context, "variables", {}).get(_SPENT_KEY, 0.0))
            # Compute the new tracker-local total first (adds this call's cost).
            with self._lock:
                self._spent += cost
                tracker_total = self._spent
            # The authoritative context value is whichever is higher: what we
            # accumulated locally or what a concurrent caller already wrote.
            new_spent = max(tracker_total, context_prior + cost)
            context.variables[_SPENT_KEY] = new_spent  # type: ignore[union-attr]
            # Sync tracker to match what was committed to the context.
            with self._lock:
                self._spent = new_spent

    def _compute_cost(self, usage: Usage | dict[str, object] | object) -> float:
        """Compute cost from a provider usage payload."""
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

        return cost


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

    Budget enforcement is postpaid: each LLM call is admitted when the spend
    known at the start of the call is below the configured limit, and the
    call's cost is recorded only after the model response returns. Concurrent
    calls that are both admitted before either records its cost can therefore
    push total spend above the cap. This is intentional; a strict pre-charge
    reservation would require holding the budget lock across the full LLM
    round-trip, serializing all concurrent calls. If strict enforcement is
    required, serialize LLM calls for the context or use external admission
    control.

    Known limitation: When used with subagents (run_task), budget spend is
    tracked per-context copy. Concurrent subagent costs are summed through
    variables.update() on completion, but last-writer-wins semantics mean
    concurrent subagents can undercount total spend. For strict per-tree budget
    enforcement, a shared external ledger is required.
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
        prior_spent = _safe_float(context.variables.get(_SPENT_KEY, 0.0))
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
        """Admit the call using postpaid budget enforcement.

        The call is admitted if spend at the start of the call is below the
        limit, after first syncing the tracker from the shared context. The
        cost is recorded after the response returns. Concurrent calls that
        are both admitted before either records may push total spend above
        the cap. This is intentional because strict pre-charge reservation
        would require holding the budget lock across the full LLM round-trip,
        serializing concurrent calls. If strict enforcement is required,
        callers should serialize LLM calls for the context or use external
        admission control.

        Known limitation: when used with subagents (run_task), budget spend is
        tracked per-context copy. Concurrent subagent costs are summed through
        last-writer-wins variable sync, so concurrent subagents can undercount
        total spend. For strict per-tree budget enforcement, use a shared
        external ledger or explicit delta-merge semantics in run_task.
        """
        if not self._budget.context_check(context):
            raise BudgetExceededError(spent=self._budget.spent, limit=self._budget.limit)

        response = await call_next(events, context)
        self._budget.atomic_update_context(response.usage, context)
        return response
