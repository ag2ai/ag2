# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ModelResponse
from autogen.beta.middleware.base import BaseMiddleware, LLMCall, MiddlewareFactory

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Observable state of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for CircuitBreakerMiddleware.

    Args:
        failure_threshold: Number of consecutive failures before the circuit opens.
        recovery_timeout_s: Seconds to wait in OPEN state before allowing a probe.
    """

    failure_threshold: int = 5
    recovery_timeout_s: float = 60.0


class _CircuitBreaker:
    """Thread-safe circuit breaker state machine.

    Not part of the public API. Use CircuitBreakerMiddleware instead.
    """

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._failure_count = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    def check_and_claim(self) -> tuple[CircuitState, bool]:
        """Return current state and whether a HALF_OPEN probe was claimed.

        Atomically checks the state and claims a probe slot when HALF_OPEN.
        Returns (state, claimed_probe).
        """
        with self._lock:
            state = self._state_unlocked()
            if state == CircuitState.HALF_OPEN and not self._probe_in_flight:
                self._probe_in_flight = True
                return state, True
            return state, False

    def release_probe(self) -> None:
        """Release a claimed probe without recording success or failure."""
        with self._lock:
            self._probe_in_flight = False

    def record_failure(self) -> CircuitState:
        """Increment failure count. Open the circuit if threshold is reached.

        Returns the new state after recording the failure.
        """
        with self._lock:
            self._failure_count += 1
            self._probe_in_flight = False
            if self._failure_count >= self._config.failure_threshold:
                if self._opened_at is None:
                    # First time tripping -- log once
                    self._opened_at = time.monotonic()
                    logger.warning(
                        "CircuitBreaker tripped to OPEN after %d consecutive failures",
                        self._failure_count,
                    )
                else:
                    # Refresh timer on subsequent failures while already open
                    self._opened_at = time.monotonic()
            return self._state_unlocked()

    def record_success(self) -> None:
        """Reset the circuit to CLOSED on a successful call."""
        with self._lock:
            self._failure_count = 0
            self._opened_at = None
            self._probe_in_flight = False

    def get_state(self) -> CircuitState:
        """Return the current circuit state."""
        with self._lock:
            return self._state_unlocked()

    def remaining_recovery_s(self) -> float:
        """Return seconds remaining until the OPEN -> HALF_OPEN transition.

        Returns 0.0 if the circuit is not OPEN, or if the recovery timeout
        has already elapsed.
        """
        with self._lock:
            if self._opened_at is None:
                return 0.0
            elapsed = time.monotonic() - self._opened_at
            return max(0.0, self._config.recovery_timeout_s - elapsed)

    def _state_unlocked(self) -> CircuitState:
        """Compute state without acquiring the lock. Caller must hold it."""
        if self._failure_count < self._config.failure_threshold:
            return CircuitState.CLOSED
        if self._opened_at is None:
            return CircuitState.CLOSED
        elapsed = time.monotonic() - self._opened_at
        if elapsed < self._config.recovery_timeout_s:
            return CircuitState.OPEN
        return CircuitState.HALF_OPEN


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a call is blocked because the circuit breaker is not CLOSED.

    Callers can await :meth:`wait_release` to sleep until the OPEN state
    expires, then retry the call::

        while True:
            try:
                result = await agent.ask(...)
                break
            except CircuitBreakerOpenError as exc:
                await exc.wait_release()

    The attribute :attr:`state` records whether the circuit was ``OPEN`` or
    ``HALF_OPEN`` (probe slot busy) at the time of rejection.
    """

    def __init__(self, breaker: "_CircuitBreaker", state: CircuitState) -> None:
        super().__init__(f"CircuitBreaker is {state.value}; LLM call blocked")
        self._breaker = breaker
        self.state = state

    async def wait_release(self) -> None:
        """Sleep until the circuit breaker's recovery timeout elapses.

        For ``OPEN`` state, this computes the exact remaining time and
        sleeps once -- no polling. For ``HALF_OPEN`` contention (a probe
        slot is already occupied), the method returns immediately; callers
        should impose their own backoff before retrying in that case, since
        there is no deterministic signal for when the in-flight probe will
        complete.
        """
        remaining = self._breaker.remaining_recovery_s()
        if remaining > 0:
            await asyncio.sleep(remaining)


class CircuitBreakerMiddleware(MiddlewareFactory):
    """Factory that creates a circuit breaker middleware for LLM calls.

    Wraps outgoing LLM calls and raises :class:`CircuitBreakerOpenError`
    when the circuit is ``OPEN`` or when a ``HALF_OPEN`` probe slot is
    already occupied.

    Usage::

        middleware = CircuitBreakerMiddleware(failure_threshold=3, recovery_timeout_s=30)
        agent = Agent(..., middleware=(middleware,))

        try:
            result = await agent.ask(...)
        except CircuitBreakerOpenError as exc:
            await exc.wait_release()
            result = await agent.ask(...)

    .. warning::
        Each :class:`CircuitBreakerMiddleware` instance owns an internal
        lock and counter. **Do not share one instance across multiple
        agents** unless you explicitly want their failures to count
        against the same circuit -- doing so couples their health and
        serializes their state updates. Prefer per-agent construction::

            agent1 = Agent(..., middleware=(CircuitBreakerMiddleware(),))
            agent2 = Agent(..., middleware=(CircuitBreakerMiddleware(),))

    Args:
        failure_threshold: Consecutive failures before the circuit opens.
        recovery_timeout_s: Seconds to wait in OPEN state before allowing a probe.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 60.0,
    ) -> None:
        self._config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout_s=recovery_timeout_s,
        )
        self._circuit_breaker = _CircuitBreaker(self._config)

    @property
    def circuit_breaker(self) -> _CircuitBreaker:
        """Underlying state machine. Primarily for testing."""
        return self._circuit_breaker

    def __call__(self, event: "BaseEvent", context: "Context") -> "BaseMiddleware":
        return _CircuitBreakerInstance(event, context, self._circuit_breaker)


class _CircuitBreakerInstance(BaseMiddleware):
    """Per-call middleware instance that enforces circuit breaker policy."""

    def __init__(
        self,
        event: "BaseEvent",
        context: "Context",
        circuit_breaker: _CircuitBreaker,
    ) -> None:
        super().__init__(event, context)
        self._cb = circuit_breaker

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        state, claimed_probe = self._cb.check_and_claim()

        if state == CircuitState.OPEN:
            logger.debug("CircuitBreaker is OPEN -- blocking LLM call")
            raise CircuitBreakerOpenError(self._cb, CircuitState.OPEN)

        if state == CircuitState.HALF_OPEN and not claimed_probe:
            logger.debug("CircuitBreaker is HALF_OPEN and probe slot occupied -- blocking LLM call")
            raise CircuitBreakerOpenError(self._cb, CircuitState.HALF_OPEN)

        try:
            response = await call_next(events, context)
        except Exception:
            self._cb.record_failure()
            raise
        except BaseException:
            # CancelledError and similar -- release probe without recording failure
            if claimed_probe:
                self._cb.release_probe()
            raise

        self._cb.record_success()
        return response
