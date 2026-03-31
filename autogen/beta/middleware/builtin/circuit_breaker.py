# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

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

    def __post_init__(self) -> None:
        if not isinstance(self.failure_threshold, int) or isinstance(self.failure_threshold, bool):
            raise TypeError("failure_threshold must be an int")
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        import math

        if not isinstance(self.recovery_timeout_s, (int, float)) or isinstance(self.recovery_timeout_s, bool):
            raise TypeError("recovery_timeout_s must be a number")
        if math.isnan(self.recovery_timeout_s) or math.isinf(self.recovery_timeout_s):
            raise ValueError("recovery_timeout_s must be a finite number")
        if self.recovery_timeout_s < 0:
            raise ValueError("recovery_timeout_s must be >= 0")


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


class CircuitBreakerMiddleware(MiddlewareFactory):
    """Factory that creates a circuit breaker middleware for LLM calls.

    Wraps outgoing LLM calls and blocks them when the circuit is OPEN
    or when a HALF_OPEN probe slot is already occupied.

    Usage::

        middleware = CircuitBreakerMiddleware(CircuitBreakerConfig(failure_threshold=3, recovery_timeout_s=30))
        agent.register_middleware(middleware)

    Args:
        config: CircuitBreakerConfig controlling thresholds and timeouts.
            Defaults to failure_threshold=5, recovery_timeout_s=60.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self._config = config or CircuitBreakerConfig()
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
            return _blocked_response()

        if state == CircuitState.HALF_OPEN and not claimed_probe:
            logger.debug("CircuitBreaker is HALF_OPEN and probe slot occupied -- blocking LLM call")
            return _blocked_response()

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


def _blocked_response() -> ModelResponse:
    """Return a sentinel response when the circuit is blocking the call."""
    return ModelResponse(message=None, usage={})
