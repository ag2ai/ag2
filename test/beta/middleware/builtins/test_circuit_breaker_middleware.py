# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from collections.abc import Sequence
from unittest import mock

import pytest

from autogen.beta.events import BaseEvent, ModelResponse
from autogen.beta.middleware.builtin.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerMiddleware,
    CircuitBreakerOpenError,
    CircuitState,
    _CircuitBreaker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event() -> mock.MagicMock:
    return mock.MagicMock(spec=BaseEvent)


def _make_context() -> mock.MagicMock:
    return mock.MagicMock()


def _make_model_response() -> ModelResponse:
    return ModelResponse(usage={"prompt_tokens": 0, "output_tokens": 0})


async def _success_call(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
    return _make_model_response()


async def _failing_call(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
    raise RuntimeError("LLM error")


# ---------------------------------------------------------------------------
# TestCircuitBreaker -- state machine unit tests
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_trips_after_threshold(self) -> None:
        # Given a circuit with threshold=3
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout_s=60.0)
        cb = _CircuitBreaker(config)

        # When 3 consecutive failures are recorded
        cb.record_failure()
        cb.record_failure()
        state = cb.record_failure()

        # Then the circuit is OPEN
        assert state == CircuitState.OPEN

    @pytest.mark.asyncio()
    async def test_open_raises_and_blocks_call(self) -> None:
        # Given an OPEN circuit (threshold=1)
        factory = CircuitBreakerMiddleware(failure_threshold=1, recovery_timeout_s=9999.0)
        factory.circuit_breaker.record_failure()

        # When on_llm_call is invoked
        call_next = mock.AsyncMock()
        mw = factory(_make_event(), _make_context())

        # Then CircuitBreakerOpenError is raised and call_next is never invoked
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await mw.on_llm_call(call_next, [], _make_context())
        assert exc_info.value.state == CircuitState.OPEN
        call_next.assert_not_called()

    def test_success_resets(self) -> None:
        # Given a circuit with some failures below threshold
        config = CircuitBreakerConfig(failure_threshold=5, recovery_timeout_s=60.0)
        cb = _CircuitBreaker(config)
        cb.record_failure()
        cb.record_failure()

        # When a success is recorded
        cb.record_success()

        # Then the circuit is CLOSED again
        assert cb.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio()
    async def test_half_open_probe_allowed(self) -> None:
        # Given an OPEN circuit whose timeout has expired (timeout=0)
        factory = CircuitBreakerMiddleware(failure_threshold=1, recovery_timeout_s=0.0)
        factory.circuit_breaker.record_failure()

        # When on_llm_call is invoked (circuit transitions to HALF_OPEN)
        call_next = mock.AsyncMock(return_value=_make_model_response())
        mw = factory(_make_event(), _make_context())
        await mw.on_llm_call(call_next, [], _make_context())

        # Then the probe call is forwarded and circuit resets to CLOSED
        call_next.assert_called_once()
        assert factory.circuit_breaker.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio()
    async def test_half_open_second_probe_blocked(self) -> None:
        # Given a HALF_OPEN circuit with a probe already in flight
        factory = CircuitBreakerMiddleware(failure_threshold=1, recovery_timeout_s=0.0)
        cb = factory.circuit_breaker
        cb.record_failure()

        # Claim the single probe slot manually
        _, claimed = cb.check_and_claim()
        assert claimed, "Expected to claim the probe slot"

        # When a second call is attempted
        call_next = mock.AsyncMock()
        mw = factory(_make_event(), _make_context())

        # Then CircuitBreakerOpenError(state=HALF_OPEN) is raised
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await mw.on_llm_call(call_next, [], _make_context())
        assert exc_info.value.state == CircuitState.HALF_OPEN
        call_next.assert_not_called()

        # Cleanup
        cb.release_probe()

    @pytest.mark.asyncio()
    async def test_cancelled_error_releases_probe_without_tripping(self) -> None:
        # Given a HALF_OPEN circuit
        factory = CircuitBreakerMiddleware(failure_threshold=1, recovery_timeout_s=0.0)
        factory.circuit_breaker.record_failure()

        # When call_next raises CancelledError (a BaseException, not Exception)
        async def _cancel(_events: Sequence[BaseEvent], _ctx: object) -> ModelResponse:
            raise asyncio.CancelledError()

        mw = factory(_make_event(), _make_context())
        with pytest.raises(asyncio.CancelledError):
            await mw.on_llm_call(_cancel, [], _make_context())

        # Then no additional failure is recorded -- circuit stays HALF_OPEN
        state = factory.circuit_breaker.get_state()
        assert state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio()
    async def test_cancelled_error_in_closed_state(self) -> None:
        # Given a CLOSED circuit
        factory = CircuitBreakerMiddleware(failure_threshold=5, recovery_timeout_s=60.0)

        # When call_next raises CancelledError
        async def _cancel(_events: Sequence[BaseEvent], _ctx: object) -> ModelResponse:
            raise asyncio.CancelledError()

        mw = factory(_make_event(), _make_context())
        with pytest.raises(asyncio.CancelledError):
            await mw.on_llm_call(_cancel, [], _make_context())

        # Then circuit stays CLOSED with zero failures
        assert factory.circuit_breaker.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio()
    async def test_circuit_open_log_on_trip(self, caplog: pytest.LogCaptureFixture) -> None:
        # Given a circuit with threshold=1
        import logging

        factory = CircuitBreakerMiddleware(failure_threshold=1, recovery_timeout_s=9999.0)

        # When the circuit trips
        with caplog.at_level(logging.WARNING, logger="autogen.beta.middleware.builtin.circuit_breaker"):

            async def _fail(_events: Sequence[BaseEvent], _ctx: object) -> ModelResponse:
                raise RuntimeError("boom")

            mw = factory(_make_event(), _make_context())
            with pytest.raises(RuntimeError):
                await mw.on_llm_call(_fail, [], _make_context())

        # Then a warning log mentions OPEN
        assert any("OPEN" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# TestCircuitBreakerMiddleware -- constructor and fallback behavior
# ---------------------------------------------------------------------------


class TestCircuitBreakerMiddleware:
    def test_flat_constructor(self) -> None:
        # Given arguments passed directly (no explicit config object)
        mw = CircuitBreakerMiddleware(failure_threshold=3, recovery_timeout_s=30.0)

        # Then the internal config is built from those arguments
        assert mw._config.failure_threshold == 3
        assert mw._config.recovery_timeout_s == 30.0

    def test_default_constructor(self) -> None:
        # Given no arguments
        mw = CircuitBreakerMiddleware()

        # Then defaults match CircuitBreakerConfig defaults
        assert mw._config.failure_threshold == 5
        assert mw._config.recovery_timeout_s == 60.0

    @pytest.mark.asyncio()
    async def test_open_error_message_mentions_state(self) -> None:
        # Given an OPEN circuit
        factory = CircuitBreakerMiddleware(failure_threshold=1, recovery_timeout_s=9999.0)
        factory.circuit_breaker.record_failure()

        # When on_llm_call is invoked
        mw = factory(_make_event(), _make_context())
        with pytest.raises(CircuitBreakerOpenError, match="open") as exc_info:
            await mw.on_llm_call(mock.AsyncMock(), [], _make_context())

        # Then the error carries a reference to the breaker and its state
        assert exc_info.value.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# TestWaitRelease
# ---------------------------------------------------------------------------


class TestWaitRelease:
    @pytest.mark.asyncio()
    async def test_wait_release_sleeps_remaining_recovery_time(self) -> None:
        # Given an OPEN circuit with a 0.1s recovery timeout
        factory = CircuitBreakerMiddleware(failure_threshold=1, recovery_timeout_s=0.1)
        factory.circuit_breaker.record_failure()

        # When we trigger the error and await wait_release
        mw = factory(_make_event(), _make_context())
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await mw.on_llm_call(mock.AsyncMock(), [], _make_context())

        start = asyncio.get_event_loop().time()
        await exc_info.value.wait_release()
        elapsed = asyncio.get_event_loop().time() - start

        # Then we slept roughly the remaining recovery time (bounded above 0)
        assert elapsed >= 0.05
        # After waking, the circuit should be HALF_OPEN or CLOSED, not OPEN
        assert factory.circuit_breaker.get_state() != CircuitState.OPEN

    @pytest.mark.asyncio()
    async def test_wait_release_half_open_returns_immediately(self) -> None:
        # Given a HALF_OPEN circuit with a probe slot already occupied
        factory = CircuitBreakerMiddleware(failure_threshold=1, recovery_timeout_s=0.0)
        cb = factory.circuit_breaker
        cb.record_failure()
        _, claimed = cb.check_and_claim()
        assert claimed

        mw = factory(_make_event(), _make_context())
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await mw.on_llm_call(mock.AsyncMock(), [], _make_context())
        assert exc_info.value.state == CircuitState.HALF_OPEN

        # When we await wait_release
        start = asyncio.get_event_loop().time()
        await exc_info.value.wait_release()
        elapsed = asyncio.get_event_loop().time() - start

        # Then it returns immediately (remaining recovery time is 0)
        assert elapsed < 0.05
        cb.release_probe()

    def test_remaining_recovery_s_zero_when_not_open(self) -> None:
        # Given a CLOSED circuit
        config = CircuitBreakerConfig(failure_threshold=5, recovery_timeout_s=60.0)
        cb = _CircuitBreaker(config)

        # Then remaining_recovery_s is 0.0
        assert cb.remaining_recovery_s() == 0.0

    def test_remaining_recovery_s_positive_when_open(self) -> None:
        # Given a freshly OPEN circuit with a long recovery timeout
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=9999.0)
        cb = _CircuitBreaker(config)
        cb.record_failure()

        # Then remaining_recovery_s is close to the full timeout
        remaining = cb.remaining_recovery_s()
        assert 9990.0 < remaining <= 9999.0


# ---------------------------------------------------------------------------
# TestConcurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_claim_probe_exactly_one(self) -> None:
        # Given a HALF_OPEN circuit
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=0.0)
        cb = _CircuitBreaker(config)
        cb.record_failure()

        # When 10 threads simultaneously try to claim the probe
        claimed_count = 0
        count_lock = threading.Lock()

        def _try_claim() -> None:
            nonlocal claimed_count
            state, claimed = cb.check_and_claim()
            if claimed:
                with count_lock:
                    claimed_count += 1

        threads = [threading.Thread(target=_try_claim) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Then exactly one thread claimed the probe
        assert claimed_count == 1

        cb.release_probe()


# ---------------------------------------------------------------------------
# TestTimerRefresh
# ---------------------------------------------------------------------------


class TestTimerRefresh:
    def test_failure_during_open_restarts_recovery_timer(self) -> None:
        # Given an OPEN circuit
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=9999.0)
        cb = _CircuitBreaker(config)
        cb.record_failure()

        import time

        first_opened_at = cb._opened_at
        assert first_opened_at is not None

        # When another failure arrives while already OPEN
        # (use 50ms sleep -- Windows monotonic clock resolution is ~15ms)
        time.sleep(0.05)
        cb.record_failure()

        # Then the recovery timer is refreshed (opened_at moved forward)
        assert cb._opened_at is not None
        assert cb._opened_at > first_opened_at
