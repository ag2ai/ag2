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
    async def test_open_blocks_call(self) -> None:
        # Given an OPEN circuit (threshold=1)
        factory = CircuitBreakerMiddleware(CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=9999.0))
        factory.circuit_breaker.record_failure()

        # When on_llm_call is invoked
        call_next = mock.AsyncMock()
        mw = factory(_make_event(), _make_context())
        response = await mw.on_llm_call(call_next, [], _make_context())

        # Then the call is blocked and call_next is never invoked
        assert response.message is None
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
        factory = CircuitBreakerMiddleware(CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=0.0))
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
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=0.0)
        factory = CircuitBreakerMiddleware(config)
        cb = factory.circuit_breaker
        cb.record_failure()

        # Claim the single probe slot manually
        _, claimed = cb.check_and_claim()
        assert claimed, "Expected to claim the probe slot"

        # When a second call is attempted
        call_next = mock.AsyncMock()
        mw = factory(_make_event(), _make_context())
        response = await mw.on_llm_call(call_next, [], _make_context())

        # Then the second call is blocked
        assert response.message is None
        call_next.assert_not_called()

        # Cleanup
        cb.release_probe()

    @pytest.mark.asyncio()
    async def test_cancelled_error_releases_probe_without_tripping(self) -> None:
        # Given a HALF_OPEN circuit
        factory = CircuitBreakerMiddleware(CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=0.0))
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
        factory = CircuitBreakerMiddleware(CircuitBreakerConfig(failure_threshold=5, recovery_timeout_s=60.0))

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

        factory = CircuitBreakerMiddleware(CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=9999.0))

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
# TestCircuitBreakerConfig -- validation tests
# ---------------------------------------------------------------------------


class TestCircuitBreakerConfig:
    def test_zero_threshold_rejected(self) -> None:
        # Given / When / Then
        with pytest.raises(ValueError, match="failure_threshold"):
            CircuitBreakerConfig(failure_threshold=0)

    def test_negative_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="recovery_timeout_s"):
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=-1.0)

    @pytest.mark.parametrize(
        "bad_threshold",
        [
            pytest.param(1.5, id="float"),
            pytest.param(float("nan"), id="nan"),
            pytest.param(float("inf"), id="inf"),
            pytest.param("3", id="string"),
        ],
    )
    def test_non_int_threshold_rejected(self, bad_threshold: object) -> None:
        with pytest.raises((TypeError, ValueError)):
            CircuitBreakerConfig(failure_threshold=bad_threshold)  # type: ignore[arg-type]


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
        time.sleep(0.01)
        cb.record_failure()

        # Then the recovery timer is refreshed (opened_at moved forward)
        assert cb._opened_at is not None
        assert cb._opened_at > first_opened_at
