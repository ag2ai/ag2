# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import pickle
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

    def test_record_success_in_open_state_does_not_close(self) -> None:
        # Given an OPEN circuit
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=9999.0)
        cb = _CircuitBreaker(config)
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN

        # When a delayed success is recorded while still OPEN
        cb.record_success()

        # Then the OPEN state is preserved
        assert cb.get_state() == CircuitState.OPEN

    def test_closed_stale_success_arriving_after_half_open_does_not_close(self) -> None:
        # Given a circuit whose recovery timeout has elapsed
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=0.0)
        cb = _CircuitBreaker(config)
        cb.record_failure()
        assert cb.get_state() == CircuitState.HALF_OPEN

        # When a stale success from a non-probe call completes
        cb.record_success(probe_token=None)

        # Then the success is ignored and the circuit remains HALF_OPEN
        assert cb.get_state() == CircuitState.HALF_OPEN

    def test_stale_probe_does_not_close_circuit_after_failure_refresh(self) -> None:
        # Given a circuit with a claimed HALF_OPEN probe
        import time

        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=0.0)
        cb = _CircuitBreaker(config)
        cb.record_failure()

        state, first_probe_token = cb.check_and_claim()
        assert state == CircuitState.HALF_OPEN
        assert first_probe_token is not None

        # When a later failure refreshes opened_at and advances the probe generation
        cb.record_failure()
        assert cb._probe_generation != first_probe_token

        # And the circuit becomes HALF_OPEN again
        cb._opened_at = time.monotonic() - 1.0
        assert cb.get_state() == CircuitState.HALF_OPEN

        # Then the stale probe success is ignored instead of closing the circuit
        cb.record_success(probe_token=first_probe_token)
        assert cb.get_state() == CircuitState.HALF_OPEN

    def test_failed_half_open_probe_reopens_to_open(self) -> None:
        # Given a circuit whose recovery timeout has elapsed
        import time

        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=60.0)
        cb = _CircuitBreaker(config)
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN
        cb._opened_at = time.monotonic() - 61.0
        assert cb.get_state() == CircuitState.HALF_OPEN

        # And the recovery probe has been claimed
        state, probe_token = cb.check_and_claim()
        assert state == CircuitState.HALF_OPEN
        assert probe_token is not None

        # When that probe fails
        state = cb.record_failure()

        # Then the circuit reopens instead of staying HALF_OPEN
        assert state == CircuitState.OPEN
        assert cb.get_state() == CircuitState.OPEN

    def test_record_success_in_closed_state_resets_failure_count(self) -> None:
        # Given a CLOSED circuit with failures below threshold
        config = CircuitBreakerConfig(failure_threshold=5, recovery_timeout_s=60.0)
        cb = _CircuitBreaker(config)
        cb.record_failure()
        cb.record_failure()
        assert cb.get_state() == CircuitState.CLOSED
        assert cb._failure_count == 2

        # When a success is recorded
        cb.record_success()

        # Then the circuit stays CLOSED and the failure count is reset
        assert cb.get_state() == CircuitState.CLOSED
        assert cb._failure_count == 0

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
        _, probe_token = cb.check_and_claim()
        assert probe_token is not None, "Expected to claim the probe slot"

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
    async def test_half_open_probe_exception_releases_slot(self) -> None:
        # Given a HALF_OPEN circuit
        factory = CircuitBreakerMiddleware(failure_threshold=1, recovery_timeout_s=0.0)
        cb = factory.circuit_breaker
        cb.record_failure()

        # When the claimed recovery probe raises a normal exception
        mw = factory(_make_event(), _make_context())
        with pytest.raises(RuntimeError, match="LLM error"):
            await mw.on_llm_call(_failing_call, [], _make_context())

        # Then the failed probe slot is released and another probe can be claimed
        assert cb._probe_in_flight is False
        state, probe_token = cb.check_and_claim()
        assert state == CircuitState.HALF_OPEN
        assert probe_token is not None
        cb.release_probe()

    @pytest.mark.asyncio()
    async def test_cancellation_release_allows_next_probe(self) -> None:
        # Given a HALF_OPEN circuit
        factory = CircuitBreakerMiddleware(failure_threshold=1, recovery_timeout_s=0.0)
        cb = factory.circuit_breaker
        cb.record_failure()

        # When the claimed recovery probe is cancelled
        async def _cancel(_events: Sequence[BaseEvent], _ctx: object) -> ModelResponse:
            raise asyncio.CancelledError()

        mw = factory(_make_event(), _make_context())
        with pytest.raises(asyncio.CancelledError):
            await mw.on_llm_call(_cancel, [], _make_context())

        # Then the cancelled probe slot is released and another probe can be claimed
        assert cb._probe_in_flight is False
        state, probe_token = cb.check_and_claim()
        assert state == CircuitState.HALF_OPEN
        assert probe_token is not None
        cb.release_probe()

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

        # Then the error carries the rejection state
        assert exc_info.value.state == CircuitState.OPEN

    def test_circuit_breaker_open_error_picklable(self) -> None:
        # Given an open-circuit error with only snapshot state
        error = CircuitBreakerOpenError(remaining_s=5.0, state=CircuitState.OPEN)

        # When the error is pickled and restored
        restored = pickle.loads(pickle.dumps(error))

        # Then the observable state and message are preserved
        assert restored.state == CircuitState.OPEN
        assert str(restored) == str(error)


# ---------------------------------------------------------------------------
# TestWaitRelease
# ---------------------------------------------------------------------------


class TestWaitRelease:
    @pytest.mark.asyncio()
    async def test_wait_release_sleeps_for_remaining_time(self) -> None:
        # Given an error with remaining recovery time
        error = CircuitBreakerOpenError(remaining_s=5.0, state=CircuitState.OPEN)

        # When wait_release is awaited
        with mock.patch(
            "autogen.beta.middleware.builtin.circuit_breaker.asyncio.sleep",
            new_callable=mock.AsyncMock,
        ) as sleep:
            await error.wait_release()

        # Then it sleeps for the captured remaining time
        sleep.assert_awaited_once_with(5.0)

    @pytest.mark.asyncio()
    async def test_wait_release_returns_immediately_when_zero(self) -> None:
        # Given an error with no remaining recovery time
        error = CircuitBreakerOpenError(remaining_s=0.0, state=CircuitState.HALF_OPEN)

        # When wait_release is awaited
        with mock.patch(
            "autogen.beta.middleware.builtin.circuit_breaker.asyncio.sleep",
            new_callable=mock.AsyncMock,
        ) as sleep:
            await error.wait_release()

        # Then it returns without sleeping
        sleep.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_wait_release_sleeps_remaining_recovery_time(self) -> None:
        # Given a CircuitBreakerOpenError with a known remaining_s
        error = CircuitBreakerOpenError(remaining_s=5.0, state=CircuitState.OPEN)

        # When wait_release is awaited, asyncio.sleep is called with the snapshot value
        with mock.patch(
            "autogen.beta.middleware.builtin.circuit_breaker.asyncio.sleep",
            new_callable=mock.AsyncMock,
        ) as sleep:
            await error.wait_release()

        # Then asyncio.sleep was called with the exact remaining snapshot
        sleep.assert_awaited_once_with(5.0)

    @pytest.mark.asyncio()
    async def test_wait_release_half_open_returns_immediately(self) -> None:
        # Given a HALF_OPEN circuit with a probe slot already occupied
        factory = CircuitBreakerMiddleware(failure_threshold=1, recovery_timeout_s=0.0)
        cb = factory.circuit_breaker
        cb.record_failure()
        _, probe_token = cb.check_and_claim()
        assert probe_token is not None

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
# TestProbeGeneration
# ---------------------------------------------------------------------------


class TestProbeGeneration:
    def test_probe_generation_advances_on_first_failure(self) -> None:
        # Given a fresh CLOSED circuit
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=60.0)
        cb = _CircuitBreaker(config)
        initial_generation = cb._probe_generation
        assert initial_generation == 0

        # When the threshold-crossing failure trips the circuit
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN

        # Then _probe_generation advances to 1
        assert cb._probe_generation == 1

    def test_probe_generation_advances_again_on_subsequent_failure(self) -> None:
        # Given an already-OPEN circuit
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=60.0)
        cb = _CircuitBreaker(config)
        cb.record_failure()
        assert cb._probe_generation == 1
        assert cb._opened_at is not None

        # When another failure arrives while already OPEN
        cb.record_failure()

        # Then _probe_generation advances again
        assert cb._probe_generation == 2

    @pytest.mark.asyncio()
    async def test_middleware_stale_claimed_probe_does_not_close_circuit(self) -> None:
        """Middleware-level regression for the stale claimed-probe timeline.

        Timeline:
        1. Trip circuit to OPEN.
        2. Let recovery elapse; probe P is admitted (HALF_OPEN).
        3. Trigger a failure that refreshes _opened_at and bumps _probe_generation.
        4. Let recovery elapse again.
        5. Complete probe P successfully.
        6. Assert circuit is not CLOSED.
        """

        event = _make_event()
        ctx = _make_context()
        factory = CircuitBreakerMiddleware(failure_threshold=1, recovery_timeout_s=0.0)

        # --- Step 1: trip the circuit ---
        call_count = 0

        async def failing(events: Sequence[BaseEvent], c: object) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("trip")

        instance = factory(event, ctx)
        with pytest.raises(RuntimeError):
            await instance.on_llm_call(failing, [], ctx)

        # --- Step 2: probe P is admitted; hold it by monkey-patching record_success ---
        probe_token_holder: list[int | None] = []

        # Expose the internal _CircuitBreaker via the factory
        cb: _CircuitBreaker = factory._circuit_breaker  # type: ignore[attr-defined]

        # recovery_timeout_s=0 means HALF_OPEN immediately -- claim the probe
        state, probe_token = cb.check_and_claim()
        assert state == CircuitState.HALF_OPEN
        assert probe_token is not None
        probe_token_holder.append(probe_token)

        # --- Step 3: intervening failure refreshes _opened_at and bumps generation ---
        generation_before = cb._probe_generation
        cb.record_failure()
        assert cb._probe_generation == generation_before + 1

        # --- Step 4: recovery elapses again (timeout=0, so already elapsed) ---
        assert cb.get_state() == CircuitState.HALF_OPEN

        # --- Step 5: stale probe completes successfully ---
        cb.record_success(probe_token=probe_token_holder[0])

        # --- Step 6: circuit must NOT be CLOSED ---
        state = cb.get_state()
        assert state in (CircuitState.HALF_OPEN, CircuitState.OPEN), f"Stale probe must not close circuit; got {state}"


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
            _state, probe_token = cb.check_and_claim()
            if probe_token is not None:
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


# ---------------------------------------------------------------------------
# TestCircuitBreakerTopLevelExport
# ---------------------------------------------------------------------------


def test_circuit_breaker_importable_from_middleware_top_level() -> None:
    # Given/When/Then: CircuitBreakerMiddleware, CircuitBreakerConfig,
    # CircuitBreakerOpenError, and CircuitState are importable from the
    # top-level autogen.beta.middleware package (export regression).
    import autogen.beta.middleware as mw_pkg

    assert mw_pkg.CircuitBreakerMiddleware is CircuitBreakerMiddleware
    assert mw_pkg.CircuitBreakerConfig is CircuitBreakerConfig
    assert mw_pkg.CircuitBreakerOpenError is CircuitBreakerOpenError
    assert mw_pkg.CircuitState is CircuitState
