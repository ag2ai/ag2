# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for StepController, AsyncStepController, RunIterResponse, and AsyncRunIterResponse."""

import asyncio
import queue
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from autogen.events.agent_events import ErrorEvent, InputRequestEvent, RunCompletionEvent, TerminationEvent, TextEvent
from autogen.io.run_response import AsyncRunIterResponse, RunIterResponse
from autogen.io.step_controller import AsyncStepController, StepController
from autogen.io.thread_io_stream import ThreadIOStream


class TestStepController:
    """Tests for the synchronous StepController."""

    def test_should_block_no_filter_blocks_all_events(self) -> None:
        """When yield_on is None, should_block returns True for all events."""
        controller = StepController(yield_on=None)

        # Create mock events of different types
        text_event = MagicMock(spec=TextEvent)
        termination_event = MagicMock(spec=TerminationEvent)

        assert controller.should_block(text_event) is True
        assert controller.should_block(termination_event) is True

    def test_should_block_with_filter_only_blocks_specified_types(self) -> None:
        """When yield_on is specified, should_block returns True only for those types."""
        controller = StepController(yield_on=[TextEvent, TerminationEvent])

        # TextEvent should be blocked - use real instance
        text_event = MagicMock(spec=TextEvent)
        text_event.__class__ = TextEvent  # type: ignore[assignment]
        assert controller.should_block(text_event) is True

        # TerminationEvent should be blocked
        termination_event = MagicMock(spec=TerminationEvent)
        termination_event.__class__ = TerminationEvent  # type: ignore[assignment]
        assert controller.should_block(termination_event) is True

        # Other events should not be blocked - use a MagicMock without setting __class__
        other_event = MagicMock()
        assert controller.should_block(other_event) is False

    def test_should_block_after_terminate_returns_false(self) -> None:
        """After terminate() is called, should_block always returns False."""
        controller = StepController(yield_on=None)
        text_event = MagicMock(spec=TextEvent)

        # Before terminate
        assert controller.should_block(text_event) is True

        # After terminate
        controller.terminate()
        assert controller.should_block(text_event) is False

    def test_step_unblocks_wait_for_step(self) -> None:
        """Calling step() should unblock a thread waiting on wait_for_step()."""
        controller = StepController(yield_on=None)
        event = MagicMock(spec=TextEvent)
        wait_completed = threading.Event()

        def producer() -> None:
            controller.wait_for_step(event)
            wait_completed.set()

        producer_thread = threading.Thread(target=producer)
        producer_thread.start()

        # Give producer time to block
        time.sleep(0.1)
        assert not wait_completed.is_set()

        # Unblock with step()
        controller.step()

        # Wait for producer to complete
        producer_thread.join(timeout=1.0)
        assert wait_completed.is_set()

    def test_wait_for_step_skips_non_matching_events(self) -> None:
        """wait_for_step should return immediately for non-matching events."""
        controller = StepController(yield_on=[TerminationEvent])

        # TextEvent should not block since we only filter on TerminationEvent
        text_event = MagicMock(spec=TextEvent)
        # Don't set __class__ so it won't match TerminationEvent

        # This should return immediately, not block
        start = time.time()
        controller.wait_for_step(text_event)
        elapsed = time.time() - start

        assert elapsed < 0.1  # Should be nearly instant

    def test_terminate_unblocks_waiting_producer(self) -> None:
        """terminate() should unblock any thread waiting on wait_for_step()."""
        controller = StepController(yield_on=None)
        event = MagicMock(spec=TextEvent)
        wait_completed = threading.Event()

        def producer() -> None:
            controller.wait_for_step(event)
            wait_completed.set()

        producer_thread = threading.Thread(target=producer)
        producer_thread.start()

        # Give producer time to block
        time.sleep(0.1)
        assert not wait_completed.is_set()

        # Terminate should unblock
        controller.terminate()

        # Wait for producer to complete
        producer_thread.join(timeout=1.0)
        assert wait_completed.is_set()


class TestAsyncStepController:
    """Tests for the asynchronous AsyncStepController."""

    def test_should_block_no_filter_blocks_all_events(self) -> None:
        """When yield_on is None, should_block returns True for all events."""
        controller = AsyncStepController(yield_on=None)

        text_event = MagicMock(spec=TextEvent)
        termination_event = MagicMock(spec=TerminationEvent)

        assert controller.should_block(text_event) is True
        assert controller.should_block(termination_event) is True

    def test_should_block_with_filter_only_blocks_specified_types(self) -> None:
        """When yield_on is specified, should_block returns True only for those types."""
        controller = AsyncStepController(yield_on=[TextEvent])

        text_event = MagicMock(spec=TextEvent)
        text_event.__class__ = TextEvent  # type: ignore[assignment]
        assert controller.should_block(text_event) is True

        other_event = MagicMock()
        assert controller.should_block(other_event) is False

    def test_should_block_after_terminate_returns_false(self) -> None:
        """After terminate() is called, should_block always returns False."""
        controller = AsyncStepController(yield_on=None)
        text_event = MagicMock(spec=TextEvent)

        assert controller.should_block(text_event) is True
        controller.terminate()
        assert controller.should_block(text_event) is False

    @pytest.mark.asyncio
    async def test_step_unblocks_wait_for_step(self) -> None:
        """Calling step() should unblock wait_for_step()."""
        controller = AsyncStepController(yield_on=None)
        event = MagicMock(spec=TextEvent)
        wait_completed = asyncio.Event()

        async def producer() -> None:
            await controller.wait_for_step(event)
            wait_completed.set()

        # Start producer task
        producer_task = asyncio.create_task(producer())

        # Give producer time to block
        await asyncio.sleep(0.1)
        assert not wait_completed.is_set()

        # Unblock with step()
        controller.step()

        # Wait for producer to complete
        await asyncio.wait_for(producer_task, timeout=1.0)
        assert wait_completed.is_set()

    @pytest.mark.asyncio
    async def test_wait_for_step_skips_non_matching_events(self) -> None:
        """wait_for_step should return immediately for non-matching events."""
        controller = AsyncStepController(yield_on=[TerminationEvent])

        text_event = MagicMock(spec=TextEvent)

        # This should return immediately
        start = time.time()
        await controller.wait_for_step(text_event)
        elapsed = time.time() - start

        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_terminate_unblocks_waiting_producer(self) -> None:
        """terminate() should unblock any task waiting on wait_for_step()."""
        controller = AsyncStepController(yield_on=None)
        event = MagicMock(spec=TextEvent)
        wait_completed = asyncio.Event()

        async def producer() -> None:
            await controller.wait_for_step(event)
            wait_completed.set()

        producer_task = asyncio.create_task(producer())

        await asyncio.sleep(0.1)
        assert not wait_completed.is_set()

        controller.terminate()

        await asyncio.wait_for(producer_task, timeout=1.0)
        assert wait_completed.is_set()


class TestRunIterResponse:
    """Tests for RunIterResponse iterator-based stepping."""

    def _create_run_iter_response(
        self,
        events: list[Any],
        yield_on: list[type] | None = None,
    ) -> RunIterResponse:
        """Create a RunIterResponse that will yield the given events.

        Args:
            events: List of events to put in the queue.
            yield_on: Event types to filter on.

        Returns:
            RunIterResponse configured to yield the events.
        """
        event_queue: queue.Queue[Any] = queue.Queue()
        for event in events:
            event_queue.put(event)

        def start_thread_func(iostream: ThreadIOStream) -> threading.Thread:
            def producer() -> None:
                assert iostream._step_controller is not None
                while True:
                    try:
                        event = event_queue.get_nowait()
                        iostream._input_stream.put(event)
                        # Wait for step to continue (simulates real behavior)
                        iostream._step_controller.wait_for_step(event)
                    except queue.Empty:
                        break

            return threading.Thread(target=producer, daemon=True)

        return RunIterResponse(
            start_thread_func=start_thread_func,
            yield_on=yield_on,
            agents=[],
        )

    def test_iteration_yields_events_until_completion(self) -> None:
        """Events are yielded until RunCompletionEvent is received."""
        text_event = MagicMock(spec=TextEvent)
        text_event.__class__ = TextEvent  # type: ignore[assignment]

        completion_event = MagicMock(spec=RunCompletionEvent)
        completion_event.__class__ = RunCompletionEvent  # type: ignore[assignment]
        completion_event.content = MagicMock()
        completion_event.content.history = []
        completion_event.content.summary = "Test summary"
        completion_event.content.cost = {}
        completion_event.content.last_speaker = "test"
        completion_event.content.context_variables = None

        response = self._create_run_iter_response([text_event, completion_event])

        events = list(response)

        assert len(events) == 1
        assert events[0] is text_event
        assert response.summary == "Test summary"

    def test_iteration_raises_on_error_event(self) -> None:
        """Iteration should raise the error from ErrorEvent."""
        test_error = ValueError("Test error message")
        error_event = MagicMock(spec=ErrorEvent)
        error_event.__class__ = ErrorEvent  # type: ignore[assignment]
        error_event.content = MagicMock()
        error_event.content.error = test_error

        response = self._create_run_iter_response([error_event])

        with pytest.raises(ValueError, match="Test error message"):
            list(response)

    def test_break_terminates_step_controller(self) -> None:
        """Breaking from iteration should terminate the step controller."""
        text_event1 = MagicMock(spec=TextEvent)
        text_event1.__class__ = TextEvent  # type: ignore[assignment]
        text_event2 = MagicMock(spec=TextEvent)
        text_event2.__class__ = TextEvent  # type: ignore[assignment]

        completion_event = MagicMock(spec=RunCompletionEvent)
        completion_event.__class__ = RunCompletionEvent  # type: ignore[assignment]
        completion_event.content = MagicMock()
        completion_event.content.history = []
        completion_event.content.summary = ""
        completion_event.content.cost = {}
        completion_event.content.last_speaker = ""
        completion_event.content.context_variables = None

        response = self._create_run_iter_response([text_event1, text_event2, completion_event])

        for event in response:
            break  # Exit early

        # Generator cleanup should have terminated the controller
        assert response._step_controller._terminated

    def test_exception_terminates_step_controller(self) -> None:
        """Exception during iteration should terminate the step controller."""
        text_event = MagicMock(spec=TextEvent)
        text_event.__class__ = TextEvent  # type: ignore[assignment]

        completion_event = MagicMock(spec=RunCompletionEvent)
        completion_event.__class__ = RunCompletionEvent  # type: ignore[assignment]
        completion_event.content = MagicMock()
        completion_event.content.history = []
        completion_event.content.summary = ""
        completion_event.content.cost = {}
        completion_event.content.last_speaker = ""
        completion_event.content.context_variables = None

        response = self._create_run_iter_response([text_event, completion_event])

        with pytest.raises(ValueError):
            for event in response:
                raise ValueError("Test exception")

        # Generator cleanup should have terminated the controller
        assert response._step_controller._terminated

    def test_lazy_start(self) -> None:
        """Thread should only start on first iteration."""
        completion_event = MagicMock(spec=RunCompletionEvent)
        completion_event.__class__ = RunCompletionEvent  # type: ignore[assignment]
        completion_event.content = MagicMock()
        completion_event.content.history = []
        completion_event.content.summary = ""
        completion_event.content.cost = {}
        completion_event.content.last_speaker = ""
        completion_event.content.context_variables = None

        response = self._create_run_iter_response([completion_event])

        # Before iteration
        assert not response._started
        assert response._thread is None

        # Start iteration
        list(response)

        # After iteration
        assert response._started
        assert response._thread is not None

    def test_yield_on_filters_events(self) -> None:
        """Only events matching yield_on should be yielded."""
        text_event = MagicMock(spec=TextEvent)
        # Don't set __class__ so it won't match filter

        termination_event = MagicMock(spec=TerminationEvent)
        termination_event.__class__ = TerminationEvent  # type: ignore[assignment]

        completion_event = MagicMock(spec=RunCompletionEvent)
        completion_event.__class__ = RunCompletionEvent  # type: ignore[assignment]
        completion_event.content = MagicMock()
        completion_event.content.history = []
        completion_event.content.summary = ""
        completion_event.content.cost = {}
        completion_event.content.last_speaker = ""
        completion_event.content.context_variables = None

        response = self._create_run_iter_response(
            [text_event, termination_event, completion_event],
            yield_on=[TerminationEvent],
        )

        events = list(response)

        # Only TerminationEvent should be yielded
        assert len(events) == 1
        assert events[0] is termination_event

    def test_input_request_always_yielded(self) -> None:
        """InputRequestEvent should always be yielded regardless of yield_on filter."""
        input_event = MagicMock(spec=InputRequestEvent)
        input_event.__class__ = InputRequestEvent  # type: ignore[assignment]
        input_event.content = MagicMock()

        completion_event = MagicMock(spec=RunCompletionEvent)
        completion_event.__class__ = RunCompletionEvent  # type: ignore[assignment]
        completion_event.content = MagicMock()
        completion_event.content.history = []
        completion_event.content.summary = ""
        completion_event.content.cost = {}
        completion_event.content.last_speaker = ""
        completion_event.content.context_variables = None

        # Filter on TerminationEvent, but InputRequestEvent should still be yielded
        response = self._create_run_iter_response(
            [input_event, completion_event],
            yield_on=[TerminationEvent],
        )

        events = list(response)

        assert len(events) == 1
        assert events[0] is input_event
        # The respond callback should be set
        assert hasattr(events[0].content, "respond")  # type: ignore[attr-defined]

    def test_properties_populated_after_completion(self) -> None:
        """Properties should be populated after iteration completes."""
        completion_event = MagicMock(spec=RunCompletionEvent)
        completion_event.__class__ = RunCompletionEvent  # type: ignore[assignment]
        completion_event.content = MagicMock()
        completion_event.content.history = [{"role": "user", "content": "test"}]
        completion_event.content.summary = "Test summary"
        completion_event.content.cost = {"total_cost": 0.01}
        completion_event.content.last_speaker = "assistant"
        completion_event.content.context_variables = {"key": "value"}

        response = self._create_run_iter_response([completion_event])

        # Before iteration
        assert response.summary is None
        assert len(response.messages) == 0

        # Iterate
        list(response)

        # After iteration
        assert response.summary == "Test summary"
        assert len(response.messages) == 1
        assert response.last_speaker == "assistant"


class TestAsyncRunIterResponse:
    """Tests for AsyncRunIterResponse async iterator-based stepping.

    Note: AsyncRunIterResponse now uses threads internally (same as sync RunIterResponse)
    to avoid needing async iostream.send() which would require runtime type checks.
    """

    def _create_async_run_iter_response(
        self,
        events: list[Any],
        yield_on: list[type] | None = None,
    ) -> AsyncRunIterResponse:
        """Create an AsyncRunIterResponse that will yield the given events.

        Args:
            events: List of events to put in the queue.
            yield_on: Event types to filter on.

        Returns:
            AsyncRunIterResponse configured to yield the events.
        """
        event_queue: queue.Queue[Any] = queue.Queue()
        for event in events:
            event_queue.put(event)

        def start_thread_func(iostream: ThreadIOStream) -> threading.Thread:
            def producer() -> None:
                assert iostream._step_controller is not None
                while True:
                    try:
                        event = event_queue.get_nowait()
                        iostream._input_stream.put(event)
                        # Wait for step to continue (simulates real behavior)
                        iostream._step_controller.wait_for_step(event)
                    except queue.Empty:
                        break

            return threading.Thread(target=producer, daemon=True)

        return AsyncRunIterResponse(
            start_thread_func=start_thread_func,
            yield_on=yield_on,
            agents=[],
        )

    @pytest.mark.asyncio
    async def test_iteration_yields_events_until_completion(self) -> None:
        """Events are yielded until RunCompletionEvent is received."""
        text_event = MagicMock(spec=TextEvent)
        text_event.__class__ = TextEvent  # type: ignore[assignment]

        completion_event = MagicMock(spec=RunCompletionEvent)
        completion_event.__class__ = RunCompletionEvent  # type: ignore[assignment]
        completion_event.content = MagicMock()
        completion_event.content.history = []
        completion_event.content.summary = "Test summary"
        completion_event.content.cost = {}
        completion_event.content.last_speaker = "test"
        completion_event.content.context_variables = None

        response = self._create_async_run_iter_response([text_event, completion_event])

        events = [event async for event in response]

        assert len(events) == 1
        assert events[0] is text_event
        assert response.summary == "Test summary"

    @pytest.mark.asyncio
    async def test_iteration_raises_on_error_event(self) -> None:
        """Iteration should raise the error from ErrorEvent."""
        test_error = ValueError("Test error message")
        error_event = MagicMock(spec=ErrorEvent)
        error_event.__class__ = ErrorEvent  # type: ignore[assignment]
        error_event.content = MagicMock()
        error_event.content.error = test_error

        response = self._create_async_run_iter_response([error_event])

        with pytest.raises(ValueError, match="Test error message"):
            _ = [event async for event in response]

    @pytest.mark.asyncio
    async def test_break_terminates_step_controller(self) -> None:
        """Breaking from iteration should terminate the step controller via explicit close."""
        text_event1 = MagicMock(spec=TextEvent)
        text_event1.__class__ = TextEvent  # type: ignore[assignment]
        text_event2 = MagicMock(spec=TextEvent)
        text_event2.__class__ = TextEvent  # type: ignore[assignment]

        completion_event = MagicMock(spec=RunCompletionEvent)
        completion_event.__class__ = RunCompletionEvent  # type: ignore[assignment]
        completion_event.content = MagicMock()
        completion_event.content.history = []
        completion_event.content.summary = ""
        completion_event.content.cost = {}
        completion_event.content.last_speaker = ""
        completion_event.content.context_variables = None

        response = self._create_async_run_iter_response([text_event1, text_event2, completion_event])

        # Use explicit iterator so we can close it properly
        iterator = aiter(response)
        await anext(iterator)  # Get first event
        await iterator.aclose()  # type: ignore[attr-defined]  # Explicitly close the async generator

        # Generator cleanup should have terminated the controller
        assert response._step_controller._terminated

    @pytest.mark.asyncio
    async def test_exception_terminates_step_controller(self) -> None:
        """Exception during iteration should terminate the step controller via explicit close."""
        text_event = MagicMock(spec=TextEvent)
        text_event.__class__ = TextEvent  # type: ignore[assignment]

        completion_event = MagicMock(spec=RunCompletionEvent)
        completion_event.__class__ = RunCompletionEvent  # type: ignore[assignment]
        completion_event.content = MagicMock()
        completion_event.content.history = []
        completion_event.content.summary = ""
        completion_event.content.cost = {}
        completion_event.content.last_speaker = ""
        completion_event.content.context_variables = None

        response = self._create_async_run_iter_response([text_event, completion_event])

        # Use explicit iterator so we can close it properly after exception
        iterator = aiter(response)
        with pytest.raises(ValueError):
            await anext(iterator)
            raise ValueError("Test exception")

        # Explicitly close the async generator after exception
        await iterator.aclose()  # type: ignore[attr-defined]

        # Generator cleanup should have terminated the controller
        assert response._step_controller._terminated

    @pytest.mark.asyncio
    async def test_lazy_start(self) -> None:
        """Thread should only start on first iteration."""
        completion_event = MagicMock(spec=RunCompletionEvent)
        completion_event.__class__ = RunCompletionEvent  # type: ignore[assignment]
        completion_event.content = MagicMock()
        completion_event.content.history = []
        completion_event.content.summary = ""
        completion_event.content.cost = {}
        completion_event.content.last_speaker = ""
        completion_event.content.context_variables = None

        response = self._create_async_run_iter_response([completion_event])

        # Before iteration
        assert not response._started
        assert response._thread is None

        # Start iteration
        _ = [event async for event in response]

        # After iteration
        assert response._started
        assert response._thread is not None

    @pytest.mark.asyncio
    async def test_yield_on_filters_events(self) -> None:
        """Only events matching yield_on should be yielded."""
        text_event = MagicMock(spec=TextEvent)
        # Don't set __class__ so it won't match filter

        termination_event = MagicMock(spec=TerminationEvent)
        termination_event.__class__ = TerminationEvent  # type: ignore[assignment]

        completion_event = MagicMock(spec=RunCompletionEvent)
        completion_event.__class__ = RunCompletionEvent  # type: ignore[assignment]
        completion_event.content = MagicMock()
        completion_event.content.history = []
        completion_event.content.summary = ""
        completion_event.content.cost = {}
        completion_event.content.last_speaker = ""
        completion_event.content.context_variables = None

        response = self._create_async_run_iter_response(
            [text_event, termination_event, completion_event],
            yield_on=[TerminationEvent],
        )

        events = [event async for event in response]

        # Only TerminationEvent should be yielded
        assert len(events) == 1
        assert events[0] is termination_event
