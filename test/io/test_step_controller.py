# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for StepController and AsyncStepController."""

import asyncio
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from autogen.events.agent_events import ErrorEvent, InputRequestEvent, RunCompletionEvent, TerminationEvent, TextEvent
from autogen.io.run_response import AsyncRunResponse, RunResponse
from autogen.io.step_controller import AsyncStepController, StepController


class TestStepController:
    """Tests for the synchronous StepController."""

    def test_should_block_no_filter_blocks_all_events(self) -> None:
        """When step_on is None, should_block returns True for all events."""
        controller = StepController(step_on=None)

        # Create mock events of different types
        text_event = MagicMock(spec=TextEvent)
        termination_event = MagicMock(spec=TerminationEvent)

        assert controller.should_block(text_event) is True
        assert controller.should_block(termination_event) is True

    def test_should_block_with_filter_only_blocks_specified_types(self) -> None:
        """When step_on is specified, should_block returns True only for those types."""
        controller = StepController(step_on=[TextEvent, TerminationEvent])

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
        controller = StepController(step_on=None)
        text_event = MagicMock(spec=TextEvent)

        # Before terminate
        assert controller.should_block(text_event) is True

        # After terminate
        controller.terminate()
        assert controller.should_block(text_event) is False

    def test_step_unblocks_wait_for_step(self) -> None:
        """Calling step() should unblock a thread waiting on wait_for_step()."""
        controller = StepController(step_on=None)
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
        controller = StepController(step_on=[TerminationEvent])

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
        controller = StepController(step_on=None)
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
        """When step_on is None, should_block returns True for all events."""
        controller = AsyncStepController(step_on=None)

        text_event = MagicMock(spec=TextEvent)
        termination_event = MagicMock(spec=TerminationEvent)

        assert controller.should_block(text_event) is True
        assert controller.should_block(termination_event) is True

    def test_should_block_with_filter_only_blocks_specified_types(self) -> None:
        """When step_on is specified, should_block returns True only for those types."""
        controller = AsyncStepController(step_on=[TextEvent])

        text_event = MagicMock(spec=TextEvent)
        text_event.__class__ = TextEvent  # type: ignore[assignment]
        assert controller.should_block(text_event) is True

        other_event = MagicMock()
        assert controller.should_block(other_event) is False

    def test_should_block_after_terminate_returns_false(self) -> None:
        """After terminate() is called, should_block always returns False."""
        controller = AsyncStepController(step_on=None)
        text_event = MagicMock(spec=TextEvent)

        assert controller.should_block(text_event) is True
        controller.terminate()
        assert controller.should_block(text_event) is False

    @pytest.mark.asyncio
    async def test_step_unblocks_wait_for_step(self) -> None:
        """Calling step() should unblock wait_for_step()."""
        controller = AsyncStepController(step_on=None)
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
        controller = AsyncStepController(step_on=[TerminationEvent])

        text_event = MagicMock(spec=TextEvent)

        # This should return immediately
        start = time.time()
        await controller.wait_for_step(text_event)
        elapsed = time.time() - start

        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_terminate_unblocks_waiting_producer(self) -> None:
        """terminate() should unblock any task waiting on wait_for_step()."""
        controller = AsyncStepController(step_on=None)
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


class TestRunResponseStep:
    """Tests for RunResponse.step() method."""

    def _create_run_response(self, step_controller: StepController | None = None) -> RunResponse:
        """Create a RunResponse with a mock iostream."""
        mock_iostream = MagicMock()
        mock_iostream._input_stream = MagicMock()
        mock_iostream._output_stream = MagicMock()
        mock_agents: list[Any] = []
        return RunResponse(mock_iostream, mock_agents, step_controller=step_controller)

    def test_step_without_step_mode_raises_error(self) -> None:
        """step() should raise RuntimeError when step_mode is not enabled."""
        response = self._create_run_response(step_controller=None)

        with pytest.raises(RuntimeError, match="step\\(\\) requires step_mode=True"):
            response.step()

    def test_step_returns_none_on_completion(self) -> None:
        """step() should return None when RunCompletionEvent is received."""
        controller = StepController(step_on=None)
        response = self._create_run_response(step_controller=controller)

        # Mock the completion event
        completion_event = MagicMock(spec=RunCompletionEvent)
        completion_event.__class__ = RunCompletionEvent  # type: ignore[assignment]
        completion_event.content = MagicMock()
        completion_event.content.history = []
        completion_event.content.summary = "Test summary"
        completion_event.content.cost = {}
        completion_event.content.last_speaker = "test"
        completion_event.content.context_variables = None

        response.iostream._input_stream.get.return_value = completion_event  # type: ignore[attr-defined]

        result = response.step()
        assert result is None
        assert response.summary == "Test summary"

    def test_step_raises_on_error_event(self) -> None:
        """step() should raise the error from ErrorEvent."""
        controller = StepController(step_on=None)
        response = self._create_run_response(step_controller=controller)

        test_error = ValueError("Test error message")
        error_event = MagicMock(spec=ErrorEvent)
        error_event.__class__ = ErrorEvent  # type: ignore[assignment]
        error_event.content = MagicMock()
        error_event.content.error = test_error

        response.iostream._input_stream.get.return_value = error_event  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="Test error message"):
            response.step()

    def test_step_returns_text_event(self) -> None:
        """step() should return TextEvent when one is received."""
        controller = StepController(step_on=None)
        response = self._create_run_response(step_controller=controller)

        text_event = MagicMock(spec=TextEvent)
        text_event.__class__ = TextEvent  # type: ignore[assignment]
        response.iostream._input_stream.get.return_value = text_event  # type: ignore[attr-defined]

        result = response.step()
        assert result is text_event

    def test_step_handles_input_request_event(self) -> None:
        """step() should set up respond callback for InputRequestEvent."""
        controller = StepController(step_on=None)
        response = self._create_run_response(step_controller=controller)

        input_event = MagicMock(spec=InputRequestEvent)
        input_event.__class__ = InputRequestEvent  # type: ignore[assignment]
        input_event.content = MagicMock()

        response.iostream._input_stream.get.return_value = input_event  # type: ignore[attr-defined]

        result = response.step()

        assert result is input_event
        # The respond callback should be set
        assert result is not None and hasattr(result.content, "respond")  # type: ignore[attr-defined]

    def test_step_filters_events_based_on_step_on(self) -> None:
        """step() should skip events not in step_on filter."""
        controller = StepController(step_on=[TerminationEvent])
        response = self._create_run_response(step_controller=controller)

        # First call returns TextEvent (not in filter), second returns TerminationEvent
        text_event = MagicMock(spec=TextEvent)
        # Don't set __class__ so it won't match filter

        termination_event = MagicMock(spec=TerminationEvent)
        termination_event.__class__ = TerminationEvent  # type: ignore[assignment]

        response.iostream._input_stream.get.side_effect = [text_event, termination_event]  # type: ignore[attr-defined]

        # Should skip TextEvent and return TerminationEvent
        result = response.step()
        assert result is termination_event


class TestRunResponseContextManager:
    """Tests for RunResponse context manager support."""

    def _create_run_response(self, step_controller: StepController | None = None) -> RunResponse:
        """Create a RunResponse with a mock iostream."""
        mock_iostream = MagicMock()
        mock_iostream._input_stream = MagicMock()
        mock_iostream._output_stream = MagicMock()
        mock_agents: list[Any] = []
        return RunResponse(mock_iostream, mock_agents, step_controller=step_controller)

    def test_context_manager_calls_close_on_exit(self) -> None:
        """Context manager should call close() on exit."""
        controller = StepController(step_on=None)
        response = self._create_run_response(step_controller=controller)

        with response:
            assert not controller._terminated

        assert controller._terminated

    def test_context_manager_calls_close_on_exception(self) -> None:
        """Context manager should call close() even when exception occurs."""
        controller = StepController(step_on=None)
        response = self._create_run_response(step_controller=controller)

        with pytest.raises(ValueError), response:
            assert not controller._terminated
            raise ValueError("Test exception")

        assert controller._terminated

    def test_close_is_idempotent(self) -> None:
        """Calling close() multiple times should be safe."""
        controller = StepController(step_on=None)
        response = self._create_run_response(step_controller=controller)

        response.close()
        response.close()  # Should not raise

        assert controller._terminated

    def test_close_without_step_controller_is_safe(self) -> None:
        """close() should be safe when no step controller is present."""
        response = self._create_run_response(step_controller=None)
        response.close()  # Should not raise


class TestAsyncRunResponseStep:
    """Tests for AsyncRunResponse.step() method."""

    def _create_async_run_response(self, step_controller: AsyncStepController | None = None) -> AsyncRunResponse:
        """Create an AsyncRunResponse with a mock iostream."""
        mock_iostream = MagicMock()
        mock_iostream._input_stream = MagicMock()
        mock_iostream._output_stream = MagicMock()
        mock_agents: list[Any] = []
        return AsyncRunResponse(mock_iostream, mock_agents, step_controller=step_controller)

    @pytest.mark.asyncio
    async def test_step_without_step_mode_raises_error(self) -> None:
        """step() should raise RuntimeError when step_mode is not enabled."""
        response = self._create_async_run_response(step_controller=None)

        with pytest.raises(RuntimeError, match="step\\(\\) requires step_mode=True"):
            await response.step()

    @pytest.mark.asyncio
    async def test_step_returns_none_on_completion(self) -> None:
        """step() should return None when RunCompletionEvent is received."""
        controller = AsyncStepController(step_on=None)
        response = self._create_async_run_response(step_controller=controller)

        completion_event = MagicMock(spec=RunCompletionEvent)
        completion_event.__class__ = RunCompletionEvent  # type: ignore[assignment]
        completion_event.content = MagicMock()
        completion_event.content.history = []
        completion_event.content.summary = "Test summary"
        completion_event.content.cost = {}
        completion_event.content.last_speaker = "test"
        completion_event.content.context_variables = None

        async def mock_get() -> Any:
            return completion_event

        response.iostream._input_stream.get = mock_get  # type: ignore[method-assign]

        result = await response.step()
        assert result is None

    @pytest.mark.asyncio
    async def test_step_raises_on_error_event(self) -> None:
        """step() should raise the error from ErrorEvent."""
        controller = AsyncStepController(step_on=None)
        response = self._create_async_run_response(step_controller=controller)

        test_error = ValueError("Test error message")
        error_event = MagicMock(spec=ErrorEvent)
        error_event.__class__ = ErrorEvent  # type: ignore[assignment]
        error_event.content = MagicMock()
        error_event.content.error = test_error

        async def mock_get() -> Any:
            return error_event

        response.iostream._input_stream.get = mock_get  # type: ignore[method-assign]

        with pytest.raises(ValueError, match="Test error message"):
            await response.step()


class TestAsyncRunResponseContextManager:
    """Tests for AsyncRunResponse async context manager support."""

    def _create_async_run_response(self, step_controller: AsyncStepController | None = None) -> AsyncRunResponse:
        """Create an AsyncRunResponse with a mock iostream."""
        mock_iostream = MagicMock()
        mock_iostream._input_stream = MagicMock()
        mock_iostream._output_stream = MagicMock()
        mock_agents: list[Any] = []
        return AsyncRunResponse(mock_iostream, mock_agents, step_controller=step_controller)

    @pytest.mark.asyncio
    async def test_async_context_manager_calls_close_on_exit(self) -> None:
        """Async context manager should call close() on exit."""
        controller = AsyncStepController(step_on=None)
        response = self._create_async_run_response(step_controller=controller)

        async with response:
            assert not controller._terminated

        assert controller._terminated

    @pytest.mark.asyncio
    async def test_async_context_manager_calls_close_on_exception(self) -> None:
        """Async context manager should call close() even when exception occurs."""
        controller = AsyncStepController(step_on=None)
        response = self._create_async_run_response(step_controller=controller)

        with pytest.raises(ValueError):
            async with response:
                assert not controller._terminated
                raise ValueError("Test exception")

        assert controller._terminated
