# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``AgentExecutor.cancel`` racing the turn's own terminal transition.

The SDK calls ``cancel`` without synchronising with ``execute``: it only checks that its consumer
has not yet processed a terminal event, so a cancel can land after the turn has already completed
(or a turn can try to complete after a cancel ended it). These tests drive the executor directly
on a recording queue to put each side into that window deterministically.
"""

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import Event, EventQueue, EventQueueLegacy
from a2a.types import (
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)

from ag2 import Agent, Context
from ag2.a2a.events import A2ATaskStatusUpdate
from ag2.a2a.executor import AgentExecutor
from ag2.a2a.mappers.events import chunk_to_text_artifact
from ag2.a2a.mappers.messages import build_user_message
from ag2.a2a.testing import ObservedAgentExecutor, TaskStatusRecorder, TurnGate
from ag2.events import ClientToolCallEvent, ModelMessageChunk, TextInput, ToolCallEvent
from ag2.observers import observer
from ag2.testing import TestConfig
from ag2.tools.final.function_tool import FunctionDefinition, FunctionToolSchema

_TASK_ID = "task-1"
_CONTEXT_ID = "ctx-1"
_TEXT_CLOSED = "text closed"


class _RecordingQueue(EventQueue):
    def __init__(self) -> None:
        self.events: list[Event] = []
        self.on_event: Callable[[Event], Awaitable[None]] | None = None

    async def enqueue_event(self, event: Event) -> None:
        # The hook runs before the event lands, as a slow enqueue would.
        if self.on_event is not None:
            await self.on_event(event)
        self.events.append(event)

    def states(self) -> list[TaskState]:
        return [e.status.state for e in self.events if isinstance(e, TaskStatusUpdateEvent)]

    def lifecycle(self) -> list[TaskState | str]:
        """Status transitions plus each close of streamed text, in wire order."""
        return [
            e.status.state if isinstance(e, TaskStatusUpdateEvent) else _TEXT_CLOSED
            for e in self.events
            if isinstance(e, TaskStatusUpdateEvent) or (isinstance(e, TaskArtifactUpdateEvent) and e.append)
        ]


def _request() -> RequestContext:
    message = Message(
        role=Role.ROLE_USER,
        parts=[Part(text="go")],
        message_id="msg-1",
        task_id=_TASK_ID,
        context_id=_CONTEXT_ID,
    )
    return RequestContext(ServerCallContext(), request=SendMessageRequest(message=message))


def _client_tool_request() -> RequestContext:
    message = build_user_message(
        [TextInput("go")],
        tool_schemas=[FunctionToolSchema(function=FunctionDefinition(name="lookup"))],
        task_id=_TASK_ID,
        context_id=_CONTEXT_ID,
        message_id="msg-1",
    )
    return RequestContext(ServerCallContext(), request=SendMessageRequest(message=message))


def _cancel_request() -> RequestContext:
    return RequestContext(ServerCallContext(), task_id=_TASK_ID, context_id=_CONTEXT_ID)


class _CancelOnStatus:
    """Lifecycle observer that cancels the task once it reaches ``state``, then raises ``error`` if given."""

    def __init__(
        self,
        state: TaskState,
        queue: _RecordingQueue,
        *,
        spawn_cancel: bool = False,
        error: Exception | None = None,
    ) -> None:
        self.state = state
        self.queue = queue
        self.spawn_cancel = spawn_cancel
        self.error = error
        self.executor: AgentExecutor | None = None
        self.observed: list[TaskState] = []

    async def on_status(self, event: A2ATaskStatusUpdate) -> None:
        self.observed.append(event.state)
        if event.state != self.state:
            return
        assert self.executor is not None
        cancellation = self.executor.cancel(_cancel_request(), self.queue)
        if self.spawn_cancel:
            await asyncio.create_task(cancellation)
        else:
            await cancellation
        if self.error is not None:
            raise self.error


class _PauseOnStatus:
    """Lifecycle observer that blocks delivery of ``state`` until resumed."""

    def __init__(self, state: TaskState) -> None:
        self.state = state
        self.paused = asyncio.Event()
        self._resumed = asyncio.Event()

    async def on_status(self, event: A2ATaskStatusUpdate) -> None:
        if event.state == self.state:
            self.paused.set()
            await self._resumed.wait()

    def resume(self) -> None:
        self._resumed.set()


class _PausingArtifactQueue(_RecordingQueue):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def enqueue_event(self, event: Event) -> None:
        if isinstance(event, TaskArtifactUpdateEvent) and not self.started.is_set():
            self.started.set()
            await self.release.wait()
        await super().enqueue_event(event)


class _DropAndCancel:
    """Interrupting lifecycle observer that drops the first ``state`` update and cancels the task."""

    def __init__(self, state: TaskState, queue: _RecordingQueue) -> None:
        self.state = state
        self.queue = queue
        self.executor: AgentExecutor | None = None
        self.pending: list[asyncio.Task[None]] = []

    async def on_status(self, event: A2ATaskStatusUpdate) -> A2ATaskStatusUpdate | None:
        if event.state != self.state or self.pending:
            return event
        assert self.executor is not None
        self.pending.append(asyncio.create_task(self.executor.cancel(_cancel_request(), self.queue)))
        return None


class _SendClientCall:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.sender: asyncio.Task[None] | None = None
        self.recovered = False

    async def ask_remote(self, context: Context) -> str:
        self.started.set()
        await self.release.wait()
        self.sender = asyncio.create_task(context.send(ClientToolCallEvent(name="lookup", arguments="{}")))
        try:
            await self.sender
        except asyncio.CancelledError:
            self.recovered = True
        return "fallback"


async def _drain_queue(queue: EventQueueLegacy) -> None:
    while True:
        await queue.dequeue_event()
        queue.task_done()


@pytest.mark.asyncio
class TestCancelRacingTheTurn:
    @pytest.mark.parametrize("cancel_sender", [False, True])
    async def test_published_client_call_is_kept_when_sender_is_canceled(self, cancel_sender: bool) -> None:
        tool = _SendClientCall()
        config = TestConfig(ToolCallEvent(name="ask_remote", arguments="{}"), "done")
        executor = AgentExecutor(Agent("server", config=config, tools=[tool.ask_remote]))
        queue = EventQueueLegacy()
        running = asyncio.create_task(executor.execute(_client_tool_request(), queue))
        draining: asyncio.Task[None] | None = None
        events: list[Event] = []
        try:
            await asyncio.wait_for(tool.started.wait(), timeout=1.0)
            child = await queue.tap(max_queue_size=1)
            await child.enqueue_event(Message(message_id="prefill"))
            tool.release.set()
            while True:
                event = await asyncio.wait_for(queue.dequeue_event(), timeout=1.0)
                queue.task_done()
                events.append(event)
                if isinstance(event, TaskArtifactUpdateEvent):
                    break

            assert tool.sender is not None
            if cancel_sender:
                tool.sender.cancel()
            draining = asyncio.create_task(_drain_queue(child))
            await asyncio.wait_for(running, timeout=1.0)
            while not queue.queue.empty():
                events.append(await queue.dequeue_event())
                queue.task_done()
        finally:
            tool.release.set()
            await queue.close(immediate=True)
            tasks = [task for task in (running, draining, tool.sender) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        assert tool.recovered is cancel_sender
        assert [e.status.state for e in events if isinstance(e, TaskStatusUpdateEvent)] == [
            TaskState.TASK_STATE_WORKING,
            TaskState.TASK_STATE_INPUT_REQUIRED,
        ]
        assert len([e for e in events if isinstance(e, TaskArtifactUpdateEvent)]) == 1

    @pytest.mark.parametrize("kind", ["text", "status"])
    @pytest.mark.parametrize("cancel_count", [1, 2])
    async def test_cancel_after_partial_enqueue_finishes_wire_commit(self, kind: str, cancel_count: int) -> None:
        gate = TurnGate()
        statuses = TaskStatusRecorder()
        config = TestConfig(ToolCallEvent(name="hold", arguments="{}"), ModelMessageChunk("thinking"), "thinking")
        executor = ObservedAgentExecutor(Agent("server", config=config, tools=[gate.hold]), statuses)
        queue = EventQueueLegacy()
        running = asyncio.create_task(executor.execute(_request(), queue))
        canceling: asyncio.Task[None] | None = None
        draining: asyncio.Task[None] | None = None
        events: list[Event] = []
        try:
            await asyncio.wait_for(gate.entered.wait(), timeout=1.0)
            child = await queue.tap(max_queue_size=1)
            if kind == "text":
                await child.enqueue_event(Message(message_id="prefill"))
            gate.release()

            while True:
                event = await asyncio.wait_for(queue.dequeue_event(), timeout=1.0)
                queue.task_done()
                events.append(event)
                if isinstance(event, TaskArtifactUpdateEvent):
                    if kind == "text":
                        break
                    if event.last_chunk:
                        await child.dequeue_event()
                        child.task_done()
                if isinstance(event, TaskStatusUpdateEvent) and event.status.state == TaskState.TASK_STATE_COMPLETED:
                    break

            if kind == "text":
                canceling = asyncio.create_task(executor.cancel(_cancel_request(), queue))
                await asyncio.sleep(0)
            for _ in range(cancel_count):
                running.cancel()
                await asyncio.sleep(0)

            draining = asyncio.create_task(_drain_queue(child))
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(running, timeout=1.0)
            if canceling is not None:
                await asyncio.wait_for(canceling, timeout=1.0)
            expected = TaskState.TASK_STATE_CANCELED if kind == "text" else TaskState.TASK_STATE_COMPLETED
            await asyncio.wait_for(statuses.wait_for(expected), timeout=1.0)
            while not queue.queue.empty():
                events.append(await queue.dequeue_event())
                queue.task_done()
        finally:
            gate.release()
            await queue.close(immediate=True)
            tasks = [task for task in (running, canceling, draining) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        assert statuses.states == [TaskState.TASK_STATE_WORKING, expected]
        assert [e.last_chunk for e in events if isinstance(e, TaskArtifactUpdateEvent)] == [False, True]
        assert [e.status.state for e in events if isinstance(e, TaskStatusUpdateEvent)] == [
            TaskState.TASK_STATE_WORKING,
            expected,
        ]

    @pytest.mark.parametrize("eager", [False, True], ids=["default-factory", "eager-factory"])
    async def test_filtered_notification_does_not_stall_later_notifications(self, eager: bool) -> None:
        eager_factory = getattr(asyncio, "eager_task_factory", None)
        if eager and eager_factory is None:
            pytest.skip("eager_task_factory requires Python 3.12+")
        loop = asyncio.get_running_loop()
        previous_factory = loop.get_task_factory()
        queue = _RecordingQueue()
        dropper = _DropAndCancel(TaskState.TASK_STATE_INPUT_REQUIRED, queue)
        statuses = TaskStatusRecorder()
        executor = ObservedAgentExecutor(
            Agent("server", config=TestConfig(ToolCallEvent(name="lookup", arguments="{}"))),
            observer(A2ATaskStatusUpdate, dropper.on_status, interrupt=True),
            statuses,
        )
        dropper.executor = executor
        try:
            loop.set_task_factory(eager_factory if eager else None)
            await asyncio.wait_for(executor.execute(_client_tool_request(), queue), timeout=1.0)
            await asyncio.wait_for(asyncio.gather(*dropper.pending), timeout=1.0)
        finally:
            loop.set_task_factory(previous_factory)

        assert queue.states() == [
            TaskState.TASK_STATE_WORKING,
            TaskState.TASK_STATE_INPUT_REQUIRED,
            TaskState.TASK_STATE_CANCELED,
        ]
        assert statuses.states == [TaskState.TASK_STATE_WORKING, TaskState.TASK_STATE_CANCELED]

    @pytest.mark.parametrize("spawn_cancel", [False, True], ids=["inline", "child-task"])
    @pytest.mark.parametrize("state", [TaskState.TASK_STATE_INPUT_REQUIRED, TaskState.TASK_STATE_COMPLETED])
    async def test_lifecycle_observer_can_await_cancellation(self, state: TaskState, spawn_cancel: bool) -> None:
        queue = _RecordingQueue()
        canceler = _CancelOnStatus(state, queue, spawn_cancel=spawn_cancel)
        statuses = TaskStatusRecorder()
        answer = (
            ToolCallEvent(name="lookup", arguments="{}") if state == TaskState.TASK_STATE_INPUT_REQUIRED else "done"
        )
        executor = ObservedAgentExecutor(
            Agent("server", config=TestConfig(ModelMessageChunk("thinking"), answer)),
            observer(A2ATaskStatusUpdate, canceler.on_status),
            statuses,
        )
        canceler.executor = executor

        await asyncio.wait_for(executor.execute(_client_tool_request(), queue), timeout=1.0)

        expected = [TaskState.TASK_STATE_WORKING, state]
        if state == TaskState.TASK_STATE_INPUT_REQUIRED:
            expected.append(TaskState.TASK_STATE_CANCELED)
        assert queue.states() == expected
        assert canceler.observed == expected
        assert statuses.states == expected

    async def test_producer_cancellation_does_not_drop_queued_terminal_notification(self) -> None:
        queue = _RecordingQueue()
        pause = _PauseOnStatus(TaskState.TASK_STATE_INPUT_REQUIRED)
        statuses = TaskStatusRecorder()
        executor = ObservedAgentExecutor(
            Agent("server", config=TestConfig(ToolCallEvent(name="lookup", arguments="{}"))),
            observer(A2ATaskStatusUpdate, pause.on_status),
            statuses,
        )
        running = asyncio.create_task(executor.execute(_client_tool_request(), queue))
        try:
            await asyncio.wait_for(pause.paused.wait(), timeout=1.0)
            await asyncio.wait_for(executor.cancel(_cancel_request(), queue), timeout=1.0)
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        finally:
            pause.resume()
            if not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)

        await asyncio.wait_for(statuses.wait_for(TaskState.TASK_STATE_CANCELED), timeout=1.0)
        assert statuses.states == [
            TaskState.TASK_STATE_WORKING,
            TaskState.TASK_STATE_INPUT_REQUIRED,
            TaskState.TASK_STATE_CANCELED,
        ]

    async def test_observer_failure_does_not_drop_queued_terminal_notification(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        queue = _RecordingQueue()
        canceler = _CancelOnStatus(
            TaskState.TASK_STATE_INPUT_REQUIRED, queue, error=ValueError("observer failed after canceling")
        )
        statuses = TaskStatusRecorder()
        executor = ObservedAgentExecutor(
            Agent("server", config=TestConfig(ToolCallEvent(name="lookup", arguments="{}"))),
            statuses,
            observer(A2ATaskStatusUpdate, canceler.on_status),
        )
        canceler.executor = executor

        # The transitions are on the wire, so the observer's failure is logged rather than raised.
        await asyncio.wait_for(executor.execute(_client_tool_request(), queue), timeout=1.0)

        assert "observer failed after canceling" in caplog.text
        assert queue.states() == [
            TaskState.TASK_STATE_WORKING,
            TaskState.TASK_STATE_INPUT_REQUIRED,
            TaskState.TASK_STATE_CANCELED,
        ]
        assert statuses.states == canceler.observed == queue.states()

    @pytest.mark.parametrize("kind", ["text", "tool-call", "custom-artifact"])
    async def test_cancel_waits_for_an_inflight_artifact(self, kind: str) -> None:
        if kind == "text":
            event = ModelMessageChunk("thinking")
        elif kind == "tool-call":
            event = ToolCallEvent(name="lookup", arguments="{}")
        else:
            event = chunk_to_text_artifact(
                ModelMessageChunk("custom"),
                artifact_id="custom",
                task_id=_TASK_ID,
                context_id=_CONTEXT_ID,
                append=False,
            )
        executor = AgentExecutor(Agent("server", config=TestConfig(event, "done")))
        queue = _PausingArtifactQueue()
        running = asyncio.create_task(executor.execute(_client_tool_request(), queue))
        canceling: asyncio.Task[None] | None = None
        try:
            await asyncio.wait_for(queue.started.wait(), timeout=1.0)
            canceling = asyncio.create_task(executor.cancel(_cancel_request(), queue))
            await asyncio.sleep(0)
            queue.release.set()
            await asyncio.wait_for(asyncio.gather(running, canceling), timeout=1.0)
        finally:
            queue.release.set()
            pending = [task for task in (running, canceling) if task is not None and not task.done()]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        assert queue.states() == [TaskState.TASK_STATE_WORKING, TaskState.TASK_STATE_CANCELED]
        assert isinstance(queue.events[-1], TaskStatusUpdateEvent)

    async def test_cancel_landing_after_the_turn_completed_leaves_it_completed(self) -> None:
        executor = AgentExecutor(Agent("server", config=TestConfig("done")))
        queue = _RecordingQueue()
        cancels: list[asyncio.Task[None]] = []

        async def cancel_on_completion(event: Event) -> None:
            if isinstance(event, TaskStatusUpdateEvent) and event.status.state == TaskState.TASK_STATE_COMPLETED:
                cancels.append(asyncio.create_task(executor.cancel(_cancel_request(), queue)))
                await asyncio.sleep(0)

        queue.on_event = cancel_on_completion

        await executor.execute(_request(), queue)
        await asyncio.gather(*cancels)

        assert queue.states() == [TaskState.TASK_STATE_WORKING, TaskState.TASK_STATE_COMPLETED]

    async def test_cancel_after_the_turn_finished_writes_nothing(self) -> None:
        # No turn is running here; the SDK closes a still-open task as CANCELED itself.
        executor = AgentExecutor(Agent("server", config=TestConfig("done")))
        queue = _RecordingQueue()
        await executor.execute(_request(), queue)
        finished = len(queue.events)

        await executor.cancel(_cancel_request(), queue)

        assert len(queue.events) == finished

    async def test_turn_finishing_after_a_cancel_ends_quietly(self) -> None:
        gate = TurnGate()
        statuses = TaskStatusRecorder()
        config = TestConfig(ToolCallEvent(name="hold", arguments="{}"), "done")
        executor = ObservedAgentExecutor(Agent("server", config=config, tools=[gate.hold]), statuses)
        queue = _RecordingQueue()

        running = asyncio.create_task(executor.execute(_request(), queue))
        await gate.entered.wait()
        await executor.cancel(_cancel_request(), queue)
        gate.release()
        await running

        assert queue.states() == [TaskState.TASK_STATE_WORKING, TaskState.TASK_STATE_CANCELED]
        # Server-side observers see the same lifecycle as the wire, not the completion that lost.
        assert statuses.states == [TaskState.TASK_STATE_WORKING, TaskState.TASK_STATE_CANCELED]

    async def test_cancel_landing_during_the_text_close_follows_input_required(self) -> None:
        # Asking for a client tool closes the streamed text before the task leaves ``working``; a
        # cancel arriving during that close waits for the input request, then cancels the task.
        config = TestConfig(ModelMessageChunk("thinking"), ToolCallEvent(name="lookup", arguments="{}"))
        executor = AgentExecutor(Agent("server", config=config))
        queue = _RecordingQueue()
        cancels: list[asyncio.Task[None]] = []

        async def cancel_during_close(event: Event) -> None:
            if isinstance(event, TaskArtifactUpdateEvent) and event.append and event.last_chunk:
                cancels.append(asyncio.create_task(executor.cancel(_cancel_request(), queue)))
                await asyncio.sleep(0)

        queue.on_event = cancel_during_close

        await executor.execute(_client_tool_request(), queue)
        await asyncio.gather(*cancels)

        # The text is closed once, before the task leaves ``working``.
        assert queue.lifecycle() == [
            TaskState.TASK_STATE_WORKING,
            _TEXT_CLOSED,
            TaskState.TASK_STATE_INPUT_REQUIRED,
            TaskState.TASK_STATE_CANCELED,
        ]
