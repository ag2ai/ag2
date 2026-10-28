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
from a2a.server.events import Event, EventQueue
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
from ag2.a2a.mappers.messages import build_user_message
from ag2.events import BaseEvent, ModelMessageChunk, TextInput, ToolCallEvent
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


@pytest.mark.asyncio
class TestCancelRacingTheTurn:
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
        started = asyncio.Event()
        release = asyncio.Event()
        mirrored: list[TaskState] = []

        async def record_status(event: BaseEvent) -> None:
            assert isinstance(event, A2ATaskStatusUpdate)
            mirrored.append(event.state)

        async def hold(context: Context) -> str:
            context.stream.where(A2ATaskStatusUpdate).subscribe(record_status)
            started.set()
            await release.wait()
            return "held"

        config = TestConfig(ToolCallEvent(name="hold", arguments="{}"), "done")
        executor = AgentExecutor(Agent("server", config=config, tools=[hold]))
        queue = _RecordingQueue()

        running = asyncio.create_task(executor.execute(_request(), queue))
        await started.wait()
        await executor.cancel(_cancel_request(), queue)
        release.set()
        await running

        assert queue.states() == [TaskState.TASK_STATE_WORKING, TaskState.TASK_STATE_CANCELED]
        # Server-side observers see the same lifecycle as the wire, not the completion that lost.
        assert mirrored == [TaskState.TASK_STATE_CANCELED]

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
