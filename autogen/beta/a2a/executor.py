# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import InternalError, Message, Task, TaskState, TaskStatus
from a2a.utils.errors import ServerError
from a2a.utils.message import new_agent_text_message

from autogen.beta.events import HumanInputRequest, HumanMessage, ModelMessageChunk
from autogen.beta.stream import MemoryStream

from .mappers import (
    a2a_message_to_inputs,
    hitl_replay_queue,
    initial_inputs,
    input_required_message,
    text_parts,
)

if TYPE_CHECKING:
    from autogen.beta import Agent


class _InputRequiredSignal(Exception):  # noqa: N818  # control-flow signal, not a user-visible error
    """Internal signal raised by the replay HITL hook to suspend a task.

    `execute()` catches this and translates it into an A2A `requires_input(...)`
    event so the client can supply the answer. On the follow-up request, the
    executor replays the agent with the answer pre-loaded in the replay queue.
    """

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        super().__init__(f"Human input required: {prompt!r}")


class AG2AgentExecutor(AgentExecutor):
    __slots__ = ("_agent",)

    def __init__(self, agent: "Agent") -> None:
        self._agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.message is None:
            raise ServerError(error=InternalError(message="RequestContext.message is required"))

        task = context.current_task or _build_initial_task(context.message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        inputs = initial_inputs(task) or a2a_message_to_inputs(context.message)
        replay_queue = hitl_replay_queue(task)

        forwarder = _ChunkForwarder(updater)
        stream = MemoryStream()
        stream.where(ModelMessageChunk).subscribe(forwarder)

        try:
            reply = await self._agent.ask(
                *inputs,
                stream=stream,
                hitl_hook=_ReplayHook(replay_queue),
            )
        except _InputRequiredSignal as signal:
            await updater.requires_input(
                message=input_required_message(signal.prompt, context_id=task.context_id, task_id=task.id),
                final=True,
            )
            return
        except Exception as exc:
            await updater.failed(
                message=new_agent_text_message(
                    text=str(exc) or type(exc).__name__,
                    context_id=task.context_id,
                    task_id=task.id,
                ),
            )
            return

        await updater.add_artifact(
            parts=text_parts(reply.body or ""),
            name="result",
            append=forwarder.streaming_started,
            last_chunk=True,
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            return
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.cancel()


class _ChunkForwarder:
    __slots__ = ("_updater", "streaming_started")

    def __init__(self, updater: TaskUpdater) -> None:
        self._updater = updater
        self.streaming_started = False

    async def __call__(self, chunk: ModelMessageChunk) -> None:
        await self._updater.add_artifact(
            parts=text_parts(chunk.content),
            name="result",
            append=self.streaming_started,
            last_chunk=False,
        )
        self.streaming_started = True


class _ReplayHook:
    """HITL hook backed by a pre-recorded queue of human inputs.

    Each `context.input(...)` call pops the next entry from the queue. When
    the queue is empty, the hook raises `_InputRequiredSignal`, which the
    executor catches to suspend the task and request input from the client.
    """

    __slots__ = ("_iter",)

    def __init__(self, queue: list[str]) -> None:
        self._iter = iter(queue)

    async def __call__(self, request: HumanInputRequest) -> HumanMessage:
        try:
            answer = next(self._iter)
        except StopIteration:
            raise _InputRequiredSignal(request.content) from None
        return HumanMessage(answer, parent_id=request.id)


def _build_initial_task(message: Message) -> Task:
    return Task(
        id=message.task_id or uuid4().hex,
        context_id=message.context_id or uuid4().hex,
        status=TaskStatus(
            state=TaskState.submitted,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
        history=[message],
    )
