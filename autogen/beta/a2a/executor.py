# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor as A2ABaseAgentExecutor
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import InternalError, Message, Task, TaskState, TaskStatus
from a2a.utils.errors import ServerError

from autogen.beta.events import HumanInputRequest, HumanMessage, ModelMessageChunk
from autogen.beta.stream import MemoryStream

from .errors import InputRequiredError
from .mappers import (
    a2a_message_to_inputs,
    hitl_replay_queue,
    initial_inputs,
    input_required_message,
    text_parts,
)
from .utils import RESULT_ARTIFACT_NAME

if TYPE_CHECKING:
    from autogen.beta import Agent
    from autogen.beta.hitl import HumanHook


class AgentExecutor(A2ABaseAgentExecutor):
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
                hitl_hook=_make_replay_hook(replay_queue),
            )
        except InputRequiredError as signal:
            await updater.requires_input(
                message=input_required_message(signal.prompt, context_id=task.context_id, task_id=task.id),
                final=True,
            )
            return
        except Exception as exc:
            raise ServerError(error=InternalError(message=str(exc))) from exc

        await updater.add_artifact(
            parts=text_parts(reply.body or ""),
            name=RESULT_ARTIFACT_NAME,
            append=forwarder.started,
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
    __slots__ = ("_updater", "started")

    def __init__(self, updater: TaskUpdater) -> None:
        self._updater = updater
        self.started = False

    async def __call__(self, chunk: ModelMessageChunk) -> None:
        await self._updater.add_artifact(
            parts=text_parts(chunk.content),
            name=RESULT_ARTIFACT_NAME,
            append=self.started,
            last_chunk=False,
        )
        self.started = True


def _make_replay_hook(queue: list[str]) -> "HumanHook":
    """Build a HITL hook backed by a pre-recorded queue of human inputs.

    Each `context.input(...)` call pops the next entry from the queue. When
    the queue is empty, the hook raises `InputRequiredError`, which the
    executor catches to suspend the task and request input from the client.
    """
    iterator = iter(queue)

    async def hook(request: HumanInputRequest) -> HumanMessage:
        try:
            answer = next(iterator)
        except StopIteration:
            raise InputRequiredError(request.content) from None
        return HumanMessage(answer, parent_id=request.id)

    return hook


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
