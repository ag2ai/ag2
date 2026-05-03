# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, Role, Task, TaskState, TaskStatus
from a2a.utils.errors import InternalError
from google.protobuf.timestamp_pb2 import Timestamp

from autogen.beta.events import (
    HumanInputRequest,
    HumanMessage,
    ModelMessageChunk,
    ModelReasoning,
)
from autogen.beta.history import MemoryStorage
from autogen.beta.stream import MemoryStream

from .client_tools import (
    ClientToolStub,
    build_client_tool_stubs,
    decode_client_tool_call,
    parse_tool_result_request,
)
from .mappers import (
    CLIENT_TOOLS_KEY,
    REASONING_KEY,
    RESULT_ARTIFACT_NAME,
    TOOL_CALL_REQUEST_KEY,
    a2a_message_to_inputs,
    build_result_metadata,
    decode_history,
    dict_to_struct,
    finish_reason_for,
    hitl_replay_queue,
    initial_inputs,
    input_required_message,
    message_metadata,
    text_from_message,
    text_parts,
)
from .server_middleware import ExecutorMiddleware, compose

if TYPE_CHECKING:
    from autogen.beta import Agent


class _InputRequiredSignal(BaseException):  # noqa: N818  # control-flow signal, not a user-visible error
    """Raised by the HITL replay hook to suspend a task.

    ``execute()`` catches this and translates it into an A2A
    ``requires_input(...)`` task transition. ``tool_call_request`` is set when
    the suspension was triggered by a client-side tool stub.

    Subclasses ``BaseException`` (not ``Exception``) so the broad
    ``except Exception`` in ``FunctionTool.__call__`` and similar tool
    dispatchers does not swallow it into a ``ToolErrorEvent`` — the signal
    must propagate up to ``_run_bare`` to translate into an A2A
    ``requires_input`` transition rather than a tool-error frame.
    """

    def __init__(self, prompt: str, *, tool_call_request: dict[str, Any] | None = None) -> None:
        self.prompt = prompt
        self.tool_call_request = tool_call_request
        super().__init__(f"Human input required: {prompt!r}")


class AG2AgentExecutor(AgentExecutor):
    """Concrete ``a2a.server.agent_execution.AgentExecutor`` for AG2 beta agents.

    Constructed by :class:`A2AServer` automatically; can also be hand-wired into
    a custom ``DefaultRequestHandler`` when integrating with bespoke transports.

    ``middleware`` runs around every ``execute`` call — see
    :mod:`autogen.beta.a2a.server_middleware`.
    """

    __slots__ = ("_agent", "_run")

    def __init__(
        self,
        agent: "Agent",
        *,
        middleware: Iterable[ExecutorMiddleware] = (),
    ) -> None:
        self._agent = agent
        self._run = compose(middleware, self._run_bare)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await self._run(context, event_queue)

    async def _run_bare(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.message is None:
            raise InternalError("RequestContext.message is required")

        task = context.current_task
        if task is None:
            ts = Timestamp()
            ts.FromDatetime(datetime.now(timezone.utc))
            task = Task(
                id=context.message.task_id or uuid4().hex,
                context_id=context.message.context_id or uuid4().hex,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED, timestamp=ts),
                history=[context.message],
            )
            await event_queue.enqueue_event(task)
        elif not task.history or task.history[-1].message_id != context.message.message_id:
            # Resumed task: ``a2a-sdk`` hands us the saved task without the
            # in-flight follow-up appended yet. ``hitl_replay_queue`` reads
            # only ``task.history`` — without this append the just-arrived
            # answer is invisible and the tool re-suspends, causing the
            # client to see the same prompt twice.
            task.history.append(context.message)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        inputs = initial_inputs(task) or a2a_message_to_inputs(context.message)
        replay_queue = hitl_replay_queue(task)
        first_message = next((m for m in task.history if m.role == Role.ROLE_USER), context.message)
        raw_schemas = message_metadata(first_message).get(CLIENT_TOOLS_KEY)
        client_tool_schemas: list[dict[str, Any]] = (
            [s for s in raw_schemas if isinstance(s, dict)] if isinstance(raw_schemas, list) else []
        )
        stub_tools = build_client_tool_stubs(client_tool_schemas)

        result_forwarder = _ArtifactForwarder(updater, name=RESULT_ARTIFACT_NAME)
        reasoning_forwarder = _ReasoningStatusForwarder(updater, task=task)

        # Stateless multi-turn: the client serializes its accumulated
        # ``BaseEvent`` stream into ``Message.metadata[HISTORY_KEY]`` on
        # every initial turn. We reseed the request-scoped stream from
        # there and never persist anything between calls. Suspended turns
        # carry no history (the in-flight task lives in the TaskStore and
        # ``hitl_replay_queue`` replays it through ``_ReplayHook``).
        seed_events = decode_history(message_metadata(context.message))
        storage = MemoryStorage()
        stream = MemoryStream(storage=storage)
        if seed_events:
            await storage.set_history(stream.id, seed_events)
        stream.where(ModelMessageChunk).subscribe(result_forwarder)
        stream.where(ModelReasoning).subscribe(reasoning_forwarder)

        try:
            reply = await self._agent.ask(
                *inputs,
                tools=stub_tools,
                stream=stream,
                hitl_hook=_ReplayHook(replay_queue),
            )
        except _InputRequiredSignal as signal:
            await self._suspend_for_input(updater, task, signal)
            return
        except Exception as exc:
            await updater.failed(
                message=Message(
                    role=Role.ROLE_AGENT,
                    parts=[Part(text=str(exc) or type(exc).__name__)],
                    message_id=uuid4().hex,
                    context_id=task.context_id,
                    task_id=task.id,
                ),
            )
            return

        # Body parts are only sent here when nothing was streamed — otherwise
        # the chunks already conveyed the full text and re-sending ``reply.body``
        # as an appended part would duplicate the content client-side.
        final_parts = [] if result_forwarder.streaming_started else text_parts(reply.body or "")
        await updater.add_artifact(
            parts=final_parts,
            name=RESULT_ARTIFACT_NAME,
            append=result_forwarder.streaming_started,
            last_chunk=True,
            metadata=build_result_metadata(
                usage=reply.response.usage,
                finish_reason=finish_reason_for(TaskState.TASK_STATE_COMPLETED),
                model=reply.response.model,
            ),
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            return
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.cancel()

    @staticmethod
    async def _suspend_for_input(
        updater: TaskUpdater,
        task: Task,
        signal: "_InputRequiredSignal",
    ) -> None:
        metadata: dict[str, Any] | None = None
        if signal.tool_call_request:
            metadata = {TOOL_CALL_REQUEST_KEY: signal.tool_call_request}
        await updater.requires_input(
            message=input_required_message(
                signal.prompt,
                context_id=task.context_id,
                task_id=task.id,
                metadata=metadata,
            ),
        )


class _ArtifactForwarder:
    """Forward streamed ``ModelMessageChunk`` content as appended
    ``TaskArtifactUpdate`` events under a fixed artifact ``name``.
    """

    __slots__ = ("_name", "_updater", "streaming_started")

    def __init__(self, updater: TaskUpdater, *, name: str) -> None:
        self._updater = updater
        self._name = name
        self.streaming_started = False

    async def __call__(self, chunk: ModelMessageChunk) -> None:
        await self._updater.add_artifact(
            parts=text_parts(chunk.content),
            name=self._name,
            append=self.streaming_started,
            last_chunk=False,
        )
        self.streaming_started = True


class _ReasoningStatusForwarder:
    """Forward streamed ``ModelReasoning`` content as ``TaskStatusUpdateEvent``
    frames with state ``WORKING`` and a ``REASONING_KEY``-tagged message.

    A2A spec reserves artifacts for produced outputs; reasoning is process
    state, so it rides the status-update channel instead. Clients filter
    by ``Message.metadata[REASONING_KEY]`` to distinguish reasoning chunks
    from regular working-state messages.
    """

    __slots__ = ("_updater", "_task")

    def __init__(self, updater: TaskUpdater, *, task: Task) -> None:
        self._updater = updater
        self._task = task

    async def __call__(self, chunk: ModelReasoning) -> None:
        await self._updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=Message(
                role=Role.ROLE_AGENT,
                parts=text_parts(chunk.content),
                message_id=uuid4().hex,
                context_id=self._task.context_id,
                task_id=self._task.id,
                metadata=dict_to_struct({REASONING_KEY: True}),
            ),
        )


class _ReplayHook:
    """Pop pre-recorded human inputs (or client-tool results) from a queue.

    Two answer shapes are recognised:

    - **Plain text** — message has no ``tool_call_result`` metadata. Returned as
      ``HumanMessage(text)``.
    - **Client-tool result** — message metadata carries ``TOOL_CALL_RESULT_KEY``.
      The output (or error) is unpacked and returned as ``HumanMessage`` so the
      stub function sees a plain string return value.
    """

    __slots__ = ("_iter",)

    def __init__(self, queue: list[Message]) -> None:
        self._iter = iter(queue)

    async def __call__(self, request: HumanInputRequest) -> HumanMessage:
        encoded_request = decode_client_tool_call(request.content)

        try:
            answer = next(self._iter)
        except StopIteration:
            raise _InputRequiredSignal(
                prompt=request.content if encoded_request is None else "",
                tool_call_request=encoded_request,
            ) from None

        result_payload = parse_tool_result_request(message_metadata(answer))
        if result_payload is not None:
            content = (
                f"Error: {result_payload['error']}"
                if "error" in result_payload
                else str(result_payload.get("output", ""))
            )
            return HumanMessage(content, parent_id=request.id)

        return HumanMessage(text_from_message(answer), parent_id=request.id)


__all__ = (
    "AG2AgentExecutor",
    "ClientToolStub",
    "build_client_tool_stubs",
)
