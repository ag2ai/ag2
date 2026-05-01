# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
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

from autogen.beta.annotations import Context as BetaContext
from autogen.beta.events import (
    HumanInputRequest,
    HumanMessage,
    ModelMessageChunk,
    ModelReasoning,
)
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.final import FunctionTool, tool

from .client_tools import parse_tool_result_request, validate_client_tool_parameters
from .mappers import (
    CLIENT_TOOLS_KEY,
    REASONING_ARTIFACT_NAME,
    RESULT_ARTIFACT_NAME,
    TOOL_CALL_REQUEST_KEY,
    a2a_message_to_inputs,
    build_result_metadata,
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

# JSON marker prefix written into ``HumanInputRequest.content`` by client-tool
# stubs. The HITL replay hook detects this and routes the request via the
# tool_call_request metadata path instead of treating the prompt as plain HITL
# text. Stuffed into ``content`` because beta's ``HumanInputRequest`` does not
# carry a metadata field; an additive change to the event would be cleaner but
# is out of scope for this PR.
_CLIENT_TOOL_CONTENT_PREFIX = "__ag2_client_tool_call__:"


class _InputRequiredSignal(Exception):  # noqa: N818  # control-flow signal, not a user-visible error
    """Raised by the HITL replay hook to suspend a task.

    ``execute()`` catches this and translates it into an A2A
    ``requires_input(...)`` task transition. ``tool_call_request`` is set when
    the suspension was triggered by a client-side tool stub.
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

        task = context.current_task or _build_initial_task(context.message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        inputs = initial_inputs(task) or a2a_message_to_inputs(context.message)
        replay_queue = hitl_replay_queue(task)
        first_message = _user_messages_first(task) or context.message
        client_tool_schemas = _client_tool_schemas(first_message)
        stub_tools = build_client_tool_stubs(client_tool_schemas)

        result_forwarder = _ArtifactForwarder(updater, name=RESULT_ARTIFACT_NAME)
        reasoning_forwarder = _ArtifactForwarder(updater, name=REASONING_ARTIFACT_NAME)

        stream = MemoryStream()
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
            await updater.failed(message=_agent_text_message(task, str(exc) or type(exc).__name__))
            return

        await updater.add_artifact(
            parts=text_parts(reply.body or ""),
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
    """Forward streamed ``ModelMessageChunk`` / ``ModelReasoning`` content as
    appended ``TaskArtifactUpdate`` events under a fixed artifact ``name``.
    """

    __slots__ = ("_name", "_updater", "streaming_started")

    def __init__(self, updater: TaskUpdater, *, name: str) -> None:
        self._updater = updater
        self._name = name
        self.streaming_started = False

    async def __call__(self, chunk: ModelMessageChunk | ModelReasoning) -> None:
        await self._updater.add_artifact(
            parts=text_parts(chunk.content),
            name=self._name,
            append=self.streaming_started,
            last_chunk=False,
        )
        self.streaming_started = True


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
        encoded_request = _decode_client_tool_request(request.content)

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


def _decode_client_tool_request(prompt: str) -> dict[str, Any] | None:
    if not prompt.startswith(_CLIENT_TOOL_CONTENT_PREFIX):
        return None
    raw = prompt[len(_CLIENT_TOOL_CONTENT_PREFIX) :]
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    if "id" not in decoded or "name" not in decoded:
        return None
    return decoded


def build_client_tool_stubs(schemas: list[dict[str, Any]]) -> list[FunctionTool]:
    """Materialise stub ``FunctionTool``s from client-supplied JSON schemas."""
    return [_build_client_tool_stub(s) for s in schemas]


class _ClientToolStub:
    """Callable that encodes its kwargs as a tool-call request and sends it via ``context.input``.

    Top-level class so ``_build_client_tool_stub`` does not define an inner
    function at runtime — only the ``@tool`` decoration is per-request work.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    async def __call__(self, context: BetaContext, **kwargs: Any) -> str:
        request = json.dumps(
            {"id": uuid4().hex, "name": self._name, "arguments": json.dumps(kwargs)},
            sort_keys=True,
        )
        return await context.input(_CLIENT_TOOL_CONTENT_PREFIX + request)


def _build_client_tool_stub(schema: dict[str, Any]) -> FunctionTool:
    name = schema.get("name") or "client_tool"
    description = schema.get("description") or f"Client-side tool {name}."
    parameters = schema.get("parameters") or {"type": "object", "properties": {}}

    validate_client_tool_parameters(parameters)
    return tool(name=name, description=description, schema=dict(parameters))(_ClientToolStub(name))


def _build_initial_task(message: Message) -> Task:
    ts = Timestamp()
    ts.FromDatetime(datetime.now(timezone.utc))
    return Task(
        id=message.task_id or uuid4().hex,
        context_id=message.context_id or uuid4().hex,
        status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED, timestamp=ts),
        history=[message],
    )


def _user_messages_first(task: Task) -> Message | None:
    for m in task.history:
        if m.role == Role.ROLE_USER:
            return m
    return None


def _client_tool_schemas(message: Message | None) -> list[dict[str, Any]]:
    if message is None:
        return []
    md = message_metadata(message)
    raw = md.get(CLIENT_TOOLS_KEY) if md else None
    if not isinstance(raw, list):
        return []
    return [s for s in raw if isinstance(s, dict)]


def _agent_text_message(task: Task, text: str) -> Message:
    return Message(
        role=Role.ROLE_AGENT,
        parts=[Part(text=text)],
        message_id=uuid4().hex,
        context_id=task.context_id,
        task_id=task.id,
    )


__all__ = (
    "AG2AgentExecutor",
    "build_client_tool_stubs",
)
