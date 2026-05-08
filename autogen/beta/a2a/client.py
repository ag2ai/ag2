# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from a2a.client import Client, ClientCallInterceptor
from a2a.client.errors import A2AClientError
from a2a.types import (
    AgentCard,
    GetExtendedAgentCardRequest,
    GetTaskRequest,
    Message,
    Part,
    SendMessageConfiguration,
    SendMessageRequest,
    StreamResponse,
    SubscribeToTaskRequest,
    Task,
    TaskState,
)
from fast_depends.library.serializer import SerializerProto

from autogen.beta.config.client import LLMClient
from autogen.beta.context import ConversationContext
from autogen.beta.events import (
    BaseEvent,
    Input,
    ModelMessage,
    ModelMessageChunk,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultsEvent,
    Usage,
)
from autogen.beta.response import ResponseProto
from autogen.beta.tools.final.function_tool import FunctionToolSchema
from autogen.beta.tools.schemas import ToolSchema

from .errors import (
    A2AClientToolsNotSupportedError,
    A2AReconnectError,
    A2ATaskFailedError,
    A2ATaskRejectedError,
)
from .extension import (
    EXTENSION_URI,
    EXTRA_PARTS_DEPENDENCY_KEY,
    MIME_TOOL_CALL,
    TENANT_VARIABLE_KEY,
)
from .mappers.messages import (
    build_input_response_message,
    build_tool_result_message,
    build_user_message,
    extract_context_update,
)
from .mappers.parts import is_data_part_with_mime, part_data_to_python
from .mappers.tools import payload_to_call
from .transports._http import fetch_card, make_a2a_client, make_httpx_client

_PROVIDER = "a2a"
_CONTEXT_ID_VAR_TEMPLATE = "a2a:context_id:{url}"

_TERMINAL_STATES = frozenset({
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_REJECTED,
    TaskState.TASK_STATE_INPUT_REQUIRED,
})


@dataclass(slots=True)
class _DriveState:
    """State accumulated across driving one ``ask`` to its terminal task.

    Survives ``input_required`` continuations: each loop appends to
    ``accumulated_text`` and ``pending_calls``, so the final
    ``ModelResponse`` reflects the entire interaction.
    """

    accumulated_text: str = ""
    pending_calls: list[ToolCallEvent] = field(default_factory=list)
    finish_reason: str = "completed"
    failed_task: Task | None = None
    rejected_task: Task | None = None


@dataclass(slots=True)
class _TurnOutcome:
    """Per-turn result handed back from a streaming/polling drain.

    ``input_required`` signals the caller to ask the HITL hook for
    ``input_prompt`` and continue the same task with the user reply.
    """

    input_required: bool = False
    input_prompt: str | None = None


class A2AClient(LLMClient):
    """``LLMClient`` implementation that delegates to a remote A2A agent.

    Lifecycle: one ``A2AClient`` instance per ``Agent.ask()`` call.
    Within that ask, ``self._task_id`` carries the server-issued task id
    across multiple ``__call__`` invocations (for client-side tool
    round-trips). Across asks, ``contextId`` lives in
    ``context.variables`` so the server can stitch the conversation.

    The client always ships the **full** AG2 conversation history on
    every outgoing turn — the server is stateless on AG2 history. See
    ``mappers/history.py`` for the wire shape.
    """

    def __init__(
        self,
        *,
        url: str,
        transports: Sequence[Literal["jsonrpc", "rest", "grpc"]] = ("jsonrpc",),
        streaming: bool = True,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = 60.0,
        max_reconnects: int = 3,
        reconnect_backoff: float = 0.5,
        polling_interval: float = 0.5,
        input_required_timeout: float | None = None,
        httpx_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        interceptors: Sequence[ClientCallInterceptor] = (),
        grpc_channel_factory: Callable[[str], Any] | None = None,
        preset_card: AgentCard | None = None,
        tenant: str | None = None,
        history_length: int | None = None,
    ) -> None:
        transports_tuple = tuple(transports)
        if not transports_tuple:
            raise ValueError("transports must contain at least one of 'jsonrpc', 'rest', 'grpc'")

        self._url = url
        self._transports = transports_tuple
        self._streaming = streaming
        self._headers = dict(headers) if headers else None
        self._timeout = timeout
        self._max_reconnects = max_reconnects
        self._reconnect_backoff = reconnect_backoff
        self._polling_interval = polling_interval
        self._input_required_timeout = input_required_timeout
        self._httpx_client_factory = httpx_client_factory
        self._interceptors = list(interceptors)
        self._grpc_channel_factory = grpc_channel_factory
        self._preset_card = preset_card
        self._tenant = tenant
        self._history_length = history_length

        self._httpx_client: httpx.AsyncClient | None = None
        self._sdk_client: Client | None = None
        self._agent_card: AgentCard | None = preset_card
        self._task_id: str | None = None

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: ConversationContext,
        *,
        tools: Iterable[ToolSchema],
        response_schema: ResponseProto | None,
        serializer: SerializerProto,
    ) -> ModelResponse:
        if response_schema is not None:
            raise NotImplementedError("response_schema is not yet supported with A2AConfig")

        await self._ensure_connected(context)
        assert self._agent_card is not None
        assert self._sdk_client is not None

        function_schemas = self._validate_and_extract_tools(tools)
        outgoing = self._build_outgoing(messages, function_schemas, context)

        state = _DriveState()
        while True:
            outcome = await self._drive_task(outgoing, context, state)
            if not outcome.input_required:
                break
            # ``INPUT_REQUIRED`` is overloaded: server emits it both for
            # client-side tool round-trips (carrying ``tool-call+json``
            # artifacts) and for genuine human-in-the-loop prompts. When
            # tool calls are pending we surface them to the outer agent
            # for local execution; only when there's nothing for the
            # agent to do do we fall back to the HITL hook.
            if state.pending_calls:
                break

            user_text = await self._await_user_input(context, outcome.input_prompt)
            outgoing = build_input_response_message(
                user_text,
                task_id=self._task_id or "",
                context_id=self._read_context_id(context),
            )

        if state.failed_task is not None:
            raise A2ATaskFailedError(state.failed_task)
        if state.rejected_task is not None:
            raise A2ATaskRejectedError(state.rejected_task)

        message = ModelMessage(state.accumulated_text) if state.accumulated_text else None

        return ModelResponse(
            message=message,
            tool_calls=ToolCallsEvent(state.pending_calls),
            usage=Usage(),
            model=self._agent_card.name if self._agent_card else None,
            provider=_PROVIDER,
            finish_reason=state.finish_reason,
        )

    async def _ensure_connected(self, context: ConversationContext) -> None:
        if self._sdk_client is not None:
            return
        self._httpx_client = make_httpx_client(
            headers=self._headers,
            timeout=self._timeout,
            factory=self._httpx_client_factory,
        )
        if self._preset_card is None:
            self._agent_card = await fetch_card(self._httpx_client, url=self._url)
        self._sdk_client = make_a2a_client(
            card=self._agent_card,
            httpx_client=self._httpx_client,
            streaming=self._streaming,
            transports=self._transports,
            interceptors=self._interceptors,
            grpc_channel_factory=self._grpc_channel_factory,
        )
        if self._agent_card.capabilities.extended_agent_card:
            kwargs = self._maybe_tenant(context)
            self._agent_card = await self._sdk_client.get_extended_agent_card(
                GetExtendedAgentCardRequest(**kwargs),
            )

    def _validate_and_extract_tools(
        self,
        tools: Iterable[ToolSchema],
    ) -> list[FunctionToolSchema]:
        function_schemas = [t for t in tools if isinstance(t, FunctionToolSchema)]
        if not function_schemas:
            return []
        if not self._card_advertises_extension():
            raise A2AClientToolsNotSupportedError(
                f"Server at {self._url!r} does not advertise extension "
                f"{EXTENSION_URI!r}; remove tools= or use a server that supports it."
            )
        return function_schemas

    def _card_advertises_extension(self) -> bool:
        if self._agent_card is None or self._agent_card.capabilities is None:
            return False
        return any(ext.uri == EXTENSION_URI for ext in self._agent_card.capabilities.extensions)

    def _streaming_enabled(self) -> bool:
        if self._agent_card is None:
            return False
        return self._streaming and self._agent_card.capabilities.streaming

    def _build_outgoing(
        self,
        messages: Sequence[BaseEvent],
        function_schemas: Sequence[FunctionToolSchema],
        context: ConversationContext,
    ) -> Message:
        context_id = self._read_context_id(context)
        last = messages[-1] if messages else None

        if isinstance(last, ToolResultsEvent) and self._task_id is not None:
            return build_tool_result_message(
                last.results,
                history_events=messages,
                tool_schemas=function_schemas,
                task_id=self._task_id,
                context_id=context_id,
                context_update=dict(context.variables) or None,
            )

        inputs = self._collect_user_inputs(messages)
        extra_parts = _read_extra_parts(context)
        return build_user_message(
            inputs,
            history_events=messages,
            tool_schemas=function_schemas,
            task_id=self._task_id,
            context_id=context_id,
            advertise_extension=bool(function_schemas) or self._task_id is not None,
            context_update=dict(context.variables) or None,
            extra_parts=extra_parts,
        )

    @staticmethod
    def _collect_user_inputs(messages: Sequence[BaseEvent]) -> list[Input]:
        for ev in reversed(messages):
            if isinstance(ev, ModelRequest):
                return list(ev.parts)
        return [TextInput("")]

    async def _drive_task(
        self,
        message: Message,
        context: ConversationContext,
        state: _DriveState,
    ) -> _TurnOutcome:
        if self._streaming_enabled():
            return await self._consume_streaming(message, context, state)
        return await self._consume_polling(message, context, state)

    async def _consume_streaming(
        self,
        message: Message,
        context: ConversationContext,
        state: _DriveState,
    ) -> _TurnOutcome:
        assert self._sdk_client is not None

        request = self._build_send_request(message, context)
        stream: AsyncIterator[Any] = self._sdk_client.send_message(request)

        attempt = 0
        while True:
            try:
                return await self._drain_stream(stream, context, state)
            except A2AClientError as exc:
                if self._task_id is None or attempt >= self._max_reconnects:
                    raise A2AReconnectError(attempt) from exc
                attempt += 1
                backoff = self._reconnect_backoff * (2 ** (attempt - 1))
                await asyncio.sleep(backoff)
                resubscribe = SubscribeToTaskRequest(**self._maybe_tenant(context, id=self._task_id))
                stream = self._sdk_client.subscribe(resubscribe)

    async def _consume_polling(
        self,
        message: Message,
        context: ConversationContext,
        state: _DriveState,
    ) -> _TurnOutcome:
        assert self._sdk_client is not None

        request = self._build_send_request(message, context)
        outcome = await self._drain_stream(self._sdk_client.send_message(request), context, state)
        if state.finish_reason in ("failed", "rejected") or outcome.input_required:
            return outcome

        if self._task_id is None:
            return outcome

        while True:
            get_kwargs = self._maybe_tenant(context, id=self._task_id)
            if self._history_length is not None:
                get_kwargs["history_length"] = self._history_length
            task = await self._sdk_client.get_task(GetTaskRequest(**get_kwargs))
            self._absorb_task_artifacts(task, context, state)
            if task.status.state in _TERMINAL_STATES:
                return self._terminal_outcome(task, state)
            await asyncio.sleep(self._polling_interval)

    async def _drain_stream(
        self,
        stream: AsyncIterator[Any],
        context: ConversationContext,
        state: _DriveState,
    ) -> _TurnOutcome:
        outcome = _TurnOutcome()
        async for event in stream:
            response = _ensure_stream_response(event)
            payload = response.WhichOneof("payload")

            if payload == "task":
                self._task_id = response.task.id
                self._save_context_id(context, response.task.context_id)
                continue

            if payload == "status_update":
                self._save_context_id(context, response.status_update.context_id)
                if response.status_update.task_id:
                    self._task_id = response.status_update.task_id
                stop = await self._handle_status_update(
                    response.status_update,
                    context,
                    state,
                    outcome,
                )
                if stop:
                    return outcome
                continue

            if payload == "artifact_update":
                self._save_context_id(context, response.artifact_update.context_id)
                text_chunk, calls = await self._handle_artifact_parts(
                    response.artifact_update.artifact.parts,
                    context,
                )
                state.accumulated_text += text_chunk
                state.pending_calls.extend(calls)
                continue

            if payload == "message":
                msg = response.message
                self._save_context_id(context, msg.context_id)
                if msg.task_id:
                    self._task_id = msg.task_id
                self._merge_context_update(context, extract_context_update(msg))
                text_chunk, calls = await self._handle_artifact_parts(msg.parts, context)
                state.accumulated_text += text_chunk
                state.pending_calls.extend(calls)
                continue

        return outcome

    async def _handle_status_update(
        self,
        status_update: Any,
        context: ConversationContext,
        state: _DriveState,
        outcome: _TurnOutcome,
    ) -> bool:
        sd_state = status_update.status.state
        if sd_state == TaskState.TASK_STATE_FAILED:
            state.failed_task = self._fake_task_for_status(status_update)
            state.finish_reason = "failed"
            return True
        if sd_state == TaskState.TASK_STATE_REJECTED:
            state.rejected_task = self._fake_task_for_status(status_update)
            state.finish_reason = "rejected"
            return True
        if sd_state == TaskState.TASK_STATE_INPUT_REQUIRED:
            state.finish_reason = "input_required"
            outcome.input_required = True
            outcome.input_prompt = _extract_status_prompt(status_update.status)
            return True
        # Servers usually attach the final agent text on the
        # ``status.message`` of the COMPLETED transition rather than
        # emitting it as a separate ``message`` payload, so we absorb
        # both here.
        if sd_state == TaskState.TASK_STATE_COMPLETED and status_update.status.HasField("message"):
            msg = status_update.status.message
            self._merge_context_update(context, extract_context_update(msg))
            text_chunk, calls = await self._handle_artifact_parts(msg.parts, context)
            state.accumulated_text += text_chunk
            state.pending_calls.extend(calls)
        return False

    def _absorb_task_artifacts(self, task: Task, context: ConversationContext, state: _DriveState) -> None:
        # Polling mode: we don't get incremental artifact-update deltas;
        # rebuild text/tool-calls from the latest task snapshot. The final
        # agent text usually lives in ``task.status.message`` (emitted by
        # ``updater.complete(message=...)``); streamed deltas land in
        # ``task.artifacts``. We absorb both so neither path drops content.
        #
        # ``task.artifacts`` is cumulative — it carries tool-call+json
        # artifacts from earlier turns even after the task has moved past
        # them. Only surface tool calls when the task is asking for input
        # right now (``INPUT_REQUIRED``); otherwise the artifacts are
        # historical and would cause the outer agent loop to re-execute
        # tools that were already handled.
        terminal_calls_visible = task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
        new_text = ""
        new_calls: list[ToolCallEvent] = []
        for artifact in task.artifacts:
            for part in artifact.parts:
                if part.text:
                    new_text += part.text
                    continue
                if terminal_calls_visible and is_data_part_with_mime(part, MIME_TOOL_CALL):
                    new_calls.append(payload_to_call(part_data_to_python(part)))
        if task.status.HasField("message"):
            for part in task.status.message.parts:
                if part.text:
                    new_text += part.text
                if terminal_calls_visible and is_data_part_with_mime(part, MIME_TOOL_CALL):
                    new_calls.append(payload_to_call(part_data_to_python(part)))
            self._merge_context_update(context, extract_context_update(task.status.message))
        state.accumulated_text = new_text or state.accumulated_text
        if new_calls:
            state.pending_calls = new_calls

    def _terminal_outcome(self, task: Task, state: _DriveState) -> _TurnOutcome:
        sd_state = task.status.state
        if sd_state == TaskState.TASK_STATE_FAILED:
            state.failed_task = task
            state.finish_reason = "failed"
        elif sd_state == TaskState.TASK_STATE_REJECTED:
            state.rejected_task = task
            state.finish_reason = "rejected"
        elif sd_state == TaskState.TASK_STATE_INPUT_REQUIRED:
            state.finish_reason = "input_required"
            return _TurnOutcome(
                input_required=True,
                input_prompt=_extract_status_prompt(task.status),
            )
        return _TurnOutcome()

    async def _handle_artifact_parts(
        self,
        parts: Iterable[Any],
        context: ConversationContext,
    ) -> tuple[str, list[ToolCallEvent]]:
        text_acc = ""
        calls: list[ToolCallEvent] = []
        for part in parts:
            if part.text:
                text_acc += part.text
                await context.send(ModelMessageChunk(part.text))
                continue
            if is_data_part_with_mime(part, MIME_TOOL_CALL):
                calls.append(payload_to_call(part_data_to_python(part)))
                continue
        return text_acc, calls

    async def _await_user_input(self, context: ConversationContext, prompt: str | None) -> str:
        # Defers to the agent's HITL hook. If none is wired up the beta
        # default raises ``HumanInputNotProvidedError`` — that's the
        # signal to the caller that this server requires HITL but the
        # client side isn't set up for it.
        return await context.input(prompt or "Please provide input:", timeout=self._input_required_timeout)

    def _build_send_request(self, message: Message, context: ConversationContext) -> SendMessageRequest:
        assert self._agent_card is not None
        config_kwargs: dict[str, Any] = {
            "accepted_output_modes": list(self._agent_card.default_output_modes) or ["text/plain", "application/json"],
        }
        if self._history_length is not None:
            config_kwargs["history_length"] = self._history_length
        request_kwargs = self._maybe_tenant(
            context,
            message=message,
            configuration=SendMessageConfiguration(**config_kwargs),
        )
        return SendMessageRequest(**request_kwargs)

    def _read_context_id(self, context: ConversationContext) -> str | None:
        return context.variables.get(_CONTEXT_ID_VAR_TEMPLATE.format(url=self._url))

    def _save_context_id(self, context: ConversationContext, context_id: str) -> None:
        if not context_id:
            return
        context.variables[_CONTEXT_ID_VAR_TEMPLATE.format(url=self._url)] = context_id

    def _resolve_tenant(self, context: ConversationContext) -> str | None:
        # Per-call override wins: a single Agent can fan out to multiple
        # tenants by setting ``a2a:tenant`` on its ``context.variables``.
        # Falls back to the tenant baked into the config at construction.
        override = context.variables.get(TENANT_VARIABLE_KEY)
        if isinstance(override, str) and override:
            return override
        return self._tenant

    def _maybe_tenant(self, context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        tenant = self._resolve_tenant(context)
        if tenant:
            kwargs["tenant"] = tenant
        return kwargs

    @staticmethod
    def _merge_context_update(context: ConversationContext, payload: Mapping[str, Any]) -> None:
        if not payload:
            return
        context.variables.update(payload)

    @staticmethod
    def _fake_task_for_status(status_update: Any) -> Task:
        # Build a minimal Task surrogate carrying the failure status so the
        # error type can be raised consistently with the spec'd terminal flow.
        return Task(id=status_update.task_id, status=status_update.status, context_id=status_update.context_id)


def _ensure_stream_response(event: Any) -> StreamResponse:
    if isinstance(event, StreamResponse):
        return event
    # SDK can yield bare protobuf payload objects for individual oneof
    # fields — wrap them so the consumer always sees a uniform type.
    if isinstance(event, Task):
        return StreamResponse(task=event)
    if isinstance(event, Message):
        return StreamResponse(message=event)
    raise TypeError(f"Unexpected stream event type: {type(event).__name__}")


def _extract_status_prompt(status: Any) -> str | None:
    if not status.HasField("message"):
        return None
    chunks = [part.text for part in status.message.parts if part.text]
    if not chunks:
        return None
    return "".join(chunks)


def _read_extra_parts(context: ConversationContext) -> list[Part]:
    """Read user-provided extra ``Part``s from context dependencies.

    Accepts either a list of ``Part`` instances directly, or anything
    iterable that yields ``Part`` instances. Anything else is silently
    ignored — extra parts are advisory.
    """
    raw = context.dependencies.get(EXTRA_PARTS_DEPENDENCY_KEY)
    if not raw:
        return []
    return [p for p in raw if isinstance(p, Part)]
