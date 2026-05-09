# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import random
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
    TaskStatus,
    TaskStatusUpdateEvent,
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
    A2ATaskAuthRequiredError,
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
from .transports import TransportName
from .transports._http import fetch_card, make_a2a_client, make_httpx_client, select_transport

if TYPE_CHECKING:
    import grpc.aio

_PROVIDER = "a2a"
_CONTEXT_ID_VAR_TEMPLATE = "a2a:context_id:{card_url}"

_TERMINAL_STATES = frozenset({
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_REJECTED,
    TaskState.TASK_STATE_INPUT_REQUIRED,
    TaskState.TASK_STATE_AUTH_REQUIRED,
})


@dataclass(slots=True)
class A2ADriveState:
    """State accumulated across driving one ``ask`` to its terminal task.

    Survives ``input_required`` continuations: each loop appends to
    ``accumulated_text`` and ``pending_calls``, so the final
    ``ModelResponse`` reflects the entire interaction.

    ``terminal_task`` holds the failed/rejected/auth-required Task; the
    specific kind is disambiguated by ``finish_reason``.
    """

    accumulated_text: str = ""
    pending_calls: list[ToolCallEvent] = field(default_factory=list)
    finish_reason: str = "completed"
    terminal_task: Task | None = None
    # Dedup keys: artifacts and messages can be replayed by the server on
    # ``SubscribeToTask`` reconnect (per spec §3.5.2 "MAY optionally resend
    # a final Task snapshot"), and polling re-reads the cumulative
    # ``task.artifacts`` snapshot on every poll. Without dedup the client
    # appends the same content twice.
    seen_artifact_ids: set[str] = field(default_factory=set)
    seen_message_ids: set[str] = field(default_factory=set)


@dataclass(slots=True)
class A2ATurnOutcome:
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
        card_url: str,
        prefer: TransportName | None = None,
        streaming: bool = True,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = 60.0,
        max_reconnects: int = 3,
        reconnect_backoff: float = 0.5,
        polling_interval: float = 0.5,
        polling_jitter: float = 0.0,
        input_required_timeout: float | None = None,
        httpx_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        interceptors: Sequence[ClientCallInterceptor] = (),
        grpc_channel_factory: Callable[[str], "grpc.aio.Channel"] | None = None,
        preset_card: AgentCard | None = None,
        tenant: str | None = None,
        history_length: int | None = None,
    ) -> None:
        self._card_url = card_url
        self._prefer = prefer
        self._streaming = streaming
        self._headers = dict(headers) if headers else None
        self._timeout = timeout
        self._max_reconnects = max_reconnects
        self._reconnect_backoff = reconnect_backoff
        self._polling_interval = polling_interval
        self._polling_jitter = max(0.0, polling_jitter)
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

        try:
            await self._ensure_connected(context)
            assert self._agent_card is not None
            assert self._sdk_client is not None

            function_schemas = self._validate_and_extract_tools(tools)
            outgoing = self._build_outgoing(messages, function_schemas, context)

            state = A2ADriveState()
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

            if state.terminal_task is not None:
                if state.finish_reason == "failed":
                    raise A2ATaskFailedError(state.terminal_task)
                if state.finish_reason == "rejected":
                    raise A2ATaskRejectedError(state.terminal_task)
                if state.finish_reason == "auth_required":
                    raise A2ATaskAuthRequiredError(state.terminal_task)

            message = ModelMessage(state.accumulated_text) if state.accumulated_text else None

            return ModelResponse(
                message=message,
                tool_calls=ToolCallsEvent(state.pending_calls),
                usage=Usage(),
                model=self._agent_card.name if self._agent_card else None,
                provider=_PROVIDER,
                finish_reason=state.finish_reason,
            )
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        """Release the underlying httpx and SDK clients.

        Idempotent — safe to call from ``finally`` blocks even when
        ``_ensure_connected`` never ran. The client lifecycle is one
        ``__call__`` per instance (see class docstring), so closing in
        ``finally`` matches the contract.
        """
        if self._httpx_client is not None:
            await self._httpx_client.aclose()
            self._httpx_client = None
        self._sdk_client = None

    async def _ensure_connected(self, context: ConversationContext) -> None:
        if self._sdk_client is not None:
            return
        self._httpx_client = make_httpx_client(
            headers=self._headers,
            timeout=self._timeout,
            factory=self._httpx_client_factory,
        )
        if self._preset_card is None:
            self._agent_card = await fetch_card(self._httpx_client, url=self._card_url)
        transport = select_transport(self._agent_card, url=self._card_url, prefer=self._prefer)
        self._sdk_client = make_a2a_client(
            card=self._agent_card,
            httpx_client=self._httpx_client,
            streaming=self._streaming,
            transport=transport,
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
                f"Server at {self._card_url!r} does not advertise extension "
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
        state: A2ADriveState,
    ) -> A2ATurnOutcome:
        if self._streaming_enabled():
            return await self._consume_streaming(message, context, state)
        return await self._consume_polling(message, context, state)

    async def _consume_streaming(
        self,
        message: Message,
        context: ConversationContext,
        state: A2ADriveState,
    ) -> A2ATurnOutcome:
        assert self._sdk_client is not None

        request = self._build_send_request(message, context)
        stream = self._sdk_client.send_message(request)

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
        state: A2ADriveState,
    ) -> A2ATurnOutcome:
        assert self._sdk_client is not None

        request = self._build_send_request(message, context)
        outcome = await self._drain_stream(self._sdk_client.send_message(request), context, state)
        if state.finish_reason in ("failed", "rejected", "auth_required") or outcome.input_required:
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
            await asyncio.sleep(self._next_poll_delay())

    def _next_poll_delay(self) -> float:
        # Optional jitter de-correlates polling clients hitting the same
        # server in a thundering herd. ``polling_jitter=0`` (default)
        # preserves the deterministic interval used by tests.
        if self._polling_jitter <= 0:
            return self._polling_interval
        spread = random.uniform(-self._polling_jitter, self._polling_jitter)
        return max(0.0, self._polling_interval + spread)

    async def _drain_stream(
        self,
        stream: AsyncIterator[StreamResponse],
        context: ConversationContext,
        state: A2ADriveState,
    ) -> A2ATurnOutcome:
        outcome = A2ATurnOutcome()
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
                artifact = response.artifact_update.artifact
                # ``append=True`` chunks reuse the same ``artifact_id``;
                # only dedup when the artifact is fully delivered. Until
                # ``last_chunk`` (or a non-append delivery) we keep
                # appending new chunks to ``accumulated_text``.
                if artifact.artifact_id in state.seen_artifact_ids:
                    continue
                text_chunk, calls = await self._handle_artifact_parts(
                    artifact.parts,
                    context,
                )
                state.accumulated_text += text_chunk
                state.pending_calls.extend(calls)
                if response.artifact_update.last_chunk or not response.artifact_update.append:
                    state.seen_artifact_ids.add(artifact.artifact_id)
                continue

            if payload == "message":
                msg = response.message
                self._save_context_id(context, msg.context_id)
                if msg.task_id:
                    self._task_id = msg.task_id
                if msg.message_id and msg.message_id in state.seen_message_ids:
                    continue
                self._merge_context_update(context, extract_context_update(msg))
                text_chunk, calls = await self._handle_artifact_parts(msg.parts, context)
                state.accumulated_text += text_chunk
                state.pending_calls.extend(calls)
                if msg.message_id:
                    state.seen_message_ids.add(msg.message_id)
                continue

        return outcome

    async def _handle_status_update(
        self,
        status_update: TaskStatusUpdateEvent,
        context: ConversationContext,
        state: A2ADriveState,
        outcome: A2ATurnOutcome,
    ) -> bool:
        sd_state = status_update.status.state
        if sd_state == TaskState.TASK_STATE_FAILED:
            state.terminal_task = self._synthesize_task_from_status(status_update)
            state.finish_reason = "failed"
            return True
        if sd_state == TaskState.TASK_STATE_REJECTED:
            state.terminal_task = self._synthesize_task_from_status(status_update)
            state.finish_reason = "rejected"
            return True
        if sd_state == TaskState.TASK_STATE_AUTH_REQUIRED:
            state.terminal_task = self._synthesize_task_from_status(status_update)
            state.finish_reason = "auth_required"
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
            if not msg.message_id or msg.message_id not in state.seen_message_ids:
                self._merge_context_update(context, extract_context_update(msg))
                text_chunk, calls = await self._handle_artifact_parts(msg.parts, context)
                state.accumulated_text += text_chunk
                state.pending_calls.extend(calls)
                if msg.message_id:
                    state.seen_message_ids.add(msg.message_id)
        return False

    def _absorb_task_artifacts(self, task: Task, context: ConversationContext, state: A2ADriveState) -> None:
        # Polling mode: every poll re-reads the cumulative ``task.artifacts``
        # snapshot. Within one ``__call__`` the per-state dedup guards
        # against double-appending the same artifact. Across calls (the
        # AG2 outer loop creates a fresh ``A2AClient`` per ask), the state
        # is empty — so we additionally gate ``tool-call+json`` artifacts
        # on ``INPUT_REQUIRED``. Historical tool-calls from earlier turns
        # remain in ``task.artifacts`` after the task moves past them and
        # would otherwise be re-emitted to the outer agent for execution.
        # Streaming does not need this gate: ``artifact_update`` events
        # are incremental, never replayed as part of a cumulative snapshot.
        terminal_calls_visible = task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
        for artifact in task.artifacts:
            if artifact.artifact_id in state.seen_artifact_ids:
                continue
            state.seen_artifact_ids.add(artifact.artifact_id)
            for part in artifact.parts:
                if part.text:
                    state.accumulated_text += part.text
                    continue
                if terminal_calls_visible and is_data_part_with_mime(part, MIME_TOOL_CALL):
                    state.pending_calls.append(payload_to_call(part_data_to_python(part)))
        # ``status.message`` on ``INPUT_REQUIRED`` is the HITL prompt — it
        # routes through ``_terminal_outcome.input_prompt`` to the hook,
        # not to the final assistant text. Skip it here to keep parity
        # with the streaming path (``_handle_status_update`` only absorbs
        # ``status.message`` on COMPLETED).
        if task.status.HasField("message") and task.status.state != TaskState.TASK_STATE_INPUT_REQUIRED:
            msg = task.status.message
            if not msg.message_id or msg.message_id not in state.seen_message_ids:
                for part in msg.parts:
                    if part.text:
                        state.accumulated_text += part.text
                    if terminal_calls_visible and is_data_part_with_mime(part, MIME_TOOL_CALL):
                        state.pending_calls.append(payload_to_call(part_data_to_python(part)))
                self._merge_context_update(context, extract_context_update(msg))
                if msg.message_id:
                    state.seen_message_ids.add(msg.message_id)

    def _terminal_outcome(self, task: Task, state: A2ADriveState) -> A2ATurnOutcome:
        sd_state = task.status.state
        if sd_state == TaskState.TASK_STATE_FAILED:
            state.terminal_task = task
            state.finish_reason = "failed"
        elif sd_state == TaskState.TASK_STATE_REJECTED:
            state.terminal_task = task
            state.finish_reason = "rejected"
        elif sd_state == TaskState.TASK_STATE_AUTH_REQUIRED:
            state.terminal_task = task
            state.finish_reason = "auth_required"
        elif sd_state == TaskState.TASK_STATE_INPUT_REQUIRED:
            state.finish_reason = "input_required"
            return A2ATurnOutcome(
                input_required=True,
                input_prompt=_extract_status_prompt(task.status),
            )
        return A2ATurnOutcome()

    async def _handle_artifact_parts(
        self,
        parts: Iterable[Part],
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
        return context.variables.get(_CONTEXT_ID_VAR_TEMPLATE.format(card_url=self._card_url))

    def _save_context_id(self, context: ConversationContext, context_id: str) -> None:
        if not context_id:
            return
        context.variables[_CONTEXT_ID_VAR_TEMPLATE.format(card_url=self._card_url)] = context_id

    def _resolve_tenant(self, context: ConversationContext) -> str | None:
        # Per-call override wins: a single Agent can fan out to multiple
        # tenants by setting ``a2a:tenant`` on its ``context.variables``.
        # Falls back to the tenant baked into the config at construction.
        override = context.variables.get(TENANT_VARIABLE_KEY)
        if isinstance(override, str) and override:
            return override
        return self._tenant

    def _maybe_tenant(self, context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        # Mirrors the ``_session.with_tenant`` rule but pulls the override
        # from ``context.variables`` (per-call) instead of an explicit arg.
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
    def _synthesize_task_from_status(status_update: TaskStatusUpdateEvent) -> Task:
        # Streaming-mode terminal events carry only ``TaskStatusUpdateEvent``
        # rather than a full ``Task`` snapshot. Synthesise a minimal Task
        # surrogate from the status so error types (``A2ATaskFailedError``
        # etc.) can be raised consistently with the polling path that gets
        # a real Task from ``get_task``.
        return Task(id=status_update.task_id, status=status_update.status, context_id=status_update.context_id)


def _ensure_stream_response(event: StreamResponse | Task | Message) -> StreamResponse:
    if isinstance(event, StreamResponse):
        return event
    # SDK can yield bare protobuf payload objects for individual oneof
    # fields — wrap them so the consumer always sees a uniform type.
    if isinstance(event, Task):
        return StreamResponse(task=event)
    if isinstance(event, Message):
        return StreamResponse(message=event)
    raise TypeError(f"Unexpected stream event type: {type(event).__name__}")


def _extract_status_prompt(status: TaskStatus) -> str | None:
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
