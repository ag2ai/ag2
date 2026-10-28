# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

import httpx
from a2a.client import A2ACardResolver, Client, ClientCallContext, ClientCallInterceptor
from a2a.client.errors import A2AClientError
from a2a.types import (
    AgentCard,
    Artifact,
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
from a2a.utils.errors import UnsupportedOperationError
from fast_depends.library.serializer import SerializerProto

from ag2.config.client import LLMClient
from ag2.context import ConversationContext, strip_reserved_variables
from ag2.events import (
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
from ag2.response import ResponseProto
from ag2.tools.final.function_tool import FunctionToolSchema
from ag2.tools.schemas import ToolSchema

from .errors import (
    A2ACardSignatureError,
    A2AClientToolsNotSupportedError,
    A2AReconnectError,
    A2ATaskAuthRequiredError,
    A2ATaskCancelledError,
    A2ATaskFailedError,
    A2ATaskRejectedError,
)
from .events import (
    A2AMessage,
    A2ATaskArtifactUpdate,
    A2ATaskSnapshot,
    A2ATaskStatusUpdate,
    A2ATextArtifact,
    A2AToolCallArtifact,
)
from .extension import (
    EXTENSION_URI,
    EXTRA_PARTS_DEPENDENCY_KEY,
    MIME_TOOL_CALL,
    TENANT_VARIABLE_KEY,
    extension_call_context,
    validate_extension_activation,
)
from .mappers import (
    build_input_response_message,
    build_tool_result_message,
    build_user_message,
    extract_context_update,
    is_data_part_with_mime,
    parse_stream_response,
    parse_task_artifact,
    part_data_to_python,
    payload_to_call,
)
from .transports import TransportName
from .transports._http import make_a2a_client, make_httpx_client, select_interface, validate_protocol_version

# ``a2a.utils.signing`` imports PyJWT at module scope and raises if it is absent,
# so importing it unguarded would make ``ag2.a2a`` unimportable on a plain
# ``ag2[a2a]`` install — the SDK ships PyJWT behind its own ``signing`` extra.
# Without that extra no verifier can be constructed either, so the stand-in below
# is never matched in practice; it only keeps the branch in ``_verify_card`` typed.
try:
    from a2a.utils.signing import SignatureVerificationError
except ImportError:  # pragma: no cover — needs an env without a2a-sdk[signing]

    class SignatureVerificationError(Exception):  # type: ignore[no-redef]
        """Stand-in for the SDK error when ``a2a-sdk[signing]`` is not installed."""


if TYPE_CHECKING:
    import grpc.aio

CardVerifier: TypeAlias = Callable[[AgentCard], None]
"""JWS verifier for consumed AgentCards (from :func:`a2a.utils.signing.create_signature_verifier`)."""

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

# Maps a task-failure terminal state to the ``finish_reason`` we surface
# in the resulting ``ModelResponse``. Driven from one table so the
# streaming and polling paths can never disagree on the mapping.
_FAILURE_REASONS: dict[TaskState, str] = {
    TaskState.TASK_STATE_FAILED: "failed",
    TaskState.TASK_STATE_CANCELED: "cancelled",
    TaskState.TASK_STATE_REJECTED: "rejected",
    TaskState.TASK_STATE_AUTH_REQUIRED: "auth_required",
}

# Mirror of ``_FAILURE_REASONS`` keyed by the surfaced ``finish_reason`` —
# turns the terminal-state cascade in ``__call__`` into a one-line lookup
# and keeps the wire→exception mapping in one place.
_FAILURE_ERRORS: dict[str, type[Exception]] = {
    "failed": A2ATaskFailedError,
    "cancelled": A2ATaskCancelledError,
    "rejected": A2ATaskRejectedError,
    "auth_required": A2ATaskAuthRequiredError,
}


@dataclass(slots=True)
class A2ADriveState:
    """State accumulated across one ``ask`` to its terminal task; survives ``input_required`` continuations."""

    # Text in order of first appearance; a text artifact owns one slot that later updates extend or replace.
    text_segments: list[str] = field(default_factory=list)
    text_slot: dict[str, int] = field(default_factory=dict)
    # Full text of each artifact as last seen; ``prior_text`` is what earlier tool-result drives consumed.
    artifact_text: dict[str, str] = field(default_factory=dict)
    prior_text: Mapping[str, str] = field(default_factory=dict)
    pending_calls: list[ToolCallEvent] = field(default_factory=list)
    finish_reason: str = "completed"
    terminal_task: Task | None = None
    # Completed artifacts (``last_chunk``): the SDK may replay them on a ``SubscribeToTask``
    # reconnect (spec §3.5.2).
    closed_artifact_ids: set[str] = field(default_factory=set)
    seen_message_ids: set[str] = field(default_factory=set)
    # Artifacts absorbed by this drive; polling re-reads cumulative ``task.artifacts`` on every poll.
    touched_artifact_ids: set[str] = field(default_factory=set)
    # ``(artifact_id, call_id)`` of every tool call surfaced so far, this ask's earlier drives included.
    seen_calls: set[tuple[str, str]] = field(default_factory=set)

    def text(self) -> str:
        return "".join(self.text_segments)

    def add_parts(self, text: str, calls: Sequence[ToolCallEvent]) -> None:
        self.text_segments.append(text)
        self.pending_calls.extend(calls)

    def add_call(self, artifact_id: str, call: ToolCallEvent) -> None:
        """Surface a tool call once; cumulative snapshots re-send calls that earlier drives already surfaced."""
        if (artifact_id, call.id) not in self.seen_calls:
            self.seen_calls.add((artifact_id, call.id))
            self.pending_calls.append(call)

    def absorb_text(self, artifact_id: str, text: str, append: bool) -> str:
        """Fold a text artifact update into its slot; return the newly arrived text.

        A non-append update, live or a snapshot, restates the whole artifact and replaces what
        was known, even when it shortens it. Text consumed by an earlier turn is not part of the
        reply: a restatement that is exactly that text leaves the artifact unchanged and adds nothing.
        """
        prior = self.prior_text.get(artifact_id, "")
        received = self.artifact_text.get(artifact_id, prior)
        content = received + text if append else text
        self.artifact_text[artifact_id] = content
        slot = self.text_slot.get(artifact_id)
        # What this reply already holds for the artifact, as opposed to its full content.
        given = self.text_segments[slot] if slot is not None else None
        if append and given is not None:
            reply_text = given + text
        elif append and content.startswith(prior):
            reply_text = content[len(prior) :]
        elif given is None and content == prior:
            reply_text = ""
        else:
            reply_text = content
        # A restatement that does not extend what was already given out cannot be retracted.
        given = given or ""
        delta = reply_text[len(given) :] if reply_text.startswith(given) else ""
        if slot is not None:
            self.text_segments[slot] = reply_text
        elif reply_text or artifact_id not in self.prior_text:
            self.text_slot[artifact_id] = len(self.text_segments)
            self.text_segments.append(reply_text)
        return delta

    def received_artifacts(self) -> list[Artifact]:
        """Every text artifact as received so far, under its own id; stands in for ``task.artifacts`` when no task was."""
        return [
            Artifact(artifact_id=artifact_id, parts=[Part(text=text)])
            for artifact_id, text in self.artifact_text.items()
            if text
        ]


@dataclass(slots=True)
class A2ATurnOutcome:
    """Per-turn result from a streaming/polling drain; ``input_required`` triggers HITL or tool-call surface."""

    input_required: bool = False
    input_prompt: str | None = None
    # The task reached a terminal or interrupted state (or the peer answered with a Message):
    # nothing further will arrive for this turn, so the task need not be read again.
    settled: bool = False


class A2AClient(LLMClient):
    """``LLMClient`` that delegates to a remote A2A agent.

    Reused across ``reply.ask(...)`` follow-ups on the same ``AgentReply``.
    Within one ``__call__``, ``self._task_id`` carries the server-issued
    id across drives (client-tool round-trips, HITL continuations); it
    resets on each new top-level user turn since the prior task is
    terminal. ``contextId`` persists in ``context.variables`` across asks.

    The client ships the full AG2 history on every turn — the server is
    stateless on AG2 history (see ``mappers/history.py``).
    """

    def __init__(  # type: ignore[no-any-unimported]
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
        input_required_timeout: float | None = None,
        httpx_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        interceptors: Sequence[ClientCallInterceptor] = (),
        grpc_channel_factory: Callable[[str], "grpc.aio.Channel"] | None = None,
        preset_card: AgentCard | None = None,
        card_signature_verifier: CardVerifier | None = None,
        tenant: str | None = None,
        history_length: int | None = None,
        extensions: Sequence[str] = (),
    ) -> None:
        self._card_url = card_url
        self._prefer = prefer
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
        self._card_signature_verifier = card_signature_verifier
        self._tenant = tenant
        self._history_length = history_length
        # Dedup, preserving the user's order — the URIs travel as a list on
        # ``Message.extensions``.
        self._extensions = tuple(dict.fromkeys(extensions))

        self._httpx_client: httpx.AsyncClient | None = None
        self._sdk_client: Client | None = None
        self._agent_card: AgentCard | None = preset_card
        self._task_id: str | None = None
        self._consumed_text: dict[str, str] = {}
        self._consumed_calls: set[tuple[str, str]] = set()
        self._call_context: ClientCallContext | None = None

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

            # Reset task id when starting a new top-level user turn. The
            # prior task is in a terminal state (COMPLETED/CANCELED/...)
            # and the server will reject a continuation; only client-tool
            # round-trips (last message is ``ToolResultsEvent``) reuse
            # the prior task id, since they continue the same task.
            last = messages[-1] if messages else None
            if not isinstance(last, ToolResultsEvent):
                self._task_id = None
                self._consumed_text = {}
                self._consumed_calls = set()

            function_schemas = self._validate_and_extract_tools(tools)
            outgoing = self._build_outgoing(messages, function_schemas, context)

            state = A2ADriveState(prior_text=dict(self._consumed_text), seen_calls=set(self._consumed_calls))
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

                user_text = await context.input(
                    outcome.input_prompt or "Please provide input:",
                    timeout=self._input_required_timeout,
                )
                outgoing = build_input_response_message(
                    user_text,
                    task_id=self._task_id or "",
                    context_id=self._read_context_id(context),
                    extra_extensions=self._extensions,
                )

            for artifact_id in state.touched_artifact_ids:
                self._consumed_text[artifact_id] = state.artifact_text.get(
                    artifact_id, self._consumed_text.get(artifact_id, "")
                )
            self._consumed_calls = state.seen_calls

            if state.terminal_task is not None and (exc_cls := _FAILURE_ERRORS.get(state.finish_reason)):
                raise exc_cls(state.terminal_task)

            message = ModelMessage(state.text()) if state.text() else None

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
        """Release httpx and SDK clients; idempotent."""
        if self._httpx_client is not None:
            await self._httpx_client.aclose()
            self._httpx_client = None
        self._sdk_client = None
        self._call_context = None

    def _verify_card(self, card: AgentCard, *, source: str) -> None:
        """Opt-in JWS check: no-op without a verifier; wraps SDK errors in ag2's."""
        # Anything raised is a rejection. The SDK only guards its key lookup
        # with ``except PyJWTError``, so a key provider that raises (unknown
        # kid, failed JWKS fetch) would otherwise escape ag2's error type.
        if self._card_signature_verifier is None:
            return
        try:
            self._card_signature_verifier(card)
        except Exception as e:
            reason = str(e) if isinstance(e, SignatureVerificationError) else f"{type(e).__name__}: {e}"
            raise A2ACardSignatureError(url=self._card_url, source=source, reason=reason) from e

    async def _ensure_connected(self, context: ConversationContext) -> None:
        if self._sdk_client is not None:
            return
        self._httpx_client = make_httpx_client(
            headers=self._headers,
            timeout=self._timeout,
            factory=self._httpx_client_factory,
        )
        if self._preset_card is None:
            self._agent_card = await A2ACardResolver(
                httpx_client=self._httpx_client, base_url=self._card_url
            ).get_agent_card()
            self._verify_card(self._agent_card, source="fetched agent card")
        else:
            self._verify_card(self._preset_card, source="preset agent card")
        assert self._agent_card is not None
        iface, transport = select_interface(self._agent_card, url=self._card_url, prefer=self._prefer)
        validate_protocol_version(iface, url=self._card_url, transport=transport)
        validate_extension_activation(self._agent_card, self._extensions, url=self._card_url)
        self._call_context = extension_call_context(self._extensions)
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
            extended = await self._sdk_client.get_extended_agent_card(
                GetExtendedAgentCardRequest(**kwargs),
                context=self._call_context,
            )
            self._verify_card(extended, source="extended agent card")
            self._agent_card = extended

    def _validate_and_extract_tools(
        self,
        tools: Iterable[ToolSchema],
    ) -> list[FunctionToolSchema]:
        function_schemas = [t for t in tools if isinstance(t, FunctionToolSchema)]
        if not function_schemas:
            return []
        assert self._agent_card is not None
        if not any(ext.uri == EXTENSION_URI for ext in self._agent_card.capabilities.extensions):
            raise A2AClientToolsNotSupportedError(
                f"Server at {self._card_url!r} does not advertise extension "
                f"{EXTENSION_URI!r}; remove tools= or use a server that supports it."
            )
        return function_schemas

    def _build_outgoing(
        self,
        messages: Sequence[BaseEvent],
        function_schemas: Sequence[FunctionToolSchema],
        context: ConversationContext,
    ) -> Message:
        context_id = self._read_context_id(context)
        last = messages[-1] if messages else None

        # Wire split: ``inputs`` / ``tool_results`` carry the *current*
        # turn; ``history_events`` carries everything *before* it. The
        # AG2 stream stores the current turn as the tail of ``messages``
        # before this client gets called, so we slice it off — otherwise
        # the server replays it via ``history.replace`` and then
        # ``_execute`` re-emits it via ``context.send``, doubling the
        # event in server-side history.
        past_events = messages[:-1]

        if isinstance(last, ToolResultsEvent) and self._task_id is not None:
            return build_tool_result_message(
                last.results,
                history_events=past_events,
                tool_schemas=function_schemas,
                task_id=self._task_id,
                context_id=context_id,
                context_update=_outgoing_variables(context),
                extra_extensions=self._extensions,
            )

        inputs: list[Input] = next(
            (list(ev.parts) for ev in reversed(messages) if isinstance(ev, ModelRequest)),
            [TextInput("")],
        )
        extra_parts = _read_extra_parts(context)
        return build_user_message(
            inputs,
            history_events=past_events,
            tool_schemas=function_schemas,
            task_id=self._task_id,
            context_id=context_id,
            advertise_extension=bool(function_schemas) or self._task_id is not None,
            context_update=_outgoing_variables(context),
            extra_parts=extra_parts,
            extra_extensions=self._extensions,
        )

    async def _drive_task(
        self,
        message: Message,
        context: ConversationContext,
        state: A2ADriveState,
    ) -> A2ATurnOutcome:
        assert self._agent_card is not None
        if self._streaming and self._agent_card.capabilities.streaming:
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
        stream = self._sdk_client.send_message(request, context=self._call_context)

        attempt = 0
        while True:
            try:
                outcome = await self._drain_stream(stream, context, state)
                if outcome.settled or self._task_id is None:
                    return outcome
                # The stream closed before the task reached a final state; the final state may
                # have been recorded without reaching the stream (a cancel, notably), so read it.
                return await self._poll_until_settled(context, state)
            except UnsupportedOperationError:
                # Spec §3.1.6: a task that ended while the stream was down cannot be subscribed to;
                # its final state is read instead. On the opening request this is a real failure.
                if attempt == 0:
                    raise
                return await self._poll_until_settled(context, state)
            except A2AClientError as exc:
                if self._task_id is None or attempt >= self._max_reconnects:
                    raise A2AReconnectError(attempt) from exc
                attempt += 1
                backoff = self._reconnect_backoff * (2 ** (attempt - 1))
                await asyncio.sleep(backoff)
                resubscribe = SubscribeToTaskRequest(**self._maybe_tenant(context, id=self._task_id))
                stream = self._sdk_client.subscribe(resubscribe, context=self._call_context)

    async def _consume_polling(
        self,
        message: Message,
        context: ConversationContext,
        state: A2ADriveState,
    ) -> A2ATurnOutcome:
        assert self._sdk_client is not None

        request = self._build_send_request(message, context)
        outcome = await self._drain_stream(
            self._sdk_client.send_message(request, context=self._call_context),
            context,
            state,
        )
        if outcome.settled or self._task_id is None:
            return outcome

        return await self._poll_until_settled(context, state)

    async def _poll_until_settled(self, context: ConversationContext, state: A2ADriveState) -> A2ATurnOutcome:
        assert self._sdk_client is not None

        while True:
            get_kwargs = self._maybe_tenant(context, id=self._task_id)
            if self._history_length is not None:
                get_kwargs["history_length"] = self._history_length
            task = await self._sdk_client.get_task(
                GetTaskRequest(**get_kwargs),
                context=self._call_context,
            )
            outcome = await self._absorb_task(task, context, state)
            if outcome.settled:
                return outcome
            await asyncio.sleep(self._polling_interval)

    async def _drain_stream(
        self,
        stream: AsyncIterator[StreamResponse],
        context: ConversationContext,
        state: A2ADriveState,
    ) -> A2ATurnOutcome:
        outcome = A2ATurnOutcome()
        async for raw in stream:
            response = _ensure_stream_response(raw)
            a2a_event = parse_stream_response(response)
            # Publish the typed A2A event into the user-facing stream so
            # observers/middleware can subscribe via stream.where(A2A...).
            await context.send(a2a_event)

            if isinstance(a2a_event, A2ATaskSnapshot):
                self._task_id = a2a_event.task.id
                self._save_context_id(context, a2a_event.task.context_id)
                # The task as it stands: a resubscribe opens with it (covering what streamed while
                # disconnected), and a stream may carry nothing else, or close on a final snapshot.
                snapshot_outcome = await self._absorb_task(a2a_event.task, context, state)
                if snapshot_outcome.settled:
                    return snapshot_outcome
                continue

            if isinstance(a2a_event, A2ATaskStatusUpdate):
                update = a2a_event.update
                self._save_context_id(context, update.context_id)
                if update.task_id:
                    self._task_id = update.task_id
                if update.status.state in _TERMINAL_STATES:
                    outcome.settled = True
                stop = await self._handle_status_update(update, context, state, outcome)
                if stop:
                    return outcome
                continue

            if isinstance(a2a_event, A2ATaskArtifactUpdate):
                self._save_context_id(context, a2a_event.update.context_id)
                await self._apply_artifact_update(a2a_event, context, state)
                continue

            if isinstance(a2a_event, A2AMessage):
                msg = a2a_event.message
                # A Message answers the request outright; no task updates follow it.
                outcome.settled = True
                self._save_context_id(context, msg.context_id)
                if msg.task_id:
                    self._task_id = msg.task_id
                if msg.message_id and msg.message_id in state.seen_message_ids:
                    continue
                self._merge_context_update(context, extract_context_update(msg))
                text_chunk, calls = await self._handle_artifact_parts(msg.parts, context)
                state.add_parts(text_chunk, calls)
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

        if reason := _FAILURE_REASONS.get(sd_state):
            state.terminal_task = Task(
                id=status_update.task_id,
                status=status_update.status,
                context_id=status_update.context_id,
                artifacts=state.received_artifacts(),
            )
            state.finish_reason = reason
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
            await self._absorb_completion_message(status_update.status.message, context, state)
        return False

    async def _absorb_completion_message(
        self,
        msg: Message,
        context: ConversationContext,
        state: A2ADriveState,
    ) -> None:
        """Absorb text/tool-calls from COMPLETED's ``status.message``; idempotent on ``message_id``."""
        if msg.message_id and msg.message_id in state.seen_message_ids:
            return
        self._merge_context_update(context, extract_context_update(msg))
        text_chunk, calls = await self._handle_artifact_parts(msg.parts, context)
        state.add_parts(text_chunk, calls)
        if msg.message_id:
            state.seen_message_ids.add(msg.message_id)

    async def _absorb_task(self, task: Task, context: ConversationContext, state: A2ADriveState) -> A2ATurnOutcome:
        """Fold a task snapshot (polled, or a stream's) into drive state; ``settled`` once it is final or interrupted.

        Each snapshot is the current full content of every artifact, reconciled against what was
        already absorbed. Tool calls are only surfaced on ``INPUT_REQUIRED``, where ``status.message``
        is the HITL prompt rather than assistant text; only ``COMPLETED`` contributes its message to
        the reply, the same as a streamed status update.
        """
        status = task.status
        input_required = status.state == TaskState.TASK_STATE_INPUT_REQUIRED
        await self._absorb_snapshot_artifacts(task, context, state, terminal_calls_visible=input_required)

        if reason := _FAILURE_REASONS.get(status.state):
            state.terminal_task = task
            state.finish_reason = reason
            return A2ATurnOutcome(settled=True)
        if input_required:
            state.finish_reason = "input_required"
            return A2ATurnOutcome(input_required=True, input_prompt=_extract_status_prompt(status), settled=True)
        if status.state == TaskState.TASK_STATE_COMPLETED:
            if status.HasField("message"):
                await self._absorb_completion_message(status.message, context, state)
            return A2ATurnOutcome(settled=True)
        return A2ATurnOutcome()

    async def _absorb_snapshot_artifacts(
        self,
        task: Task,
        context: ConversationContext,
        state: A2ADriveState,
        *,
        terminal_calls_visible: bool,
    ) -> None:
        for artifact in task.artifacts:
            a2a_event = parse_task_artifact(artifact, task_id=task.id, context_id=task.context_id)
            if isinstance(a2a_event, A2AToolCallArtifact) and not terminal_calls_visible:
                # Left unseen: the call is only surfaced once the task asks for input.
                continue
            await self._apply_artifact_update(a2a_event, context, state)

    async def _apply_artifact_update(
        self,
        a2a_event: A2ATaskArtifactUpdate,
        context: ConversationContext,
        state: A2ADriveState,
    ) -> None:
        """Apply an artifact update to drive state; typed subclasses route directly, others scan parts."""
        update = a2a_event.update
        artifact = update.artifact
        artifact_id = artifact.artifact_id
        if artifact_id in state.closed_artifact_ids:
            return
        state.touched_artifact_ids.add(artifact_id)

        if isinstance(a2a_event, A2ATextArtifact):
            if fresh := state.absorb_text(artifact_id, a2a_event.text, a2a_event.append):
                await context.send(ModelMessageChunk(fresh))
        elif isinstance(a2a_event, A2AToolCallArtifact):
            state.add_call(artifact_id, a2a_event.call)
        else:
            # Mixed artifact: its text reconciles by identity like a text artifact, its calls by call id.
            text = "".join(part.text for part in artifact.parts)
            if fresh := state.absorb_text(artifact_id, text, a2a_event.append):
                await context.send(ModelMessageChunk(fresh))
            for part in artifact.parts:
                if is_data_part_with_mime(part, MIME_TOOL_CALL):
                    state.add_call(artifact_id, payload_to_call(part_data_to_python(part)))

        if a2a_event.last_chunk:
            state.closed_artifact_ids.add(artifact_id)

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

    def _maybe_tenant(self, context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        override = context.variables.get(TENANT_VARIABLE_KEY)
        tenant = override if isinstance(override, str) and override else self._tenant
        if tenant:
            kwargs["tenant"] = tenant
        return kwargs

    @staticmethod
    def _merge_context_update(context: ConversationContext, payload: Mapping[str, Any]) -> None:
        if not payload:
            return
        context.variables.update(strip_reserved_variables(payload, source="an A2A peer response"))


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
    """Read user-supplied extra ``Part``s from context dependencies; advisory, silently filtered."""
    raw = context.dependencies.get(EXTRA_PARTS_DEPENDENCY_KEY)
    if not raw:
        return []
    return [p for p in raw if isinstance(p, Part)]


def _outgoing_variables(context: ConversationContext) -> dict[str, Any] | None:
    """The variables sync this request carries, or ``None`` when nothing is left to send.

    The peer has no business reading this side's control-plane state — including
    the context ids, which ride their own protocol field — so reserved keys stay home.
    """
    return strip_reserved_variables(context.variables, source="an outgoing A2A request", warn=False) or None
