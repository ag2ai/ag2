# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
from a2a.utils.errors import A2AError, UnsupportedOperationError
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

from ._states import SETTLED_STATES
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

logger = logging.getLogger(__name__)

_PROVIDER = "a2a"
_CONTEXT_ID_VAR_TEMPLATE = "a2a:context_id:{card_url}"

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
class _ArtifactText:
    """One artifact of the remote task, reconciled by identity across every turn on that task.

    ``baseline`` is the content when the current turn began. The turn's reply is the content past
    it when the content extends it, the whole content otherwise. It depends only on the baseline
    and the latest content, so live updates and snapshots, and therefore streaming and polling,
    agree on it.

    So a full re-send that begins with an earlier turn's text continues it, even when the server
    meant a whole new answer: a polled snapshot of appended text is byte-identical, and telling
    the two apart only when streaming would make the modes disagree. A server that wants a new
    answer in a later turn uses a new artifact id, as the AG2 server does. This reading is
    proposed and awaits maintainer agreement (PR #3494, Open Decision 1), as does the matching
    paragraph of ``client.mdx``.

    The reply orders artifacts by their first appearance on the task, not in the turn: that is
    the order of ``task.artifacts``, which is all a polling client sees, so a reused artifact
    keeps its place ahead of one the current turn added.
    """

    # Complete received parts/metadata, used when a terminal status has no task snapshot.
    received: Artifact | None = None
    content: str = ""
    baseline: str = ""
    # Reply text already sent as ``ModelMessageChunk`` in this turn; it cannot be retracted.
    emitted: str = ""
    # A ``last_chunk`` arrived in the current run: the SDK may replay the artifact on a
    # ``SubscribeToTask`` reconnect (spec §3.5.2).
    closed: bool = False

    def reply(self) -> str:
        baseline = self.baseline
        return self.content[len(baseline) :] if self.content.startswith(baseline) else self.content

    def merged_artifact(self, artifact: Artifact, append: bool) -> Artifact:
        """Build the updated snapshot without mutating the prior turn's shared snapshot."""
        snapshot = Artifact()
        if append and self.received is not None:
            snapshot.CopyFrom(self.received)
            snapshot.parts.extend(artifact.parts)
            snapshot.metadata.update(dict(artifact.metadata.items()))
        else:
            snapshot.CopyFrom(artifact)
        return snapshot

    def update(self, artifact: Artifact, append: bool) -> str:
        """Apply a live update or a snapshot (``append=False`` restates the whole artifact); return the new reply text."""
        self.received = self.merged_artifact(artifact, append)
        self.content = _joined_text(self.received.parts)
        reply = self.reply()
        delta = reply[len(self.emitted) :] if reply.startswith(self.emitted) else ""
        self.emitted += delta
        return delta

    def begin_turn(self) -> None:
        self.baseline = self.content
        self.emitted = ""


@dataclass(slots=True)
class A2ADriveState:
    """State accumulated across one turn to its settled task; survives ``input_required`` continuations."""

    # Every artifact of the task, in order of first appearance on it; the client keeps it across
    # the turns on that task.
    artifacts: dict[str, _ArtifactText] = field(default_factory=dict)
    # ``(artifact_id, call_id)`` of every tool call surfaced so far, this task's earlier turns
    # included; kept by the client like ``artifacts``.
    seen_calls: set[tuple[str, str]] = field(default_factory=set)
    # Text of the messages answering this turn, in arrival order. They end a run, so they follow the artifacts.
    messages: list[str] = field(default_factory=list)
    pending_calls: list[ToolCallEvent] = field(default_factory=list)
    finish_reason: str = "completed"
    terminal_task: Task | None = None
    seen_message_ids: set[str] = field(default_factory=set)
    # An update would have changed an artifact closed in this run; only the stored task can say
    # whether it was a replay or text the server really wrote after ``last_chunk``.
    closed_artifact_changed: bool = False

    def text(self) -> str:
        return "".join(artifact.reply() for artifact in self.artifacts.values()) + "".join(self.messages)

    def begin_run(self) -> None:
        """Reopen the task's artifacts for a new run (the first, or a HITL continuation)."""
        for artifact in self.artifacts.values():
            artifact.closed = False
        self.closed_artifact_changed = False

    def add_parts(self, text: str, calls: Sequence[ToolCallEvent]) -> None:
        self.messages.append(text)
        self.pending_calls.extend(calls)

    def add_call(self, artifact_id: str, call: ToolCallEvent) -> None:
        """Surface a tool call once; cumulative snapshots re-send calls that earlier turns already surfaced."""
        if (artifact_id, call.id) not in self.seen_calls:
            self.seen_calls.add((artifact_id, call.id))
            self.pending_calls.append(call)

    def received_artifacts(self) -> list[Artifact]:
        """Every artifact as received so far, including empty parts and nontext content."""
        return [artifact.received for artifact in self.artifacts.values() if artifact.received is not None]


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
        self._artifacts: dict[str, _ArtifactText] = {}
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
                self._artifacts = {}
                self._consumed_calls = set()

            function_schemas = self._validate_and_extract_tools(tools)
            outgoing = self._build_outgoing(messages, function_schemas, context)

            # The drive works on a copy that is committed once it ends, so a drive that raises leaves
            # the task's artifacts and calls as they were, and a retry of this turn starts from them.
            state = A2ADriveState(
                artifacts={artifact_id: replace(artifact) for artifact_id, artifact in self._artifacts.items()},
                seen_calls=set(self._consumed_calls),
            )
            # A new turn on the same task: what the task's artifacts hold now was the earlier turns' reply.
            for artifact in state.artifacts.values():
                artifact.begin_turn()
            while True:
                # A HITL continuation stays in this turn: its result replaces the previous
                # INPUT_REQUIRED reason and may update artifacts closed in the previous run.
                # Keep their content and baselines, but reopen them.
                state.finish_reason = "completed"
                state.begin_run()
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
            self._artifacts = state.artifacts
            self._consumed_calls = state.seen_calls

            if state.terminal_task is not None and (exc_cls := _FAILURE_ERRORS.get(state.finish_reason)):
                raise exc_cls(state.terminal_task)

            if state.pending_calls and state.finish_reason != "input_required":
                # Only a task waiting for input takes tool results back; a finished one would refuse them.
                logger.warning(
                    "A2A peer requested client tool calls %s on a task that did not wait for input; ignoring them",
                    [call.name for call in state.pending_calls],
                )
                state.pending_calls.clear()

            text = state.text()
            message = ModelMessage(text) if text else None

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
            interface=iface,
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
            outcome = await self._consume_streaming(message, context, state)
        else:
            outcome = await self._consume_polling(message, context, state)
        if state.closed_artifact_changed and self._task_id is not None:
            try:
                await self._read_back_artifacts(context, state)
            except (A2AClientError, A2AError):
                # The turn is already settled; the read-back only reconciles, so keep what arrived.
                logger.warning("A2A task %s: reading back its artifacts failed", self._task_id, exc_info=True)
        return outcome

    async def _read_back_artifacts(self, context: ConversationContext, state: A2ADriveState) -> None:
        """Reconcile the artifacts with the stored task, which a polling client would have read.

        Only the artifacts: the run's status and messages are already absorbed, and a message
        without an id would be counted twice.
        """
        assert self._sdk_client is not None
        task = await self._sdk_client.get_task(
            GetTaskRequest(**self._maybe_tenant(context, id=self._task_id)),
            context=self._call_context,
        )
        state.begin_run()
        for artifact in task.artifacts:
            await self._apply_artifact_update(
                parse_task_artifact(artifact, task_id=task.id, context_id=task.context_id), context, state
            )
        if state.terminal_task is not None:
            state.terminal_task = task

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
                if update.status.state in SETTLED_STATES:
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
        """Absorb text/tool-calls from COMPLETED's ``status.message``; idempotent on ``message_id``.

        A message whose text is exactly the reply the artifacts already carried restates it, and adds no text.
        """
        if msg.message_id and msg.message_id in state.seen_message_ids:
            return
        self._merge_context_update(context, extract_context_update(msg))
        parts: Iterable[Part] = msg.parts
        if (reply := state.text()) and _joined_text(msg.parts) == reply:
            parts = [part for part in msg.parts if not part.text]
        text_chunk, calls = await self._handle_artifact_parts(parts, context)
        state.add_parts(text_chunk, calls)
        if msg.message_id:
            state.seen_message_ids.add(msg.message_id)

    async def _absorb_task(
        self,
        task: Task,
        context: ConversationContext,
        state: A2ADriveState,
    ) -> A2ATurnOutcome:
        """Fold a task snapshot (polled, or a stream's) into drive state; ``settled`` once it is final or interrupted.

        Each snapshot is the current full content of every artifact, reconciled against what was
        already absorbed; a tool call it repeats is surfaced once. Only ``COMPLETED`` contributes its
        message to the reply, the same as a streamed status update: on ``INPUT_REQUIRED`` the message
        is the HITL prompt rather than assistant text.
        """
        status = task.status
        input_required = status.state == TaskState.TASK_STATE_INPUT_REQUIRED
        for artifact in task.artifacts:
            await self._apply_artifact_update(
                parse_task_artifact(artifact, task_id=task.id, context_id=task.context_id), context, state
            )

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

    async def _apply_artifact_update(
        self,
        a2a_event: A2ATaskArtifactUpdate,
        context: ConversationContext,
        state: A2ADriveState,
    ) -> None:
        """Apply an artifact update to drive state; typed subclasses route directly, others scan parts."""
        artifact = a2a_event.update.artifact
        artifact_id = artifact.artifact_id
        record = state.artifacts.setdefault(artifact_id, _ArtifactText())
        calls = _artifact_calls(a2a_event)
        if record.closed:
            if record.merged_artifact(artifact, a2a_event.append) != record.received:
                state.closed_artifact_changed = True
            return

        # A mixed artifact's text reconciles like a text artifact's; its calls dedup by call id.
        if fresh := record.update(artifact, a2a_event.append):
            await context.send(ModelMessageChunk(fresh))
        for call in calls:
            state.add_call(artifact_id, call)

        if a2a_event.last_chunk:
            record.closed = True

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


def _joined_text(parts: Iterable[Part]) -> str:
    return "".join(part.text for part in parts)


def _artifact_calls(a2a_event: A2ATaskArtifactUpdate) -> list[ToolCallEvent]:
    if isinstance(a2a_event, A2AToolCallArtifact):
        return [a2a_event.call]
    if isinstance(a2a_event, A2ATextArtifact):
        return []
    return [
        payload_to_call(part_data_to_python(part))
        for part in a2a_event.update.artifact.parts
        if is_data_part_with_mime(part, MIME_TOOL_CALL)
    ]


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
