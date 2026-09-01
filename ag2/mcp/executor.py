# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AbstractContextManager, ExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.shared.exceptions import MCPError
from mcp.types import (
    INVALID_PARAMS,
    CallToolResult,
    ContentBlock,
    InputRequiredResult,
    InputResponses,
    TextContent,
)
from mcp.types import Tool as MCPTool
from mcp_types.version import MODERN_PROTOCOL_VERSIONS
from pydantic import ValidationError

from ag2.agent import Agent
from ag2.events import (
    BaseEvent,
    HumanInputRequest,
    ModelMessageChunk,
    TextInput,
    ToolCallEvent,
    ToolResultEvent,
)
from ag2.hitl import ElicitationPolicy
from ag2.stream import MemoryStream

from .elicitation import ClientElicitor
from .errors import MCPAgentConfigError, MCPSamplingUnavailableError, UnknownConversationError
from .info import build_ask_tool, object_output_schema
from .mappers import reply_to_content, to_structured_dict, tool_error
from .pause import PauseState, PausedRuns, SuspendedTurn, question_digest
from .sampling import ClientModel, ClientModelConfig, client_can_sample
from .sessions import CONVERSATION_META_KEY, STDIO_SESSION, Conversation, SessionStore

if TYPE_CHECKING:
    from mcp.server.context import ServerRequestContext

# ``mcp`` 2.0's ``on_call_tool`` handler returns a complete result model, so the
# executor builds one for every outcome — success, structured success, and error.

_LOGGER_NAME = "ag2.mcp"


@dataclass(slots=True)
class AskContext:
    """Per-request context to inject into the agent turn — the kwargs
    :meth:`Agent.ask` accepts. Returned by a ``context_provider``; any field
    left ``None`` is omitted, so the default is the stateless behavior."""

    variables: dict[str, Any] | None = None
    tools: list[Any] | None = None
    prompt: list[str] | str | None = None


# Async hook: given the request's authenticated token (or ``None``), return the
# per-request :class:`AskContext` to feed into ``Agent.ask``. Lets a host inject
# session context (variables / tools / prompt) the stateless executor otherwise
# omits — e.g. resolving the principal from the token and loading their tools.
ContextProvider = Callable[[AccessToken | None], Awaitable[AskContext]]


class AgentExecutor:
    """Bridge an MCP ``tools/call`` to a single :meth:`Agent.ask` turn.

    Without a ``session_store`` each call is stateless: a fresh
    :class:`MemoryStream` is created per invocation (mirroring the A2A executor)
    so any server replica can handle any request.

    With a ``session_store`` the conversation a call lands in depends on whether
    the caller named one and on the protocol era, because each era sanctions a
    different mechanism. :class:`~ag2.mcp.MCPServer` carries that table — the one
    a user reads — and :meth:`_conversation_cm` below is where it is decided;
    restating it here would give it a second copy to drift from.

    While the agent runs, its stream events are forwarded to the MCP client as
    progress / log notifications when ``stream_progress`` is enabled.
    """

    __slots__ = (
        "_agent",
        "_tool_name",
        "_tool_description",
        "_stream_progress",
        "_context_provider",
        "_session_store",
        "_elicitation_policy",
        "_client_model",
        "_paused",
    )

    def __init__(
        self,
        agent: Agent,
        *,
        tool_name: str = "ask",
        tool_description: str | None = None,
        stream_progress: bool = True,
        context_provider: "ContextProvider | None" = None,
        session_store: SessionStore | None = None,
        elicitation_policy: ElicitationPolicy = "ask",
        client_model: ClientModel | None = None,
        paused_runs: PausedRuns | None = None,
    ) -> None:
        self._agent = agent
        self._tool_name = tool_name
        self._tool_description = tool_description
        self._stream_progress = stream_progress
        self._context_provider = context_provider
        self._session_store = session_store
        self._elicitation_policy = elicitation_policy
        # Off unless the operator passed one: borrowing the caller's model moves
        # cost, capability and reproducibility to them, so it is never on because
        # the transport allows it.
        self._client_model = client_model
        # Where a modern-era turn waits between rounds. ``None`` leaves the
        # modern era without the pause transport, which is what a caller
        # constructing this executor directly (no ``requestState`` protection
        # installed) gets: there would be nowhere safe to put the state.
        self._paused = paused_runs

    def list_tools(self) -> list[MCPTool]:
        return [
            build_ask_tool(
                self._agent,
                tool_name=self._tool_name,
                tool_description=self._tool_description,
                response_schema=self._agent._response_schema,
                conversation_bounds=self._session_store.bounds if self._session_store is not None else None,
            )
        ]

    async def call(
        self,
        name: str,
        *,
        message: str,
        context: str | None = None,
        conversation: str | None = None,
        input_responses: "InputResponses | None" = None,
        request_state: str | None = None,
        request_context: "ServerRequestContext[Any, Any]",
    ) -> "CallToolResult | InputRequiredResult":
        if name != self._tool_name:
            return tool_error(f"Unknown tool: {name!r}.")

        # A retry answering a question this server asked. It resumes the run
        # already held for it, so nothing about the original arguments is read
        # again — and nothing needs to be: the state boundary has already bound
        # this token to the same tool, the same arguments and the same principal,
        # and refused it otherwise.
        if request_state is not None:
            return await self._resume(request_state, input_responses, request_context)

        # A deployment running on its callers' models is allowed to hold no model
        # of its own — that is the case this exists for. A caller that cannot
        # lend one is then a different failure, reported where that is known.
        if self._agent.config is None and self._client_model is None:
            raise MCPAgentConfigError(self._agent.name)
        if not message:
            return tool_error("Missing required 'message' argument.")

        # A blank handle names no conversation, so it is read as no conversation
        # rather than as an unknown one. The caller this matters for is the model:
        # the handle is put in readable content precisely because the model drives
        # the argument, and a model given an optional string argument routinely
        # sends "" instead of leaving the key out — which would make its every
        # first call an error and leave it unable to start a conversation at all.
        # Nothing is lost: no minted handle is blank.
        if conversation is not None and not conversation.strip():
            conversation = None

        # Conversations are off, so a presented handle names nothing here and
        # never could: this server mints none. Saying so is the whole point —
        # accepting the argument and dropping it would hand the caller the
        # unannounced loss of continuity this argument exists to remove. It is
        # deliberately not ``UnknownConversationError``: that error's remedy is
        # to retry without the argument, which here would not restore continuity
        # either, and the handle is not unknown so much as unsupported.
        if conversation is not None and self._session_store is None:
            return tool_error(
                "This server does not maintain conversations, so the 'conversation' argument is "
                "not supported; omit it. Each call is independent."
            )

        # The conversation is held for the whole turn: for a keyed one that means
        # holding its turn lock, serializing concurrent same-conversation calls.
        # A modern-era turn that pauses releases that lock while it waits — a
        # paused run is not a running one, and holding the lock would block the
        # very retry that resumes it.
        try:
            async with self._conversation_cm(request_context, conversation) as convo:
                if self._paused is None or not _is_modern(request_context):
                    return await self._turn(convo, message, context, request_context)
                return await self._start_suspendable(convo, message, context, request_context)
        except UnknownConversationError as e:
            # A tool-level error, never a JSON-RPC one: the protocol draws that
            # line so the model can start a new conversation rather than fail the
            # turn. Only resolving the conversation raises this.
            return tool_error(str(e))

    async def _start_suspendable(
        self,
        convo: Conversation,
        message: str,
        context: str | None,
        request_context: "ServerRequestContext[Any, Any]",
    ) -> "CallToolResult | InputRequiredResult":
        """Begin a modern-era turn that may come back as a question.

        The turn is launched as a task rather than awaited, so that when a tool
        inside it asks the client something this call can return the question
        while the run stays suspended, exactly where it was.
        """
        assert self._paused is not None
        turn = SuspendedTurn(conversation=convo.handle, stream=convo.stream, created=self._paused.now())
        turn.start(
            self._turn(
                convo,
                message,
                context,
                request_context,
                suspended=turn,
                progress=False,
            )
        )
        return await self._advance(turn, convo.stream, request_context)

    async def _resume(
        self,
        request_state: str,
        input_responses: "InputResponses | None",
        request_context: "ServerRequestContext[Any, Any]",
    ) -> "CallToolResult | InputRequiredResult":
        """Continue the run this state names, from exactly where it stopped.

        The state arrives boundary-verified — expiry, request binding, audience
        and principal all checked, fail-closed — so what is left to establish is
        only whether the run it names is still here.
        """
        if self._paused is None:
            # No modern-era transport was configured, so this server has never
            # minted state and cannot have paused a run.
            return tool_error("This server does not pause calls, so there is no state to resume.")
        state = PauseState.decode(request_state)
        turn = self._paused.take(state.pause) if state is not None else None
        if state is None or turn is None:
            # A protocol error, not a tool error: the caller's remedy is to start
            # the call again, and the model cannot recover from it by rewording.
            raise MCPError(
                code=INVALID_PARAMS,
                message="Invalid or expired requestState",
                data={"reason": "invalid_request_state"},
            )
        answer = (input_responses or {}).get(state.question)
        if answer is not None:
            # ``False`` means the answer was not for the question the run is
            # actually waiting on; nothing is consumed and the current question
            # is asked again below. Whether it is the *kind* of answer the run
            # asked for is the asker's to judge, not this frame's.
            turn.answer(state.question, state.digest, answer)
        return await self._advance(turn, turn.stream, request_context)

    async def _advance(
        self,
        turn: SuspendedTurn,
        stream: MemoryStream,
        request_context: "ServerRequestContext[Any, Any]",
    ) -> "CallToolResult | InputRequiredResult":
        """Run ``turn`` until it finishes or asks, and shape whichever happens.

        Progress forwarding is scoped to this round because it holds the round's
        request context: notifications belong to the call being answered now, not
        to the one that started the run.
        """
        assert self._paused is not None
        with ExitStack() as stack:
            if self._stream_progress:
                stack.enter_context(_progress_scope(stream, request_context))
            outcome = await turn.advance()
        if isinstance(outcome, CallToolResult):
            return outcome
        key, request = outcome
        self._paused.register(turn)
        state = PauseState.mint(
            pause=turn.id,
            question=key,
            digest=question_digest(request),
        )
        # The state goes out in plaintext here and is sealed by the
        # ``RequestStateBoundary`` at the wire boundary, which is the only place
        # the codec is ever touched.
        return InputRequiredResult(input_requests={key: request}, request_state=state.encode())

    async def _turn(
        self,
        convo: Conversation,
        message: str,
        context: str | None,
        request_context: "ServerRequestContext[Any, Any]",
        *,
        suspended: SuspendedTurn | None = None,
        progress: bool = True,
    ) -> CallToolResult:
        """Run one agent turn inside ``convo`` and shape its reply into a result.

        ``suspended`` is present on the modern era, where a question comes back
        as the call's result: the elicitor asks *through* it and the coroutine
        stays parked here until a retry answers.

        ``progress`` is off for a suspendable turn, because this coroutine spans
        more than one round and forwarding belongs to the round being answered —
        :meth:`_advance` scopes it per round instead.
        """
        with ExitStack() as progress_stack:
            if progress and self._stream_progress:
                progress_stack.enter_context(_progress_scope(convo.stream, request_context))
            return await self._run(convo, message, context, request_context, suspended)

    async def _run(
        self,
        convo: Conversation,
        message: str,
        context: str | None,
        request_context: "ServerRequestContext[Any, Any]",
        suspended: SuspendedTurn | None,
    ) -> CallToolResult:

        # Optional per-request context (variables/tools/prompt) from the host,
        # derived from the authenticated token. Omitted fields keep ask()'s
        # defaults, so without a provider this is the stateless behavior.
        ask_kwargs: dict[str, Any] = {}
        if self._context_provider is not None:
            ctx = await self._context_provider(get_access_token())
            if ctx.variables is not None:
                ask_kwargs["variables"] = ctx.variables
            if ctx.tools is not None:
                ask_kwargs["tools"] = ctx.tools
            if ctx.prompt is not None:
                ask_kwargs["prompt"] = ctx.prompt

        model = self._client_model_config(request_context, suspended)
        if model is not None:
            ask_kwargs["config"] = model

        # Scoped to this turn, and registered *before* ``ask`` so it runs ahead
        # of whatever the agent registers for itself: the calling client's human
        # is asked first, and a client that cannot answer falls through to the
        # agent's own ``hitl_hook`` — or to the existing "nobody could be asked"
        # failure. Scoped because the elicitor holds this call's request context,
        # and a keyed conversation's stream outlives the call.
        with ExitStack() as stack:
            stack.enter_context(
                convo.stream.where(HumanInputRequest).sub_scope(
                    self._elicitor(request_context, suspended),
                    interrupt=True,
                )
            )
            reply = await self._agent.ask(*_build_inputs(message, context), stream=convo.stream, **ask_kwargs)
        content = reply_to_content(reply)

        if not self._has_object_output():
            return _result(content, handle=convo.handle)

        try:
            validated = await reply.content()
        except ValidationError as e:
            return _result(
                [TextContent(type="text", text=f"Structured-output validation failed: {e}")],
                handle=convo.handle,
                is_error=True,
            )
        structured = to_structured_dict(validated)
        if structured is None:
            return _result(content, handle=convo.handle, is_error=True)
        return _result(content, handle=convo.handle, structured=structured)

    def _conversation_cm(
        self,
        request_context: "ServerRequestContext[Any, Any]",
        conversation: str | None,
    ) -> AbstractAsyncContextManager[Conversation]:
        """The conversation this call runs in, resolved in the order the protocol allows.

        A named conversation wins in either era. Otherwise the handshake era
        falls back to the MCP session it already has, and the modern era — which
        has none, and may not derive one from the connection or the process —
        starts a fresh one. A handle the registry does not know raises rather
        than falling through: falling through would let a caller name a
        conversation of their choosing and evict other callers' out of the bound.

        Either name is answerable only to the principal that created it, on every
        call rather than at creation. A handle is revalidated here, by the store,
        because a handle travels through model context and logs. An MCP session
        id needs no check of ours: the transport already refuses a session id
        presented with a credential other than the one that opened it, answering
        as though the session did not exist, so a swapped credential never
        reaches this far with that name.
        """
        if self._session_store is None:
            return _stateless_conversation()
        principal = _principal()
        if conversation is not None:
            return self._session_store.by_handle(conversation, principal=principal)
        if not _is_modern(request_context) and (session_id := _session_id(request_context)) is not None:
            return self._session_store.session(session_id, principal=principal)
        return self._session_store.fresh(principal=principal)

    def _elicitor(
        self,
        request_context: "ServerRequestContext[Any, Any]",
        suspended: SuspendedTurn | None,
    ) -> ClientElicitor:
        """The hook that puts this turn's questions to the calling client."""
        return ClientElicitor(request_context, policy=self._elicitation_policy, suspended=suspended)

    def _client_model_config(
        self,
        request_context: "ServerRequestContext[Any, Any]",
        suspended: SuspendedTurn | None,
    ) -> ClientModelConfig | None:
        """The peer-backed model for this turn, or ``None`` to use the agent's own.

        Raises:
            MCPSamplingUnavailableError: The server was configured to run on its
                caller's model, this caller advertised none, and no fallback was
                allowed. Failing here — before the turn starts — is what stops a
                caller being handed an answer from a model they did not lend.
        """
        if self._client_model is None:
            return None
        if not client_can_sample(request_context.session):
            # Falling back needs something to fall back *to*; a server with
            # neither is misconfigured, and says the same thing either way.
            if self._client_model.fallback and self._agent.config is not None:
                return None
            raise MCPSamplingUnavailableError()
        return ClientModelConfig(
            request_context,
            suspended=suspended,
            max_tokens=self._client_model.max_tokens,
        )

    def _has_object_output(self) -> bool:
        return object_output_schema(self._agent._response_schema) is not None


def _progress_scope(
    stream: MemoryStream,
    request_context: "ServerRequestContext[Any, Any]",
) -> "AbstractContextManager[None]":
    """Forward this stream's events to the client for as long as the scope is open.

    Scoped rather than left installed: the forwarder holds one round's request
    context, and both a keyed conversation's stream and a suspended run outlive
    the round that created it.
    """
    return stream.sub_scope(_ProgressForwarder(request_context))


class _ProgressForwarder:
    """Turns agent stream events into MCP progress / log notifications."""

    __slots__ = ("_token", "_session", "_progress")

    def __init__(self, request_context: "ServerRequestContext[Any, Any]") -> None:
        # ``_meta`` is an open mapping in ``mcp`` 2.0, not a model.
        self._token = request_context.meta.get("progress_token") if request_context.meta else None
        self._session = request_context.session
        self._progress = _Counter()

    async def __call__(self, event: BaseEvent) -> None:
        if isinstance(event, ModelMessageChunk):
            if self._token is not None:
                await self._session.send_progress_notification(
                    self._token, self._progress.next(), message=event.content
                )
            return
        if isinstance(event, ToolResultEvent):
            await self._session.send_log_message("info", f"tool result: {event.name}", logger=_LOGGER_NAME)
            return
        if isinstance(event, ToolCallEvent):
            await self._session.send_log_message("info", f"tool call: {event.name}", logger=_LOGGER_NAME)


class _Counter:
    """Monotonically increasing float source for MCP progress values."""

    __slots__ = ("_value",)

    def __init__(self) -> None:
        self._value = 0.0

    def next(self) -> float:
        self._value += 1.0
        return self._value


@asynccontextmanager
async def _stateless_conversation() -> AsyncGenerator[Conversation]:
    """A fresh per-call stream — no shared history, no cross-call lock, no handle."""
    yield Conversation(stream=MemoryStream())


def _result(
    content: list[ContentBlock],
    *,
    handle: str | None,
    structured: dict[str, Any] | None = None,
    is_error: bool = False,
) -> CallToolResult:
    """The tool result, carrying the conversation handle for both of its readers.

    A text block, because the protocol puts recovery from an expired handle on
    the model and the model does not read protocol metadata; and ``_meta``, for
    clients threading it programmatically. ``structuredContent`` is deliberately
    left alone: on this tool it is the agent's response schema, advertised
    verbatim as ``outputSchema``, which MCP requires structured content to
    conform to — a server field mixed in would break the tool's own contract.
    """
    if handle is None:
        return CallToolResult(content=content, structuredContent=structured, isError=is_error)
    return CallToolResult(
        content=[*content, TextContent(type="text", text=_handle_text(handle))],
        structuredContent=structured,
        isError=is_error,
        _meta={CONVERSATION_META_KEY: handle},
    )


def _handle_text(handle: str) -> str:
    return f"Conversation handle: {handle}\nPass it back as the 'conversation' argument to continue this conversation."


def _principal() -> str | None:
    """Who this call is on behalf of, or ``None`` when no authentication is configured.

    The access token's subject, falling back to its client id, which is always
    present. With nothing to bind to, a conversation handle is the sole
    credential for the conversation it names.
    """
    token = get_access_token()
    if token is None:
        return None
    return token.subject or token.client_id


def _is_modern(request_context: "ServerRequestContext[Any, Any]") -> bool:
    """Whether this call arrived on a modern-era (2026-07-28) revision.

    The negotiated protocol version is a first-class field of the request
    context, so this reads the same over HTTP and over streams — the era is a
    protocol fact, not a transport detail. The membership test comes from the
    SDK's own registry so a new modern revision needs no change here.
    """
    return request_context.protocol_version in MODERN_PROTOCOL_VERSIONS


def _session_id(request_context: "ServerRequestContext[Any, Any]") -> str | None:
    """Extract the MCP session key for this call — handshake era only.

    Over streamable HTTP the transport's ``Request`` carries an ``mcp-session-id``
    header (present only when the transport runs stateful); over stdio there is no
    HTTP request, so all turns share one per-process session. The modern era
    issues no session id and forbids keying on the process, so callers must
    establish the era before consulting this.
    """
    request = getattr(request_context, "request", None)
    if request is None:
        return STDIO_SESSION
    headers = getattr(request, "headers", None)
    return headers.get("mcp-session-id") if headers is not None else None


def _build_inputs(message: str, context: str | None) -> list[TextInput]:
    inputs: list[TextInput] = []
    if context:
        inputs.append(TextInput(f"Context:\n{context}"))
    inputs.append(TextInput(message))
    return inputs
