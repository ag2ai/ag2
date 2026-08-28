# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.types import CallToolResult, ContentBlock, TextContent
from mcp.types import Tool as MCPTool
from mcp_types.version import MODERN_PROTOCOL_VERSIONS
from pydantic import ValidationError

from ag2.agent import Agent
from ag2.events import (
    BaseEvent,
    ModelMessageChunk,
    TextInput,
    ToolCallEvent,
    ToolResultEvent,
)
from ag2.stream import MemoryStream

from .errors import MCPAgentConfigError, UnknownConversationError
from .info import build_ask_tool, object_output_schema
from .mappers import reply_to_content, to_structured_dict, tool_error
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

    __slots__ = ("_agent", "_tool_name", "_tool_description", "_stream_progress", "_context_provider", "_session_store")

    def __init__(
        self,
        agent: Agent,
        *,
        tool_name: str = "ask",
        tool_description: str | None = None,
        stream_progress: bool = True,
        context_provider: "ContextProvider | None" = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self._agent = agent
        self._tool_name = tool_name
        self._tool_description = tool_description
        self._stream_progress = stream_progress
        self._context_provider = context_provider
        self._session_store = session_store

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
        request_context: "ServerRequestContext[Any, Any]",
    ) -> CallToolResult:
        if name != self._tool_name:
            return tool_error(f"Unknown tool: {name!r}.")
        if self._agent.config is None:
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
        try:
            async with self._conversation_cm(request_context, conversation) as convo:
                return await self._turn(convo, message, context, request_context)
        except UnknownConversationError as e:
            # A tool-level error, never a JSON-RPC one: the protocol draws that
            # line so the model can start a new conversation rather than fail the
            # turn. Only resolving the conversation raises this.
            return tool_error(str(e))

    async def _turn(
        self,
        convo: Conversation,
        message: str,
        context: str | None,
        request_context: "ServerRequestContext[Any, Any]",
    ) -> CallToolResult:
        """Run one agent turn inside ``convo`` and shape its reply into a result."""
        if self._stream_progress:
            self._wire_progress(convo.stream, request_context)

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

    def _has_object_output(self) -> bool:
        return object_output_schema(self._agent._response_schema) is not None

    def _wire_progress(
        self,
        stream: MemoryStream,
        request_context: "ServerRequestContext[Any, Any]",
    ) -> None:
        # ``_meta`` is an open mapping in ``mcp`` 2.0, not a model.
        token = request_context.meta.get("progress_token") if request_context.meta else None
        session = request_context.session
        progress = _Counter()

        @stream.subscribe
        async def forward(event: BaseEvent) -> None:
            if isinstance(event, ModelMessageChunk):
                if token is not None:
                    await session.send_progress_notification(token, progress.next(), message=event.content)
                return
            if isinstance(event, ToolResultEvent):
                await session.send_log_message("info", f"tool result: {event.name}", logger=_LOGGER_NAME)
                return
            if isinstance(event, ToolCallEvent):
                await session.send_log_message("info", f"tool call: {event.name}", logger=_LOGGER_NAME)


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
