# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.types import CallToolResult, ContentBlock, TextContent
from mcp.types import Tool as MCPTool
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

from .errors import MCPAgentConfigError
from .info import build_ask_tool, object_output_schema
from .mappers import reply_to_content, to_structured_dict
from .sessions import STDIO_SESSION, SessionStore

if TYPE_CHECKING:
    from mcp.server.session import ServerSession
    from mcp.shared.context import RequestContext

# Return contract accepted by ``mcp``'s ``@server.call_tool()`` handler: bare
# content (unstructured), a ``(content, structured)`` tuple, or a fully-formed
# ``CallToolResult`` (used for error short-circuits, bypassing output validation).
CallToolReturn = list[ContentBlock] | tuple[list[ContentBlock], dict[str, Any]] | CallToolResult

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
    so any server replica can handle any request. With a ``session_store``,
    history is keyed by the transport's ``mcp-session-id`` (or a per-process
    sentinel over stdio) and accumulates across calls. While the agent runs, its
    stream events are forwarded to the MCP client as progress / log notifications
    when ``stream_progress`` is enabled.
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
            )
        ]

    async def call(
        self,
        name: str,
        *,
        message: str,
        context: str | None = None,
        request_context: "RequestContext[ServerSession, Any, Any]",
    ) -> CallToolReturn:
        if name != self._tool_name:
            return _error(f"Unknown tool: {name!r}.")
        if self._agent.config is None:
            raise MCPAgentConfigError(self._agent.name)
        if not message:
            return _error("Missing required 'message' argument.")

        # The stream is held for the whole turn: for a keyed session that means
        # holding its turn lock, serializing concurrent same-session calls.
        async with self._stream_cm(request_context) as stream:
            if self._stream_progress:
                self._wire_progress(stream, request_context)

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

            reply = await self._agent.ask(*_build_inputs(message, context), stream=stream, **ask_kwargs)
            content = reply_to_content(reply)

            if not self._has_object_output():
                return content

            try:
                validated = await reply.content()
            except ValidationError as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Structured-output validation failed: {e}")],
                    isError=True,
                )
            structured = to_structured_dict(validated)
            if structured is None:
                return CallToolResult(
                    content=content,
                    isError=True,
                )
            return content, structured

    def _stream_cm(
        self, request_context: "RequestContext[ServerSession, Any, Any]"
    ) -> AbstractAsyncContextManager[MemoryStream]:
        """The stream context for this call: a keyed session (continuing history,
        turn-locked) or a fresh stateless stream."""
        if self._session_store is not None:
            session_id = _session_id(request_context)
            if session_id is not None:
                return self._session_store.session(session_id)
        # No store, or no server-issued session id (stateless HTTP) — stay stateless.
        return _stateless_stream()

    def _has_object_output(self) -> bool:
        return object_output_schema(self._agent._response_schema) is not None

    def _wire_progress(
        self,
        stream: MemoryStream,
        request_context: "RequestContext[ServerSession, Any, Any]",
    ) -> None:
        token = request_context.meta.progressToken if request_context.meta else None
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
async def _stateless_stream() -> AsyncIterator[MemoryStream]:
    """A fresh per-call stream — no shared history, no cross-call lock."""
    yield MemoryStream()


def _session_id(request_context: "RequestContext[ServerSession, Any, Any]") -> str | None:
    """Extract the session key for this call.

    Over streamable HTTP the transport's ``Request`` carries an ``mcp-session-id``
    header (present only when the transport runs stateful); over stdio there is no
    HTTP request, so all turns share one per-process session.
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


def _error(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=True)
