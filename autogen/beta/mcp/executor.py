# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from mcp.types import CallToolResult, ContentBlock, TextContent
from mcp.types import Tool as MCPTool
from pydantic import ValidationError

from autogen.beta.agent import Agent
from autogen.beta.events import (
    BaseEvent,
    ModelMessageChunk,
    TextInput,
    ToolCallEvent,
    ToolResultEvent,
)
from autogen.beta.stream import MemoryStream

from .errors import MCPAgentConfigError
from .info import build_ask_tool
from .mappers import reply_to_content, to_structured_dict

if TYPE_CHECKING:
    from mcp.server.session import ServerSession
    from mcp.shared.context import RequestContext

# Return contract accepted by ``mcp``'s ``@server.call_tool()`` handler: bare
# content (unstructured), a ``(content, structured)`` tuple, or a fully-formed
# ``CallToolResult`` (used for error short-circuits, bypassing output validation).
CallToolReturn = list[ContentBlock] | tuple[list[ContentBlock], dict[str, Any]] | CallToolResult

_LOGGER_NAME = "ag2.mcp"


class AgentExecutor:
    """Bridge an MCP ``tools/call`` to a single :meth:`Agent.ask` turn.

    Each call is stateless: a fresh :class:`MemoryStream` is created per
    invocation (mirroring the A2A executor) so any server replica can handle any
    request. While the agent runs, its stream events are forwarded to the MCP
    client as progress / log notifications when ``stream_progress`` is enabled.
    """

    __slots__ = ("_agent", "_tool_name", "_tool_description", "_stream_progress")

    def __init__(
        self,
        agent: Agent,
        *,
        tool_name: str = "ask",
        tool_description: str | None = None,
        stream_progress: bool = True,
    ) -> None:
        self._agent = agent
        self._tool_name = tool_name
        self._tool_description = tool_description
        self._stream_progress = stream_progress

    def list_tools(self) -> list[MCPTool]:
        return [build_ask_tool(self._agent, tool_name=self._tool_name, tool_description=self._tool_description)]

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

        stream = MemoryStream()
        if self._stream_progress:
            self._wire_progress(stream, request_context)

        reply = await self._agent.ask(*_build_inputs(message, context), stream=stream)
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

    def _has_object_output(self) -> bool:
        schema = self._agent._response_schema
        json_schema = schema.json_schema if schema is not None else None
        return isinstance(json_schema, dict) and json_schema.get("type") == "object"

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


def _build_inputs(message: str, context: str | None) -> list[TextInput]:
    inputs: list[TextInput] = []
    if context:
        inputs.append(TextInput(f"Context:\n{context}"))
    inputs.append(TextInput(message))
    return inputs


def _error(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=True)
