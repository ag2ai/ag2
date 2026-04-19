# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import replace
from typing import Any

from autogen.beta.annotations import Context, Variable
from autogen.beta.events.tool_events import (
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
)
from autogen.beta.middleware import BaseMiddleware, ToolExecution, ToolMiddleware, ToolResultType
from autogen.beta.tools.final import Toolkit
from autogen.beta.tools.final.function_tool import FunctionDefinition, FunctionToolSchema
from autogen.beta.tools.tool import Tool

from .connection import MCPConnection
from .types import MCPServerConfig, RawMCPTool


class _MCPProxyTool(Tool):
    """A function-tool-shaped proxy that forwards calls to a remote MCP server."""

    __slots__ = ("name", "schema", "_connection", "_middleware")

    def __init__(
        self,
        connection: MCPConnection,
        raw_tool: RawMCPTool,
        middleware: tuple[ToolMiddleware, ...] = (),
    ) -> None:
        self._connection = connection
        self._middleware = middleware
        self.name = raw_tool["name"]
        self.schema = FunctionToolSchema(
            function=FunctionDefinition(
                name=self.name,
                description=raw_tool.get("description", ""),
                parameters=dict(raw_tool.get("inputSchema") or {}),
            )
        )

    async def schemas(self, context: "Context") -> list[FunctionToolSchema]:
        return [self.schema]

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        execution: ToolExecution = self
        for hook in reversed(self._middleware):
            execution = _wrap_middleware(hook, execution)
        for mw in middleware:
            execution = _wrap_middleware(mw.on_tool_execution, execution)

        async def execute(event: "ToolCallEvent", context: "Context") -> None:
            result = await execution(event, context)
            await context.send(result)

        stack.enter_context(context.stream.where(ToolCallEvent.name == self.name).sub_scope(execute))

    async def __call__(self, event: "ToolCallEvent", context: "Context") -> "ToolResultEvent | ToolErrorEvent":
        try:
            response = await self._connection.call_tool(self.name, event.serialized_arguments)
        except Exception as e:
            return ToolErrorEvent.from_call(event, error=e)

        if err := response.get("error"):
            message = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return ToolErrorEvent.from_call(event, error=RuntimeError(message))

        result = response.get("result", {})
        if result.get("isError"):
            return ToolErrorEvent.from_call(event, error=RuntimeError(_extract_content(result)))

        return ToolResultEvent.from_call(event, result=_extract_content(result))


class MCPServer(Toolkit):
    """Expose the tools of a remote MCP server as ordinary local tools.

    Tool discovery is lazy: the first call to :meth:`schemas` performs the
    MCP handshake, lists the server's tools, and registers a proxy for each
    one. The agent never sees that these are MCP tools — they look and behave
    like ordinary :class:`FunctionTool` instances.
    """

    __slots__ = ("config", "_connection", "_discovered")

    def __init__(
        self,
        server: str | MCPServerConfig,
        *,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        if isinstance(server, str):
            server = MCPServerConfig(server_url=server)
        self.config = server
        self._connection: MCPConnection | None = None
        self._discovered = False

        label = server.server_label if isinstance(server.server_label, str) else ""
        super().__init__(
            name=label or "mcp_toolkit",
            middleware=middleware,
        )

    async def schemas(self, context: "Context") -> Iterable[FunctionToolSchema]:
        await self._discover_tools(context)
        return await super().schemas(context)

    async def _discover_tools(self, context: "Context") -> None:
        if self._discovered:
            return

        resolved = _resolve_config(self.config, context)
        self._connection = MCPConnection(config=resolved)

        raw_tools = await self._connection.get_tools()
        allowed = resolved.allowed_tools
        blocked = set(resolved.blocked_tools or [])

        for raw in raw_tools:
            name = raw["name"]
            if allowed is not None and name not in allowed:
                continue
            if name in blocked:
                continue
            self.tools.append(
                _MCPProxyTool(
                    connection=self._connection,
                    raw_tool=raw,
                    middleware=self._middleware,
                )
            )

        self._discovered = True


def _wrap_middleware(hook: "ToolMiddleware", inner: "ToolExecution") -> "ToolExecution":
    async def call(event: "ToolCallEvent", context: "Context") -> "ToolResultType":
        return await hook(inner, event, context)

    return call


def _extract_content(result: dict[str, Any]) -> str:
    """Flatten an MCP ``tools/call`` result into a string for the model."""
    parts = result.get("content")
    if not parts:
        return json.dumps(result)

    chunks: list[str] = []
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text":
            chunks.append(p.get("text", ""))
        else:
            chunks.append(json.dumps(p))
    return "\n".join(chunks)


def _resolve_value(value: Any, context: "Context") -> Any:
    if not isinstance(value, Variable):
        return value
    name = value.name
    if name in context.variables:
        return context.variables[name]
    if value.default is not Ellipsis:
        return value.default
    if value.default_factory is not Ellipsis:
        return value.default_factory()
    raise KeyError(f"Context variable {name!r} not found and no default provided")


def _resolve_config(config: MCPServerConfig, context: "Context") -> MCPServerConfig:
    headers = dict(_resolve_value(config.headers, context) or {})
    auth = _resolve_value(config.authorization_token, context)
    if auth and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {auth}"

    return replace(
        config,
        server_url=_resolve_value(config.server_url, context),
        server_label=_resolve_value(config.server_label, context) or "",
        authorization_token=auth,
        description=_resolve_value(config.description, context),
        allowed_tools=_resolve_value(config.allowed_tools, context),
        blocked_tools=_resolve_value(config.blocked_tools, context),
        headers=headers or None,
    )
