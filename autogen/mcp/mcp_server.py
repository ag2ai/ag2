# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""MCP Server — exposes AG2 tools as discoverable MCP tools.

This module is the server-side counterpart to the existing MCP client
(``autogen.mcp.mcp_client``). It allows AG2 ``Tool`` objects to be
exposed as MCP tools that any MCP client can discover via ``tools/list``
and invoke via ``tools/call``.

Two transports are supported out of the box:

- **stdio**: for subprocess-based MCP clients
- **SSE** (Server-Sent Events): for HTTP-based MCP clients

Typical usage::

    from autogen.mcp.mcp_server import create_mcp_server
    from mcp.server.stdio import stdio_server
    import anyio

    server = create_mcp_server(my_tools, server_name="my-agent")

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    anyio.run(main)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..doc_utils import export_module
from ..import_utils import optional_import_block, require_optional_import
from ..tools import Tool, Toolkit

with optional_import_block():
    from mcp.server.lowlevel import Server  # type: ignore[no-any-unimported]
    from mcp.types import (  # type: ignore[no-any-unimported]
        CallToolResult,
        ListToolsResult,
        TextContent,
        Tool as MCPTool,
    )

__all__ = ["create_mcp_server"]

logger = logging.getLogger(__name__)


@export_module("autogen.mcp")
@require_optional_import("mcp", "mcp")
def create_mcp_server(
    tools: list[Tool] | Toolkit,
    *,
    server_name: str = "ag2-mcp-server",
    server_version: str | None = None,
) -> Server:
    """Create an MCP server that exposes AG2 tools as MCP tools.

    Each AG2 ``Tool`` is registered with the returned server under its
    original name, description and JSON parameter schema.  Tools can
    be synchronous or asynchronous — the dispatcher handles both.

    Args:
        tools: A list of AG2 ``Tool`` objects, or a ``Toolkit`` instance.
        server_name: A human-readable name for the MCP server (default
            ``"ag2-mcp-server"``).
        server_version: An optional semantic version string advertised
            to clients during initialization.

    Returns:
        A low-level ``mcp.server.Server`` instance, ready to be run with
        ``stdio_server``, ``sse_server``, or a custom transport.

    Example
    -------
    Run via stdio (the most common transport for subprocess-based MCP
    clients)::

        from autogen.mcp.mcp_server import create_mcp_server
        from mcp.server.stdio import stdio_server
        import anyio

        server = create_mcp_server(my_tools)

        async def main():
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                )

        anyio.run(main)

    Run via SSE (HTTP)::

        from autogen.mcp.mcp_server import create_mcp_server
        from mcp.server.sse import sse_server
        import anyio

        server = create_mcp_server(my_tools)

        async def main():
            async with sse_server("127.0.0.1", 8000) as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                )

        anyio.run(main)
    """
    # Normalise input to a name-indexed mapping.
    if isinstance(tools, Toolkit):
        ag2_tools: list[Tool] = list(tools.tools)
    else:
        ag2_tools = list(tools)

    tool_map: dict[str, Tool] = {t.name: t for t in ag2_tools}

    # ------------------------------------------------------------------
    # Handler for tools/list
    # ------------------------------------------------------------------
    async def handle_list_tools(
        ctx: Any,
        params: Any = None,  # noqa: ANN401
    ) -> ListToolsResult:
        mcp_tools: list[MCPTool] = []
        for ag2_tool in ag2_tools:
            # AG2's tool_schema returns the OpenAI function-calling format:
            #   {"type": "function", "function": {name, description, parameters}}
            schema = ag2_tool.tool_schema
            func_info = schema.get("function", {})
            params_schema: dict[str, Any] = func_info.get(
                "parameters", {"type": "object", "properties": {}}
            )
            mcp_tools.append(
                MCPTool(
                    name=ag2_tool.name,
                    description=ag2_tool.description or "",
                    input_schema=params_schema,
                )
            )
        return ListToolsResult(tools=mcp_tools)

    # ------------------------------------------------------------------
    # Handler for tools/call
    # ------------------------------------------------------------------
    async def handle_call_tool(
        ctx: Any,
        params: Any,  # noqa: ANN401
    ) -> CallToolResult:
        tool_name: str = params.name
        arguments: dict[str, Any] = params.arguments or {}

        ag2_tool = tool_map.get(tool_name)
        if ag2_tool is None:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {tool_name}")],
                is_error=True,
            )

        try:
            # Dispatch through the AG2 Tool — this handles both sync and
            # async underlying functions transparently.
            result = await _call_tool_function(ag2_tool, arguments)

            if isinstance(result, str):
                text: str = result
            else:
                text = json.dumps(result, ensure_ascii=False, default=str)

            return CallToolResult(content=[TextContent(type="text", text=text)])
        except Exception as e:
            logger.exception("Error calling tool '%s'", tool_name)
            return CallToolResult(
                content=[TextContent(type="text", text=str(e))],
                is_error=True,
            )

    return Server(
        name=server_name,
        version=server_version,
        on_list_tools=handle_list_tools,
        on_call_tool=handle_call_tool,
    )


async def _call_tool_function(tool: Tool, arguments: dict[str, Any]) -> Any:
    """Invoke an AG2 Tool, supporting both sync and async callables."""
    import inspect

    if inspect.iscoroutinefunction(tool.func):
        return await tool(**arguments)
    return tool(**arguments)
