# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A stdio MCP server whose tool catalog depends on its launch configuration."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams, TextContent, Tool


class Catalog:
    def __init__(self, name: str) -> None:
        self.name = name

    async def list_tools(
        self, context: ServerRequestContext[Any, Any], params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        return ListToolsResult(tools=[Tool(name=self.name, inputSchema={"type": "object", "properties": {}})])

    async def call_tool(self, context: ServerRequestContext[Any, Any], params: CallToolRequestParams) -> CallToolResult:
        if params.name != self.name:
            return CallToolResult(isError=True, content=[TextContent(type="text", text="Unknown tool")])
        return CallToolResult(content=[TextContent(type="text", text=f"called {self.name}")])


async def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("AG2_MCP_TEST_TOOL", Path.cwd().name)
    catalog = Catalog(name)
    server = Server("context-discovery", on_list_tools=catalog.list_tools, on_call_tool=catalog.call_tool)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
