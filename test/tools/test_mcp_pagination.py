# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Discover a paginated third-party catalog through the real HTTP MCP transport."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("mcp")

from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams, TextContent
from mcp.types import Tool as MCPTool
from starlette.applications import Starlette
from starlette.routing import Mount

from ag2 import Agent, Context
from ag2.events import ToolCallEvent
from ag2.testing import TestConfig
from ag2.tools import MCPServerConfig, MCPToolkit
from ag2.tools.toolkits.mcp_server.types import ProtocolMode
from test._serving import serving


def page(*names: str, next_cursor: str | None = None) -> ListToolsResult:
    return ListToolsResult(
        tools=[MCPTool(name=name, inputSchema={"type": "object"}) for name in names],
        nextCursor=next_cursor,
    )


class Catalog:
    def __init__(self, pages: dict[str | None, ListToolsResult | Exception]) -> None:
        self.pages = pages
        self.cursors: list[str | None] = []
        self.calls: list[str] = []
        self.server = Server("paginated-tools", on_list_tools=self.list_tools, on_call_tool=self.call_tool)

    async def list_tools(
        self, ctx: ServerRequestContext[Any, Any], params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        cursor = params.cursor if params else None
        self.cursors.append(cursor)
        result = self.pages[cursor]
        if isinstance(result, Exception):
            raise result
        return result

    async def call_tool(self, ctx: ServerRequestContext[Any, Any], params: CallToolRequestParams) -> CallToolResult:
        self.calls.append(params.name)
        return CallToolResult(content=[TextContent(type="text", text=f"called {params.name}")])


@asynccontextmanager
async def serve_catalog(catalog: Catalog) -> AsyncGenerator[str]:
    manager = StreamableHTTPSessionManager(catalog.server, json_response=True)
    app = Starlette(routes=[Mount("/", app=manager.asgi_app)])
    async with manager.run(), serving(app) as url:
        yield f"{url}/"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["legacy", "auto"])
@pytest.mark.parametrize("empty_first_page", [False, True])
async def test_discovers_all_pages_and_caches_the_complete_catalog(
    context: Context, mode: ProtocolMode, empty_first_page: bool
) -> None:
    names = [] if empty_first_page else ["first"]
    catalog = Catalog({
        None: page(*names, next_cursor="opaque:/page+two="),
        "opaque:/page+two=": page(next_cursor=""),
        "": page("last"),
    })
    async with serve_catalog(catalog) as url:
        toolkit = MCPToolkit(MCPServerConfig(server_url=url, protocol_mode=mode))
        first, concurrent = await asyncio.gather(toolkit.schemas(context), toolkit.schemas(context))
        cached = await toolkit.schemas(context)

    assert [[s.function.name for s in schemas] for schemas in (first, concurrent, cached)] == [names + ["last"]] * 3
    assert catalog.cursors == [None, "opaque:/page+two=", ""]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["legacy", "auto"])
async def test_filters_and_calls_a_tool_from_a_later_page(context: Context, mode: ProtocolMode) -> None:
    catalog = Catalog({
        None: page("unlisted", next_cursor="next"),
        "next": page("blocked", "allowed"),
    })
    async with serve_catalog(catalog) as url:
        toolkit = MCPToolkit(
            MCPServerConfig(
                server_url=url,
                protocol_mode=mode,
                allowed_tools=["blocked", "allowed"],
                blocked_tools=["blocked"],
                tool_name_prefix="remote_",
            )
        )
        schemas = await toolkit.schemas(context)
        agent = Agent(
            "caller",
            tools=[toolkit],
            config=TestConfig(ToolCallEvent(name="remote_allowed", arguments="{}"), "done"),
        )
        reply = await agent.ask("Call the allowed tool")

    assert reply.body == "done"
    assert [schema.function.name for schema in schemas] == ["remote_allowed"]
    assert catalog.calls == ["allowed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["legacy", "auto"])
async def test_failed_later_page_leaves_discovery_retryable(context: Context, mode: ProtocolMode) -> None:
    catalog = Catalog({
        None: page("stale", next_cursor="next"),
        "next": MCPError(code=-32603, message="next page unavailable"),
    })
    async with serve_catalog(catalog) as url:
        toolkit = MCPToolkit(MCPServerConfig(server_url=url, protocol_mode=mode))
        with pytest.RaisesGroup(pytest.RaisesExc(MCPError, match="next page unavailable"), flatten_subgroups=True):
            await toolkit.schemas(context)
        assert list(toolkit.tools) == []

        catalog.pages = {None: page("fresh", next_cursor="next"), "next": page("last")}
        schemas = await toolkit.schemas(context)

    assert [schema.function.name for schema in schemas] == ["fresh", "last"]
    assert catalog.cursors == [None, "next", None, "next"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["legacy", "auto"])
@pytest.mark.parametrize("cycle", [False, True])
async def test_repeated_cursor_does_not_loop(context: Context, mode: ProtocolMode, cycle: bool) -> None:
    catalog = Catalog({
        None: page("first", next_cursor="a"),
        "a": page("second", next_cursor="b" if cycle else "a"),
        "b": page("third", next_cursor="a"),
    })
    async with serve_catalog(catalog) as url:
        toolkit = MCPToolkit(MCPServerConfig(server_url=url, protocol_mode=mode))
        with pytest.RaisesGroup(
            pytest.RaisesExc(RuntimeError, match="repeated pagination cursor"), flatten_subgroups=True
        ):
            await toolkit.schemas(context)
        assert list(toolkit.tools) == []

    assert catalog.cursors == ([None, "a", "b"] if cycle else [None, "a"])
