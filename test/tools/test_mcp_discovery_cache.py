# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Reuse of discovered MCP catalogs across a toolkit's recently used configurations.

A shared toolkit whose configuration resolves differently per turn — an
``allowed_tools=Variable("allowed")`` allowlist is the motivating case — should
not repeat the handshake and ``tools/list`` every time it returns to a
configuration it has already discovered. These tests count the discovery
requests a real HTTP MCP server receives, since that is the cost being removed.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import partial
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

from ag2 import Context, Variable
from ag2.tools import MCPServerConfig, MCPToolkit
from test._serving import serving


def _catalog(names: list[str]) -> ListToolsResult:
    return ListToolsResult(tools=[MCPTool(name=name, inputSchema={"type": "object"}) for name in names])


class CountingCatalog:
    """An MCP server exposing a fixed catalog, counting how often it is listed.

    A single page per call, so one ``list_tools`` reaches the server per discovery.
    """

    def __init__(self, names: list[str]) -> None:
        self.names = list(names)
        self.list_calls = 0
        self.server = Server("catalog", on_list_tools=self.list_tools, on_call_tool=self.call_tool)

    async def list_tools(
        self, ctx: ServerRequestContext[Any, Any], params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        self.list_calls += 1
        return _catalog(self.names)

    async def call_tool(self, ctx: ServerRequestContext[Any, Any], params: CallToolRequestParams) -> CallToolResult:
        return CallToolResult(content=[TextContent(type="text", text=f"called {params.name}")])


class OnceFailingCatalog(CountingCatalog):
    """A catalog whose first ``list_tools`` fails, then succeeds."""

    def __init__(self, names: list[str]) -> None:
        super().__init__(names)
        self.fail_next = True

    async def list_tools(
        self, ctx: ServerRequestContext[Any, Any], params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        self.list_calls += 1
        if self.fail_next:
            self.fail_next = False
            raise MCPError(code=-32603, message="catalog unavailable")
        return _catalog(self.names)


@asynccontextmanager
async def serve(catalog: CountingCatalog) -> AsyncGenerator[str]:
    manager = StreamableHTTPSessionManager(catalog.server, json_response=True)
    app = Starlette(routes=[Mount("/", app=manager.asgi_app)])
    async with manager.run(), serving(app) as url:
        yield f"{url}/"


def _allowlist(values: list[list[str]]) -> Variable:
    """A variable yielding the next allowlist on each resolution."""
    return Variable("allowed", default_factory=partial(next, iter(values), values[-1]))


@pytest.mark.asyncio
async def test_alternating_configurations_reuse_each_retained_catalog(context: Context) -> None:
    catalog = CountingCatalog(["echo", "reverse"])
    alternating = [["echo"], ["reverse"]] * 6

    async with serve(catalog) as url:
        toolkit = MCPToolkit(MCPServerConfig(server_url=url, allowed_tools=_allowlist(alternating)))
        exposed = [[schema.function.name for schema in await toolkit.schemas(context)] for _ in alternating]

    assert exposed == alternating
    # Two catalogs discovered across twelve turns, not one discovery per turn.
    assert catalog.list_calls == 2


@pytest.mark.asyncio
async def test_configuration_beyond_the_retained_bound_is_discovered_again(context: Context) -> None:
    names = [f"tool{index}" for index in range(9)]
    catalog = CountingCatalog(names)
    # The allowlist is driven turn by turn, so an earlier configuration can be
    # returned to once it is no longer among those retained.
    allow = ["tool0"]
    config = MCPServerConfig(server_url="", allowed_tools=Variable("allowed", default_factory=lambda: allow))

    async with serve(catalog) as url:
        config.server_url = url
        toolkit = MCPToolkit(config)
        first_pass = []
        for name in names:
            allow[:] = [name]
            first_pass.append([schema.function.name for schema in await toolkit.schemas(context)])
        allow[:] = [names[0]]
        revisited = [schema.function.name for schema in await toolkit.schemas(context)]

    assert first_pass == [[name] for name in names]
    assert revisited == [names[0]]
    # Nine distinct configurations plus the rediscovery of the evicted first one.
    assert catalog.list_calls == 10


@pytest.mark.asyncio
async def test_failed_discovery_is_not_retained(context: Context) -> None:
    catalog = OnceFailingCatalog(["echo"])

    async with serve(catalog) as url:
        toolkit = MCPToolkit(MCPServerConfig(server_url=url, allowed_tools=_allowlist([["echo"], ["echo"]])))
        with pytest.RaisesGroup(pytest.RaisesExc(MCPError, match="catalog unavailable"), flatten_subgroups=True):
            await toolkit.schemas(context)

        # A failure leaves nothing to reuse, so the next turn reaches the server again.
        exposed = [schema.function.name for schema in await toolkit.schemas(context)]

    assert exposed == ["echo"]
    assert catalog.list_calls == 2
