# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from dirty_equals import IsPartialDict
from mcp.types import TextContent

from ag2 import Agent
from ag2.mcp import MCPServer, MCPUITool, mcp_tool
from ag2.mcp.errors import MCPToolNameConflictError
from ag2.mcp.testing import connect
from ag2.testing import TestConfig


async def _echo(args: dict[str, Any]) -> TextContent:
    return TextContent(type="text", text=f"got {args.get('x')}")


@pytest.mark.asyncio
class TestCustomTools:
    async def test_custom_tools_listed_next_to_ask(self) -> None:
        server = MCPServer(
            Agent("g", config=TestConfig("hi")),
            tools=[MCPUITool("echo", "Echo x", _echo, {"type": "object", "properties": {"x": {"type": "string"}}})],
        )

        async with connect(server) as session:
            tools = await session.list_tools()

        assert [t.name for t in tools.tools] == ["ask", "echo"]

    async def test_custom_tool_dispatches_to_handler(self) -> None:
        server = MCPServer(Agent("g", config=TestConfig("hi")), tools=[MCPUITool("echo", "Echo x", _echo)])

        async with connect(server) as session:
            result = await session.call_tool("echo", {"x": "42"})

        assert result.isError is False
        assert result.content == [TextContent(type="text", text="got 42")]

    async def test_ask_still_works_alongside_custom_tools(self) -> None:
        server = MCPServer(Agent("g", config=TestConfig("hello")), tools=[MCPUITool("echo", "Echo x", _echo)])

        async with connect(server) as session:
            result = await session.call_tool("ask", {"message": "hi"})

        assert result.content == [TextContent(type="text", text="hello")]

    async def test_sync_handler_runs(self) -> None:
        def sync_handler(args: dict[str, Any]) -> TextContent:
            return TextContent(type="text", text="sync ok")

        server = MCPServer(Agent("g", config=TestConfig("hi")), tools=[MCPUITool("s", "sync", sync_handler)])

        async with connect(server) as session:
            result = await session.call_tool("s", {})

        assert result.content == [TextContent(type="text", text="sync ok")]

    async def test_name_collision_with_ask_is_rejected(self) -> None:
        with pytest.raises(MCPToolNameConflictError):
            MCPServer(Agent("g", config=TestConfig("hi")), tools=[MCPUITool("ask", "clash", _echo)])

    async def test_collision_respects_custom_tool_name(self) -> None:
        # The reserved name follows ``tool_name``; "ask" is free when renamed.
        server = MCPServer(
            Agent("g", config=TestConfig("hi")),
            tool_name="chat",
            tools=[MCPUITool("ask", "now free", _echo)],
        )

        async with connect(server) as session:
            tools = await session.list_tools()

        assert [t.name for t in tools.tools] == ["chat", "ask"]


@pytest.mark.asyncio
class TestMcpToolDecorator:
    async def test_derives_name_description_and_schema(self) -> None:
        @mcp_tool
        async def greet(name: str = "world") -> TextContent:
            """Greet someone."""
            return TextContent(type="text", text=f"hi {name}")

        assert isinstance(greet, MCPUITool)
        assert greet.name == "greet"
        assert greet.description == "Greet someone."
        assert greet.input_schema == IsPartialDict({
            "type": "object",
            "properties": {"name": IsPartialDict({"type": "string", "default": "world"})},
        })

    async def test_dispatches_with_named_arguments(self) -> None:
        @mcp_tool
        async def add(good_id: str) -> TextContent:
            """Add an item."""
            return TextContent(type="text", text=f"added {good_id}")

        server = MCPServer(Agent("g", config=TestConfig("hi")), tools=[add])

        async with connect(server) as session:
            tools = await session.list_tools()
            result = await session.call_tool("add", {"good_id": "42"})

        assert tools.tools[1].inputSchema == IsPartialDict({"required": ["good_id"]})
        assert result.content == [TextContent(type="text", text="added 42")]

    async def test_default_argument_is_optional(self) -> None:
        @mcp_tool
        async def greet(name: str = "world") -> TextContent:
            """Greet."""
            return TextContent(type="text", text=f"hi {name}")

        server = MCPServer(Agent("g", config=TestConfig("hi")), tools=[greet])

        async with connect(server) as session:
            result = await session.call_tool("greet", {})

        assert result.content == [TextContent(type="text", text="hi world")]

    async def test_explicit_name_override(self) -> None:
        @mcp_tool(name="custom", description="overridden")
        async def greet() -> TextContent:
            """ignored docstring."""
            return TextContent(type="text", text="ok")

        assert greet.name == "custom"
        assert greet.description == "overridden"
