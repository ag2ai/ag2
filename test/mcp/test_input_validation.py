# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from mcp.types import TextContent

from ag2 import Agent
from ag2.mcp import MCPFunctionTool, MCPServer, mcp_tool
from ag2.mcp.testing import connect
from ag2.mcp.tools import ToolContext
from ag2.testing import TestConfig

_SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
}


async def _echo_n(args: dict[str, Any], _ctx: ToolContext) -> TextContent:
    n = args["n"]
    return TextContent(type="text", text=f"{n!r} {type(n).__name__}")


def _server(*tools: MCPFunctionTool) -> MCPServer:
    return MCPServer(Agent("g", config=TestConfig("hi")), tools=list(tools))


def _text(result: Any) -> str:
    return result.content[0].text


@pytest.mark.asyncio
class TestDeclaredSchemaIsEnforced:
    """``mcp`` 1.x's ``@server.call_tool()`` validated arguments against the
    advertised ``inputSchema``; 2.0 dropped the mechanism, so the server does it.
    """

    async def test_wrong_type_is_a_tool_error(self) -> None:
        server = _server(MCPFunctionTool("echo", "Echo n", _echo_n, _SCHEMA))

        async with connect(server) as session:
            result = await session.call_tool("echo", {"n": "5"})

        assert result.is_error is True
        assert _text(result).startswith("Input validation error:")
        assert "'5' is not of type 'integer'" in _text(result)

    async def test_missing_required_argument_is_a_tool_error(self) -> None:
        server = _server(MCPFunctionTool("echo", "Echo n", _echo_n, _SCHEMA))

        async with connect(server) as session:
            result = await session.call_tool("echo", {})

        assert result.is_error is True
        assert _text(result) == "Input validation error: 'n' is a required property"

    async def test_valid_arguments_still_reach_the_handler(self) -> None:
        server = _server(MCPFunctionTool("echo", "Echo n", _echo_n, _SCHEMA))

        async with connect(server) as session:
            result = await session.call_tool("echo", {"n": 5})

        assert result.is_error is False
        assert _text(result) == "5 int"

    async def test_schemaless_tool_accepts_anything(self) -> None:
        """The default schema is a bare object, so it constrains nothing."""

        async def _any(args: dict[str, Any], _ctx: ToolContext) -> TextContent:
            return TextContent(type="text", text=f"got {sorted(args)}")

        server = _server(MCPFunctionTool("any", "Anything", _any))

        async with connect(server) as session:
            result = await session.call_tool("any", {"whatever": 1})

        assert result.is_error is False
        assert _text(result) == "got ['whatever']"

    async def test_a_malformed_schema_stays_a_tool_error(self) -> None:
        """Validating against an invalid schema raises ``SchemaError``, which must
        not escape as a protocol error — 1.x validated inside the decorator that
        converted anything raised into a tool-level error.
        """
        bad = {"type": "object", "properties": {"n": {"type": "nonsense"}}}
        server = _server(MCPFunctionTool("echo", "Echo n", _echo_n, bad))

        async with connect(server, raise_exceptions=False) as session:
            result = await session.call_tool("echo", {"n": 1})

        assert result.is_error is True


@pytest.mark.asyncio
class TestTypedToolErrorsStayClean:
    async def test_missing_argument_does_not_leak_the_request_context(self) -> None:
        """The pydantic layer renders the whole argument dict — including the
        injected context — into its message, so validation must run before it.
        """

        @mcp_tool
        def add(n: int) -> str:
            """Add."""
            return str(n)

        server = _server(add)

        async with connect(server) as session:
            result = await session.call_tool("add", {})

        assert result.is_error is True
        assert _text(result) == "Input validation error: 'n' is a required property"
        assert "__ctx__" not in _text(result)
        assert "ServerRequest" not in _text(result)


@pytest.mark.asyncio
class TestAgentToolIsValidatedToo:
    async def test_ask_rejects_a_non_string_message(self) -> None:
        server = _server()

        async with connect(server) as session:
            result = await session.call_tool("ask", {"message": 7})

        assert result.is_error is True
        assert _text(result).startswith("Input validation error:")
