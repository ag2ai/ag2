# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import pytest

from ag2 import Agent
from ag2.mcp import MCPFunctionTool, MCPServer
from ag2.mcp.testing import connect
from ag2.mcp.tools import ToolContext
from ag2.testing import TestConfig


class _SilentError(Exception):
    """An exception whose ``str()`` is empty — common for bare ``raise SomeError``."""


async def _raises_silently(_args: dict[str, Any], _ctx: ToolContext) -> Any:
    raise _SilentError


@pytest.mark.asyncio
class TestErrors:
    async def test_missing_message_argument(self) -> None:
        server = MCPServer(Agent("greeter", config=TestConfig("hi")))

        async with connect(server, raise_exceptions=False) as session:
            result = await session.call_tool("ask", {})

        assert result.is_error is True

    async def test_unknown_tool(self) -> None:
        server = MCPServer(Agent("greeter", config=TestConfig("hi")))

        async with connect(server, raise_exceptions=False) as session:
            result = await session.call_tool("nope", {"message": "hi"})

        assert result.is_error is True

    async def test_agent_without_config_surfaces_as_tool_error(self) -> None:
        server = MCPServer(Agent("no-config"))

        async with connect(server, raise_exceptions=False) as session:
            result = await session.call_tool("ask", {"message": "hi"})

        assert result.is_error is True


@pytest.mark.asyncio
class TestToolErrorsAreLegible:
    async def test_a_message_less_exception_still_names_itself(self) -> None:
        """``str(exc)`` is empty for a bare ``raise``, which would ship an empty
        text block; the class name is the least the client can act on.
        """
        server = MCPServer(
            Agent("g", config=TestConfig("hi")), tools=[MCPFunctionTool("boom", "Boom", _raises_silently)]
        )

        async with connect(server, raise_exceptions=False) as session:
            result = await session.call_tool("boom", {})

        assert result.is_error is True
        assert result.content[0].text == "_SilentError"

    async def test_the_traceback_is_logged_server_side(self, caplog: pytest.LogCaptureFixture) -> None:
        """The wire carries only the message, so without a log the stack is lost."""
        server = MCPServer(
            Agent("g", config=TestConfig("hi")), tools=[MCPFunctionTool("boom", "Boom", _raises_silently)]
        )

        with caplog.at_level(logging.ERROR, logger="ag2.mcp.server"):
            async with connect(server, raise_exceptions=False) as session:
                await session.call_tool("boom", {})

        assert any(r.exc_info is not None and "boom" in r.getMessage() for r in caplog.records)
