# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Agent
from ag2.mcp import MCPServer
from ag2.mcp.testing import connect
from ag2.testing import TestConfig


@pytest.mark.asyncio
class TestErrors:
    async def test_missing_message_argument(self) -> None:
        server = MCPServer(Agent("greeter", config=TestConfig("hi")))

        async with connect(server, raise_exceptions=False) as session:
            result = await session.call_tool("ask", {})

        assert result.isError is True

    async def test_unknown_tool(self) -> None:
        server = MCPServer(Agent("greeter", config=TestConfig("hi")))

        async with connect(server, raise_exceptions=False) as session:
            result = await session.call_tool("nope", {"message": "hi"})

        assert result.isError is True

    async def test_agent_without_config_surfaces_as_tool_error(self) -> None:
        server = MCPServer(Agent("no-config"))

        async with connect(server, raise_exceptions=False) as session:
            result = await session.call_tool("ask", {"message": "hi"})

        assert result.isError is True
