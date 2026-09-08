# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from mcp.types import TextContent
from mcp_types.version import LATEST_MODERN_VERSION

from ag2 import Agent
from ag2.mcp import MCPServer
from ag2.mcp.testing import connect_modern
from ag2.testing import TestConfig


@pytest.mark.asyncio
class TestE2EModern:
    """The served agent, driven over protocol revision 2026-07-28.

    The handshake-era suites reach the same server through ``connect``; these
    pin that the modern-era seam reaches it too, so the era's own semantics can
    be asserted rather than inferred.
    """

    async def test_negotiated_version_is_the_modern_revision(self) -> None:
        server = MCPServer(Agent("greeter", config=TestConfig("hi")))

        async with connect_modern(server) as session:
            negotiated = session.protocol_version

        assert negotiated == LATEST_MODERN_VERSION

    async def test_list_tools_exposes_ask(self) -> None:
        server = MCPServer(Agent("greeter", "Be nice.", config=TestConfig("hi")))

        async with connect_modern(server) as session:
            tools = await session.list_tools()

        assert [t.name for t in tools.tools] == ["ask"]

    async def test_call_tool_returns_reply(self) -> None:
        server = MCPServer(Agent("greeter", config=TestConfig("hello there!")))

        async with connect_modern(server) as session:
            result = await session.call_tool("ask", {"message": "hi"})

        assert result.is_error is False
        reply, _trailer = result.content
        assert reply == TextContent(type="text", text="hello there!")
