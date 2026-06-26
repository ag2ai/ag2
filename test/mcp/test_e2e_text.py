# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Agent
from ag2.events import ModelRequest, TextInput
from ag2.mcp import MCPServer
from ag2.mcp.testing import connect
from ag2.testing import TestConfig, TrackingConfig


@pytest.mark.asyncio
class TestE2EText:
    async def test_list_tools_exposes_ask(self) -> None:
        server = MCPServer(Agent("greeter", "Be nice.", config=TestConfig("hi")))

        async with connect(server) as session:
            tools = await session.list_tools()

        assert [t.name for t in tools.tools] == ["ask"]

    async def test_call_tool_returns_reply(self) -> None:
        server = MCPServer(Agent("greeter", config=TestConfig("hello there!")))

        async with connect(server) as session:
            result = await session.call_tool("ask", {"message": "hi"})

        assert result.isError is False
        assert [c.text for c in result.content if c.type == "text"] == ["hello there!"]

    async def test_context_is_prepended(self) -> None:
        tracking = TrackingConfig(TestConfig("ok"))
        server = MCPServer(Agent("echo", config=tracking))

        async with connect(server) as session:
            result = await session.call_tool("ask", {"message": "do it", "context": "be brief"})

        assert result.isError is False
        # The model receives the context input prepended before the message input.
        tracking.mock.assert_called_with(ModelRequest([TextInput("Context:\nbe brief"), TextInput("do it")]))

    async def test_custom_tool_name(self) -> None:
        server = MCPServer(Agent("greeter", config=TestConfig("yo")), tool_name="chat")

        async with connect(server) as session:
            tools = await session.list_tools()
            result = await session.call_tool("chat", {"message": "hi"})

        assert [t.name for t in tools.tools] == ["chat"]
        assert [c.text for c in result.content if c.type == "text"] == ["yo"]
