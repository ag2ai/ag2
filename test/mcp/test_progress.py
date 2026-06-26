# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Agent
from ag2.events import ToolCallEvent
from ag2.mcp import MCPServer
from ag2.mcp.testing import connect
from ag2.testing import TestConfig

from ._helpers import ChunkConfig, make_agent


@pytest.mark.asyncio
class TestProgress:
    async def test_chunks_forwarded_as_progress(self) -> None:
        agent = make_agent(config=ChunkConfig("Hello, ", "world!", final="Hello, world!"))
        server = MCPServer(agent)

        updates: list[tuple[float, float | None, str | None]] = []

        async def on_progress(progress: float, total: float | None, message: str | None) -> None:
            updates.append((progress, total, message))

        async with connect(server) as session:
            result = await session.call_tool("ask", {"message": "hi"}, progress_callback=on_progress)

        assert result.isError is False
        # One progress notification per streamed chunk, monotonically increasing.
        assert [m for _, _, m in updates] == ["Hello, ", "world!"]
        assert [p for p, _, _ in updates] == [1.0, 2.0]
        # Final body is still returned in full.
        assert [c.text for c in result.content if c.type == "text"] == ["Hello, world!"]

    async def test_no_progress_without_token(self) -> None:
        agent = make_agent(config=ChunkConfig("a", "b"))
        server = MCPServer(agent)

        # No progress_callback => no progressToken => the call still succeeds.
        async with connect(server) as session:
            result = await session.call_tool("ask", {"message": "hi"})

        assert result.isError is False

    async def test_progress_disabled(self) -> None:
        agent = make_agent(config=ChunkConfig("a", "b"))
        server = MCPServer(agent, stream_progress=False)

        updates: list[str | None] = []

        async def on_progress(progress: float, total: float | None, message: str | None) -> None:
            updates.append(message)

        async with connect(server) as session:
            result = await session.call_tool("ask", {"message": "hi"}, progress_callback=on_progress)

        assert result.isError is False
        assert updates == []

    async def test_tool_events_are_logged(self) -> None:
        # An agent that makes a tool call emits ToolCall/ToolResult events, which
        # the progress forwarder reports as MCP log notifications.
        agent = Agent("tooler", config=TestConfig(ToolCallEvent(name="ping", arguments="{}"), "done"))

        @agent.tool
        def ping() -> str:
            return "pong"

        async with connect(MCPServer(agent)) as session:
            result = await session.call_tool("ask", {"message": "go"})

        assert result.isError is False
        assert [c.text for c in result.content if c.type == "text"] == ["done"]
