# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""An MCP server's tool sharing a name with another tool of the agent, over a real server.

Lives apart from ``test_mcp.py`` for the reason ``test_mcp_live_transport.py`` does:
serving on a socket needs ``uvicorn``, and its skip guard would take the rest down.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest

pytest.importorskip("mcp")

from ag2 import Agent, tool
from ag2.events import ToolCallEvent
from ag2.mcp import MCPServer, mcp_tool
from ag2.middleware import approval_required
from ag2.testing import TestConfig
from ag2.tools import MCPToolkit
from test._helpers import ScriptedHuman
from test._serving import serving


@asynccontextmanager
async def serving_tool(name: str, runs: list[str]) -> AsyncGenerator[str]:
    """Serve an MCP server exposing one tool ``name``, yielding its URL.

    Each call the server executes appends ``name`` to ``runs``.
    """

    @mcp_tool(name=name)
    def remote() -> str:
        """A tool served over MCP."""
        runs.append(name)
        return "remote"

    served = MCPServer(Agent("served", config=TestConfig("unused")), tools=[remote], path="/mcp")
    async with serving(served) as base_url:
        yield f"{base_url}/mcp/"


@pytest.mark.asyncio
class TestMCPToolSharingAName:
    async def test_denied_local_tool_is_not_replaced_by_the_mcp_tool(self) -> None:
        runs: list[str] = []

        @tool(name="deploy", middleware=[approval_required()])
        def deploy() -> str:
            runs.append("local")
            return "deployed"

        async with serving_tool("deploy", runs) as url:
            agent = Agent(
                "",
                tools=[deploy, MCPToolkit(url)],
                config=TestConfig(ToolCallEvent(name="deploy", arguments="{}"), "done"),
            )

            await agent.ask("Deploy", hitl_hook=ScriptedHuman("n"))

        assert runs == []

    async def test_mcp_tool_is_dropped_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        runs: list[str] = []

        @tool(name="deploy")
        def deploy() -> str:
            runs.append("local")
            return "deployed"

        async with serving_tool("deploy", runs) as url:
            agent = Agent(
                "",
                tools=[MCPToolkit(url), deploy],
                config=TestConfig(ToolCallEvent(name="deploy", arguments="{}"), "done"),
            )

            with caplog.at_level("WARNING", logger="ag2.tools.precedence"):
                await agent.ask("Deploy")

        assert runs == ["local"]
        [warning] = caplog.records
        assert "deploy" in warning.getMessage()

    async def test_first_server_wins_a_shared_name(self) -> None:
        first: list[str] = []
        second: list[str] = []

        async with serving_tool("search", first) as first_url, serving_tool("search", second) as second_url:
            agent = Agent(
                "",
                tools=[MCPToolkit(first_url), MCPToolkit(second_url)],
                config=TestConfig(ToolCallEvent(name="search", arguments="{}"), "done"),
            )

            await agent.ask("Search")

        assert first == ["search"]
        assert second == []

    async def test_always_for_a_local_tool_does_not_approve_the_mcp_tool(self) -> None:
        runs: list[str] = []

        @tool(name="deploy", middleware=[approval_required()])
        def deploy() -> str:
            runs.append("local")
            return "deployed"

        async with serving_tool("deploy", runs) as url:
            agent = Agent(
                "",
                tools=[MCPToolkit(url, middleware=[approval_required()])],
                config=TestConfig(
                    ToolCallEvent(name="deploy", arguments="{}"),
                    "done",
                    ToolCallEvent(name="deploy", arguments="{}"),
                    "done",
                ),
            )
            human = ScriptedHuman("always")

            reply = await agent.ask("Deploy", tools=[deploy], hitl_hook=human)
            human.answer = "n"
            await reply.ask("Deploy again", hitl_hook=human)

        assert runs == ["local"]
        assert human.questions == 2

    async def test_always_through_a_shared_hook_does_not_approve_the_mcp_tool(self) -> None:
        runs: list[str] = []
        gate = approval_required()

        @tool(name="deploy", middleware=[gate])
        def deploy() -> str:
            runs.append("local")
            return "deployed"

        async with serving_tool("deploy", runs) as url:
            agent = Agent(
                "",
                tools=[MCPToolkit(url, middleware=[gate])],
                config=TestConfig(
                    ToolCallEvent(name="deploy", arguments="{}"),
                    "done",
                    ToolCallEvent(name="deploy", arguments="{}"),
                    "done",
                ),
            )
            human = ScriptedHuman("always")

            reply = await agent.ask("Deploy", tools=[deploy], hitl_hook=human)
            human.answer = "n"
            await reply.ask("Deploy again", hitl_hook=human)

        assert runs == ["local"]
        assert human.questions == 2
