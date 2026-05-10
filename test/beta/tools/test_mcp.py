# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from contextlib import asynccontextmanager

import pytest
from dirty_equals import IsPartialDict

pytest.importorskip("mcp")
from mcp.types import CallToolResult, ListToolsResult, TextContent
from mcp.types import Tool as MCPTool

from autogen.beta import Agent, Context
from autogen.beta.events import ToolCallEvent
from autogen.beta.testing import TestConfig
from autogen.beta.tools import MCPServer, MCPStdioServerConfig
from autogen.beta.tools.toolkits.mcp_server import toolkit as _toolkit_module

MCPSessionPatch = Callable[[list[MCPTool], dict[str, CallToolResult] | None], "_FakeMCPSession"]


@pytest.fixture
def patch_mcp_session(monkeypatch: pytest.MonkeyPatch) -> MCPSessionPatch:
    """Replace ``_mcp_session`` with a fake that yields a controllable session."""

    def _install(
        tools: list[MCPTool],
        call_results: dict[str, CallToolResult] | None = None,
    ) -> _FakeMCPSession:
        session = _FakeMCPSession(tools, call_results)

        @asynccontextmanager
        async def fake(_):
            yield session

        monkeypatch.setattr(_toolkit_module, "_mcp_session", fake)
        return session

    return _install


@pytest.mark.asyncio
async def test_tool_registered_from_http_mcp_server(
    patch_mcp_session: MCPSessionPatch,
    context: Context,
) -> None:
    patch_mcp_session([
        MCPTool(
            name="test_tool_name",
            description="test_tool_description",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The message to send"},
                    "count": {"type": "integer", "description": "How many times"},
                },
                "required": ["message"],
            },
        )
    ])

    toolkit = MCPServer("https://mcp.example.com")
    [schema] = list(await toolkit.schemas(context))

    assert schema.function.name == "test_tool_name"
    assert schema.function.description == "test_tool_description"
    assert schema.function.parameters == IsPartialDict({
        "type": "object",
        "required": ["message"],
        "properties": IsPartialDict({
            "message": IsPartialDict({"type": "string"}),
            "count": IsPartialDict({"type": "integer"}),
        }),
    })


@pytest.mark.asyncio
async def test_tool_registered_from_stdio_mcp_server(
    patch_mcp_session: MCPSessionPatch,
    context: Context,
) -> None:
    patch_mcp_session([
        MCPTool(
            name="ping",
            description="returns pong",
            inputSchema={"type": "object", "properties": {}},
        )
    ])

    toolkit = MCPServer(
        MCPStdioServerConfig(
            command="some-mcp-binary",
            args=["--flag"],
        )
    )
    [schema] = list(await toolkit.schemas(context))

    assert schema.function.name == "ping"
    assert schema.function.description == "returns pong"


@pytest.mark.asyncio
async def test_allowed_and_blocked_tools_are_filtered(
    patch_mcp_session: MCPSessionPatch,
    context: Context,
) -> None:
    patch_mcp_session([
        MCPTool(name="keep", description="", inputSchema={"type": "object"}),
        MCPTool(name="drop_blocked", description="", inputSchema={"type": "object"}),
        MCPTool(name="drop_unlisted", description="", inputSchema={"type": "object"}),
    ])

    toolkit = MCPServer(
        MCPStdioServerConfig(
            command="x",
            allowed_tools=["keep", "drop_blocked"],
            blocked_tools=["drop_blocked"],
        )
    )
    schemas = list(await toolkit.schemas(context))

    assert [s.function.name for s in schemas] == ["keep"]


@pytest.mark.asyncio
async def test_mcp_tool_result_is_returned_to_agent(
    patch_mcp_session: MCPSessionPatch,
) -> None:
    session = patch_mcp_session(
        [MCPTool(name="echo", description="", inputSchema={"type": "object"})],
        call_results={
            "echo": CallToolResult(content=[TextContent(type="text", text="hello world")]),
        },
    )

    agent = Agent(
        name="test",
        tools=[MCPServer(MCPStdioServerConfig(command="x"))],
        config=TestConfig(
            ToolCallEvent(name="echo", arguments="{}"),
            "done",
        ),
    )
    result = await agent.ask("test")

    assert result.body == "done"
    assert session.calls == [("echo", {})]


class _FakeMCPSession:
    """In-memory stand-in for ``mcp.ClientSession`` used by the toolkit."""

    def __init__(
        self,
        tools: list[MCPTool],
        call_results: dict[str, CallToolResult] | None = None,
    ) -> None:
        self._tools = tools
        self._call_results = call_results or {}
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self) -> ListToolsResult:
        return ListToolsResult(tools=self._tools)

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        self.calls.append((name, arguments))
        return self._call_results.get(
            name,
            CallToolResult(content=[TextContent(type="text", text="ok")]),
        )
