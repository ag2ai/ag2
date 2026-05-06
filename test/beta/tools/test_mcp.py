# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent
from mcp.types import Tool as MCPTool

from autogen.beta import Agent, Context
from autogen.beta.testing import TestConfig
from autogen.beta.tools import MCPServer, MCPStdioServerConfig
from autogen.beta.tools.toolkits.mcp_server import toolkit as _toolkit_module


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


@pytest.fixture
def patch_mcp_session(monkeypatch):
    """Replace ``_mcp_session`` with a fake that yields a controllable session."""

    seen_configs: list = []

    def _install(
        tools: list[MCPTool],
        call_results: dict[str, CallToolResult] | None = None,
    ) -> tuple[_FakeMCPSession, list]:
        session = _FakeMCPSession(tools, call_results)

        @asynccontextmanager
        async def fake(config):
            seen_configs.append(config)
            yield session

        monkeypatch.setattr(_toolkit_module, "_mcp_session", fake)
        return session, seen_configs

    return _install


@pytest.mark.asyncio()
async def test_tool_registered_from_http_mcp_server(patch_mcp_session, context: Context):
    tool_name = "test_tool_name"
    tool_description = "test_tool_description"

    patch_mcp_session([
        MCPTool(
            name=tool_name,
            description=tool_description,
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

    agent = Agent(
        name="test",
        tools=[MCPServer("https://mcp.example.com")],
        config=TestConfig('{"data": 42}'),
    )
    await agent.ask("test")

    assert len(agent.tools) == 1
    schemas = await agent.tools[0].schemas(context)
    assert schemas[0].function.name == tool_name
    assert schemas[0].function.description == tool_description

    params = schemas[0].function.parameters
    assert params["type"] == "object"
    assert "message" in params["properties"]
    assert params["properties"]["message"]["type"] == "string"
    assert "count" in params["properties"]
    assert params["properties"]["count"]["type"] == "integer"
    assert "message" in params["required"]
    assert "count" not in params["required"]


@pytest.mark.asyncio()
async def test_tool_registered_from_stdio_mcp_server(patch_mcp_session, context: Context):
    tool_name = "ping"
    tool_description = "returns pong"

    _, seen_configs = patch_mcp_session([
        MCPTool(
            name=tool_name,
            description=tool_description,
            inputSchema={"type": "object", "properties": {}},
        )
    ])

    agent = Agent(
        name="test",
        tools=[
            MCPServer(
                MCPStdioServerConfig(
                    command="some-mcp-binary",
                    args=["--flag"],
                )
            )
        ],
        config=TestConfig('{"data": 42}'),
    )
    await agent.ask("test")

    assert len(agent.tools) == 1
    schemas = await agent.tools[0].schemas(context)
    assert schemas[0].function.name == tool_name
    assert schemas[0].function.description == tool_description

    # The toolkit must have dispatched on the stdio config, preserving its fields
    # through the variable-resolution step.
    assert len(seen_configs) >= 1
    config = seen_configs[0]
    assert isinstance(config, MCPStdioServerConfig)
    assert config.command == "some-mcp-binary"
    assert config.args == ["--flag"]


@pytest.mark.asyncio()
async def test_allowed_and_blocked_tools_are_filtered(patch_mcp_session, context: Context):
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


@pytest.mark.asyncio()
async def test_call_tool_returns_text_content(patch_mcp_session, context: Context):
    session, _ = patch_mcp_session(
        [MCPTool(name="echo", description="", inputSchema={"type": "object"})],
        call_results={
            "echo": CallToolResult(content=[TextContent(type="text", text="hello world")]),
        },
    )

    toolkit = MCPServer(MCPStdioServerConfig(command="x"))
    # Trigger discovery so the proxy tool is registered.
    await toolkit.schemas(context)

    proxy = toolkit.tools[0]
    from autogen.beta.events.tool_events import ToolCallEvent, ToolResultEvent

    event = ToolCallEvent(name="echo", arguments="{}")
    result = await proxy(event, context)

    assert isinstance(result, ToolResultEvent)
    assert result.result.content == "hello world"
    assert session.calls == [("echo", {})]
