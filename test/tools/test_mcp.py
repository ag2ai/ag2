# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
from collections.abc import Callable
from contextlib import asynccontextmanager

import pytest
from dirty_equals import IsPartialDict

pytest.importorskip("mcp")
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ListToolsResult,
    ResourceLink,
    TextContent,
    TextResourceContents,
)
from mcp.types import Tool as MCPTool

from ag2 import Agent, Context, Variable
from ag2.events import BinaryInput, BinaryType, TextInput, ToolCallEvent, ToolResultEvent, UrlInput
from ag2.testing import TestConfig
from ag2.tools import MCPStdioServerConfig, MCPToolkit
from ag2.tools.toolkits.mcp_server import toolkit as _toolkit_module
from ag2.tools.types import FunctionToolSchema

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

    toolkit = MCPToolkit("https://mcp.example.com")
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

    toolkit = MCPToolkit(
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

    toolkit = MCPToolkit(
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
        tools=[MCPToolkit(MCPStdioServerConfig(command="x"))],
        config=TestConfig(
            ToolCallEvent(name="echo", arguments="{}"),
            "done",
        ),
    )
    result = await agent.ask("test")

    assert result.body == "done"
    assert session.calls == [("echo", {})]


@pytest.mark.asyncio
async def test_tool_name_prefix_namespaces_local_name_but_calls_remote_name(
    patch_mcp_session: MCPSessionPatch,
    context: Context,
) -> None:
    session = patch_mcp_session(
        [MCPTool(name="search", description="", inputSchema={"type": "object"})],
        {
            "search": CallToolResult(content=[TextContent(type="text", text="found")]),
        },
    )
    toolkit = MCPToolkit(MCPStdioServerConfig(command="x", tool_name_prefix="github_"))
    [schema] = list(await toolkit.schemas(context))
    agent = Agent(
        name="test",
        tools=[toolkit],
        config=TestConfig(
            ToolCallEvent(name="github_search", arguments='{"query": "ag2"}'),
            "done",
        ),
    )

    result = await agent.ask("search")

    assert result.body == "done"
    assert isinstance(schema, FunctionToolSchema)
    assert schema.function.name == "github_search"
    assert session.calls == [("search", {"query": "ag2"})]


@pytest.mark.asyncio
async def test_discovery_filters_match_remote_names_not_prefixed_ones(
    patch_mcp_session: MCPSessionPatch,
    context: Context,
) -> None:
    """``allowed_tools`` / ``blocked_tools`` are written in the server's own names."""
    patch_mcp_session([
        MCPTool(name="keep", description="", inputSchema={"type": "object"}),
        MCPTool(name="drop_blocked", description="", inputSchema={"type": "object"}),
        MCPTool(name="drop_unlisted", description="", inputSchema={"type": "object"}),
    ])

    toolkit = MCPToolkit(
        MCPStdioServerConfig(
            command="x",
            allowed_tools=["keep", "drop_blocked"],
            blocked_tools=["drop_blocked"],
            tool_name_prefix="github_",
        )
    )
    schemas = list(await toolkit.schemas(context))

    assert [s.function.name for s in schemas] == ["github_keep"]


@pytest.mark.asyncio
async def test_prefixes_keep_two_servers_exposing_the_same_tool_name_apart(
    patch_mcp_session: MCPSessionPatch,
    context: Context,
) -> None:
    """Without prefixes both proxies answer one call; with them, only the addressed one does."""
    session = patch_mcp_session([MCPTool(name="search", description="", inputSchema={"type": "object"})])
    github = MCPToolkit(MCPStdioServerConfig(command="github-mcp", tool_name_prefix="github_"))
    docs = MCPToolkit(MCPStdioServerConfig(command="docs-mcp", tool_name_prefix="docs_"))

    schemas = list(await github.schemas(context)) + list(await docs.schemas(context))
    agent = Agent(
        name="test",
        tools=[github, docs],
        config=TestConfig(
            ToolCallEvent(name="docs_search", arguments="{}"),
            "done",
        ),
    )

    result = await agent.ask("search")

    assert result.body == "done"
    assert [s.function.name for s in schemas] == ["github_search", "docs_search"]
    assert session.calls == [("search", {})]


@pytest.mark.asyncio
class TestToolNamePrefixVariable:
    async def test_resolved(
        self,
        patch_mcp_session: MCPSessionPatch,
        make_context: Callable[..., Context],
    ) -> None:
        patch_mcp_session([MCPTool(name="search", description="", inputSchema={"type": "object"})])
        ctx = make_context(tenant="acme_")

        toolkit = MCPToolkit(MCPStdioServerConfig(command="x", tool_name_prefix=Variable("tenant")))
        [schema] = list(await toolkit.schemas(ctx))

        assert isinstance(schema, FunctionToolSchema)
        assert schema.function.name == "acme_search"

    async def test_missing_raises(
        self,
        patch_mcp_session: MCPSessionPatch,
        context: Context,
    ) -> None:
        patch_mcp_session([MCPTool(name="search", description="", inputSchema={"type": "object"})])

        toolkit = MCPToolkit(MCPStdioServerConfig(command="x", tool_name_prefix=Variable("tenant")))

        with pytest.raises(KeyError, match="tenant"):
            await toolkit.schemas(context)


@pytest.mark.asyncio
async def test_extract_maps_content_blocks_to_typed_inputs(
    patch_mcp_session: MCPSessionPatch, context: Context
) -> None:
    """Each MCP ContentBlock variant maps to the matching AG2 Input type."""
    image_bytes = b"\x89PNG\r\n\x1a\nfake-image-bytes"
    audio_bytes = b"fake-audio-bytes"
    blob_bytes = b"fake-blob-bytes"

    patch_mcp_session(
        [MCPTool(name="multi", description="", inputSchema={"type": "object"})],
        call_results={
            "multi": CallToolResult(
                content=[
                    TextContent(type="text", text="hello"),
                    ImageContent(
                        type="image",
                        data=base64.b64encode(image_bytes).decode("ascii"),
                        mimeType="image/png",
                    ),
                    AudioContent(
                        type="audio",
                        data=base64.b64encode(audio_bytes).decode("ascii"),
                        mimeType="audio/wav",
                    ),
                    ResourceLink(
                        type="resource_link",
                        name="doc",
                        uri="https://example.com/doc.pdf",
                        mimeType="application/pdf",
                    ),
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri="file:///etc/hosts",
                            text="127.0.0.1 localhost",
                            mimeType="text/plain",
                        ),
                    ),
                    EmbeddedResource(
                        type="resource",
                        resource=BlobResourceContents(
                            uri="file:///tmp/blob.bin",
                            blob=base64.b64encode(blob_bytes).decode("ascii"),
                            mimeType="image/jpeg",
                        ),
                    ),
                ]
            ),
        },
    )

    mcp = MCPToolkit(MCPStdioServerConfig(command="x"))
    await mcp.schemas(context)
    proxy = next(t for t in mcp.tools if t.name == "multi")

    result = await proxy(ToolCallEvent(name="multi", arguments="{}"), context)

    assert isinstance(result, ToolResultEvent)
    parts = result.result.parts
    assert parts == [
        TextInput(content="hello"),
        BinaryInput(data=image_bytes, media_type="image/png", kind=BinaryType.IMAGE),
        BinaryInput(data=audio_bytes, media_type="audio/wav", kind=BinaryType.AUDIO),
        UrlInput("https://example.com/doc.pdf", kind=BinaryType.DOCUMENT),
        TextInput(content="127.0.0.1 localhost"),
        BinaryInput(data=blob_bytes, media_type="image/jpeg", kind=BinaryType.IMAGE),
    ]


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
