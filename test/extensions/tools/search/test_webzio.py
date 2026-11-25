# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from dirty_equals import IsPartialDict

pytest.importorskip("mcp")
from mcp.types import CallToolResult, ListToolsResult, TextContent
from mcp.types import Tool as MCPTool

from ag2 import Agent, Context
from ag2.events import ToolCallEvent
from ag2.extensions.tools.search import webzio as webzio_module
from ag2.extensions.tools.search.webzio import (
    DEFAULT_MCP_URL,
    PREFERRED_TOOL_NAME,
    SERVER_LABEL,
    TOKEN_ENV_NAME,
    WebzioConfigError,
    WebzioNewsSearchToolkit,
    build_mcp_server_config,
    resolve_api_token,
    resolve_mcp_url,
)
from ag2.testing import TestConfig
from ag2.tools.toolkits.mcp_server import toolkit as _toolkit_module

WEBZIO_SOURCE = Path(inspect.getfile(webzio_module)).read_text(encoding="utf-8")

FILTER_NAMES_OWNED_BY_MCP = (
    "allow_all_dates",
    "exclude_domain",
    "domain_rank_gte",
    "domain_rank_lte",
    "trust_category",
    "political_bias",
    "min_similarity",
    "allow_multiple_chunks_per_article",
)

NEWS_SEARCH_TOOL = MCPTool(
    name=PREFERRED_TOOL_NAME,
    description="Semantic news search.",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language search query"},
            "k": {"type": "integer", "description": "Articles to return"},
        },
        "required": ["query"],
    },
)

MCPSessionPatch = Callable[[list[MCPTool], dict[str, CallToolResult] | None], "_FakeMCPSession"]


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


def test_resolve_api_token_requires_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV_NAME, raising=False)
    with pytest.raises(WebzioConfigError, match="missing Webz API token"):
        resolve_api_token()


def test_resolve_api_token_prefers_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKEN_ENV_NAME, "from-env")
    assert resolve_api_token(" from-arg ") == "from-arg"


def test_resolve_mcp_url_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEBZ_MCP_URL", raising=False)
    assert resolve_mcp_url() == DEFAULT_MCP_URL
    assert resolve_mcp_url("https://localhost:8765/mcp/") == "https://localhost:8765/mcp"


def test_build_mcp_server_config_uses_bearer_and_allowed_tools() -> None:
    config = build_mcp_server_config("secret-token", mcp_url="https://example.test/mcp")
    assert config.server_url == "https://example.test/mcp"
    assert config.authorization_token == "secret-token"
    assert config.allowed_tools == [PREFERRED_TOOL_NAME]
    assert config.server_label == SERVER_LABEL


def test_toolkit_wraps_mcp_server_config() -> None:
    toolkit = WebzioNewsSearchToolkit(api_token="tok", mcp_url="https://example.test/mcp")
    assert isinstance(toolkit, WebzioNewsSearchToolkit)
    assert toolkit.search() is toolkit
    assert toolkit.config.server_url == "https://example.test/mcp"
    assert toolkit.config.authorization_token == "tok"
    assert toolkit.config.allowed_tools == [PREFERRED_TOOL_NAME]


def test_toolkit_source_does_not_hardcode_mcp_filters() -> None:
    for name in FILTER_NAMES_OWNED_BY_MCP:
        assert name not in WEBZIO_SOURCE, f"wrapper must not hardcode MCP filter {name}"


def test_public_export() -> None:
    from ag2.extensions.tools.search import WebzioNewsSearchToolkit as Exported

    assert Exported is WebzioNewsSearchToolkit


@pytest.mark.asyncio
async def test_discovers_news_search_schema(
    patch_mcp_session: MCPSessionPatch,
    context: Context,
) -> None:
    patch_mcp_session([
        NEWS_SEARCH_TOOL,
        MCPTool(name="other_tool", description="", inputSchema={"type": "object"}),
    ])
    toolkit = WebzioNewsSearchToolkit(api_token="tok")

    [schema] = list(await toolkit.schemas(context))

    assert schema.function.name == PREFERRED_TOOL_NAME
    assert schema.function.parameters == IsPartialDict({
        "type": "object",
        "required": ["query"],
        "properties": IsPartialDict({
            "query": IsPartialDict({"type": "string"}),
            "k": IsPartialDict({"type": "integer"}),
        }),
    })


@pytest.mark.asyncio
async def test_search_result_is_returned_to_agent(patch_mcp_session: MCPSessionPatch) -> None:
    session = patch_mcp_session(
        [NEWS_SEARCH_TOOL],
        call_results={
            PREFERRED_TOOL_NAME: CallToolResult(
                content=[TextContent(type="text", text="Query: AI regulation\nTitle: Example")]
            ),
        },
    )
    agent = Agent(
        name="researcher",
        tools=[WebzioNewsSearchToolkit(api_token="tok")],
        config=TestConfig(
            ToolCallEvent(
                name=PREFERRED_TOOL_NAME,
                arguments='{"query": "AI regulation", "k": 3}',
            ),
            "done",
        ),
    )

    result = await agent.ask("search")

    assert result.body == "done"
    assert session.calls == [(PREFERRED_TOOL_NAME, {"query": "AI regulation", "k": 3})]
