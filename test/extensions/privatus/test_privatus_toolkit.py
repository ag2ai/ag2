# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("mcp")

from ag2 import Agent, Variable
from ag2.extensions.privatus import PRIVATUS_MCP_URL, PrivatusAnalyticsToolkit
from ag2.mcp import MCPServer, mcp_tool
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools import MCPServerConfig, MCPToolkit
from ag2.tools.types import FunctionToolSchema

TOKEN = "pat_test_token"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _no_ambient_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRIVATUS_API_TOKEN", raising=False)


def _config(toolkit: PrivatusAnalyticsToolkit) -> MCPServerConfig:
    assert isinstance(toolkit.config, MCPServerConfig)
    return toolkit.config


def test_defaults_point_at_the_hosted_server() -> None:
    toolkit = PrivatusAnalyticsToolkit(TOKEN)

    config = _config(toolkit)
    assert isinstance(toolkit, MCPToolkit)
    assert config.server_url == PRIVATUS_MCP_URL == "https://privatusanalytics.com/mcp"
    assert config.authorization_token == TOKEN
    assert config.server_label == "privatus_analytics"
    assert config.description
    assert config.allowed_tools is None
    assert config.blocked_tools is None
    assert toolkit.name == "privatus_analytics"


def test_token_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIVATUS_API_TOKEN", "pat_from_env")

    assert _config(PrivatusAnalyticsToolkit()).authorization_token == "pat_from_env"


def test_explicit_token_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIVATUS_API_TOKEN", "pat_from_env")

    assert _config(PrivatusAnalyticsToolkit(TOKEN)).authorization_token == TOKEN


@pytest.mark.parametrize("token", [None, ""])
def test_missing_token_raises_a_helpful_error(token: str | None) -> None:
    with pytest.raises(ValueError, match="PRIVATUS_API_TOKEN"):
        PrivatusAnalyticsToolkit(token)


def test_variable_token_is_kept_for_per_run_resolution() -> None:
    token = Variable("privatus_token")

    assert _config(PrivatusAnalyticsToolkit(token)).authorization_token is token


def test_filters_and_settings_reach_the_server_config() -> None:
    toolkit = PrivatusAnalyticsToolkit(
        TOKEN,
        server_url="https://staging.example.test/mcp",
        allowed_tools=["sites_list", "stats_aggregate"],
        blocked_tools=["stats_aggregate"],
        tool_name_prefix="pa_",
        connection_timeout=5.0,
    )

    config = _config(toolkit)
    assert config.server_url == "https://staging.example.test/mcp"
    assert config.allowed_tools == ["sites_list", "stats_aggregate"]
    assert config.blocked_tools == ["stats_aggregate"]
    assert config.tool_name_prefix == "pa_"
    assert config.connection_timeout == 5.0


@mcp_tool
def sites_list() -> str:
    """List the sites in the workspace."""
    return "pa_EXAMPLE1"


@mcp_tool
def stats_aggregate(site_id: str) -> str:
    """Return visitor totals for a site."""
    return f"{site_id}: 42 visitors"


class _RecordAuthorization:
    """ASGI wrapper that remembers the Authorization header of each HTTP request."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.seen: list[str] = []

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            self.seen.append(headers.get(b"authorization", b"").decode())
        await self.app(scope, receive, send)


@asynccontextmanager
async def _serve() -> AsyncGenerator[tuple[str, _RecordAuthorization]]:
    from test._serving import serving

    server = MCPServer(
        Agent("served", config=TestConfig("unused")),
        tools=[sites_list, stats_aggregate],
        path="/mcp",
    )
    app = _RecordAuthorization(server)
    async with serving(app) as base_url:
        yield f"{base_url}/mcp/", app


@pytest.mark.asyncio
async def test_agent_discovers_tools_with_the_bearer_token() -> None:
    tracking = TrackingConfig(TestConfig("done"))
    async with _serve() as (url, app):
        toolkit = PrivatusAnalyticsToolkit(TOKEN, server_url=url, allowed_tools=["sites_list", "stats_aggregate"])
        agent = Agent("analyst", config=tracking, tools=[toolkit])
        await agent.ask("Which sites do I have?")

    offered = sorted(
        schema.function.name for schema in tracking.calls[0].tools if isinstance(schema, FunctionToolSchema)
    )
    assert offered == ["sites_list", "stats_aggregate"]
    assert app.seen
    assert set(app.seen) == {f"Bearer {TOKEN}"}
