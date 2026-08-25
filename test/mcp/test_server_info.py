# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata

import pytest
from mcp.server import CacheHint
from pydantic import BaseModel

from ag2 import Agent
from ag2.mcp import MCPServer, build_ask_tool
from ag2.mcp.testing import serve
from ag2.testing import TestConfig


class Weather(BaseModel):
    city: str
    temp_c: float


def test_server_exposes_agent() -> None:
    agent = Agent("greeter", config=TestConfig("hi"))

    assert MCPServer(agent).agent is agent


class TestServerInfo:
    def test_defaults_from_agent(self) -> None:
        agent = Agent("greeter", "You are a greeter.", config=TestConfig("hi"))

        server = MCPServer(agent).server

        assert server.name == "greeter"
        assert server.version == importlib.metadata.version("ag2")
        # instructions is client-facing usage guidance, NOT the agent's system prompt.
        assert server.instructions is None
        # title/description are presentation-only and never derived from the agent.
        assert server.title is None
        assert server.description is None

    def test_overrides(self) -> None:
        agent = Agent("greeter", "You are a greeter.", config=TestConfig("hi"))

        server = MCPServer(
            agent,
            name="custom",
            version="2.0.0",
            title="Greeter",
            description="Answers greetings.",
            instructions="override",
            website_url="https://example.com",
        ).server

        assert server.name == "custom"
        assert server.version == "2.0.0"
        assert server.title == "Greeter"
        assert server.description == "Answers greetings."
        assert server.instructions == "override"
        assert server.website_url == "https://example.com"

    def test_server_info_carries_presentation_fields(self) -> None:
        agent = Agent("greeter", config=TestConfig("hi"))

        info = MCPServer(agent, title="Greeter", description="Answers greetings.").server.server_info

        assert info.title == "Greeter"
        assert info.description == "Answers greetings."


class TestCacheHints:
    def test_rejects_uncacheable_method(self) -> None:
        agent = Agent("greeter", config=TestConfig("hi"))

        with pytest.raises(ValueError, match="cacheable"):
            # The key set is closed for type-checked callers; the runtime gate is
            # for maps that arrive untyped (e.g. deserialized from config).
            MCPServer(agent, cache_hints={"tools/call": CacheHint(ttl_ms=1000)})  # type: ignore[dict-item]

    @pytest.mark.asyncio
    async def test_fills_list_tools_freshness(self) -> None:
        # Freshness hints are a 2026-07-28 surface: handshake-era serialization
        # drops ttlMs/cacheScope, so this goes over the modern per-request path.
        app = MCPServer(
            Agent("greeter", config=TestConfig("hi")),
            json_response=True,
            cache_hints={"tools/list": CacheHint(ttl_ms=60_000, scope="public")},
        )

        async with serve(app) as client:
            resp = await client.post(
                "/mcp",
                headers={
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                    "MCP-Protocol-Version": "2026-07-28",
                    "MCP-Method": "tools/list",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    # The modern era has no initialize handshake; every request
                    # carries the version/capabilities envelope in params._meta.
                    "params": {
                        "_meta": {
                            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                            "io.modelcontextprotocol/clientCapabilities": {},
                        }
                    },
                },
            )

        assert resp.status_code == 200
        result = resp.json()["result"]
        assert [t["name"] for t in result["tools"]] == ["ask"]
        assert result["ttlMs"] == 60_000
        assert result["cacheScope"] == "public"


class TestAskTool:
    def test_input_schema(self) -> None:
        agent = Agent("greeter", config=TestConfig("hi"))

        tool = build_ask_tool(agent)

        assert tool.name == "ask"
        assert tool.input_schema["required"] == ["message"]
        assert set(tool.input_schema["properties"]) == {"message", "context"}

    def test_custom_tool_name_and_description(self) -> None:
        agent = Agent("greeter", config=TestConfig("hi"))

        tool = build_ask_tool(agent, tool_name="chat", tool_description="Talk to me")

        assert tool.name == "chat"
        assert tool.description == "Talk to me"

    def test_no_output_schema_without_response_schema(self) -> None:
        agent = Agent("greeter", config=TestConfig("hi"))

        tool = build_ask_tool(agent, response_schema=agent._response_schema)

        assert tool.output_schema is None

    def test_output_schema_from_response_schema(self) -> None:
        agent = Agent("weather", config=TestConfig("hi"), response_schema=Weather)

        tool = build_ask_tool(agent, response_schema=agent._response_schema)

        assert tool.output_schema is not None
        assert tool.output_schema["type"] == "object"
        assert set(tool.output_schema["properties"]) == {"city", "temp_c"}
