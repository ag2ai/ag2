# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata

from pydantic import BaseModel

from ag2 import Agent
from ag2.mcp import MCPServer, build_ask_tool
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

    def test_overrides(self) -> None:
        agent = Agent("greeter", "You are a greeter.", config=TestConfig("hi"))

        server = MCPServer(
            agent,
            name="custom",
            version="2.0.0",
            instructions="override",
            website_url="https://example.com",
        ).server

        assert server.name == "custom"
        assert server.version == "2.0.0"
        assert server.instructions == "override"
        assert server.website_url == "https://example.com"


class TestAskTool:
    def test_input_schema(self) -> None:
        agent = Agent("greeter", config=TestConfig("hi"))

        tool = build_ask_tool(agent)

        assert tool.name == "ask"
        assert tool.inputSchema["required"] == ["message"]
        assert set(tool.inputSchema["properties"]) == {"message", "context"}

    def test_custom_tool_name_and_description(self) -> None:
        agent = Agent("greeter", config=TestConfig("hi"))

        tool = build_ask_tool(agent, tool_name="chat", tool_description="Talk to me")

        assert tool.name == "chat"
        assert tool.description == "Talk to me"

    def test_no_output_schema_without_response_schema(self) -> None:
        agent = Agent("greeter", config=TestConfig("hi"))

        tool = build_ask_tool(agent, response_schema=agent._response_schema)

        assert tool.outputSchema is None

    def test_output_schema_from_response_schema(self) -> None:
        agent = Agent("weather", config=TestConfig("hi"), response_schema=Weather)

        tool = build_ask_tool(agent, response_schema=agent._response_schema)

        assert tool.outputSchema is not None
        assert tool.outputSchema["type"] == "object"
        assert set(tool.outputSchema["properties"]) == {"city", "temp_c"}
