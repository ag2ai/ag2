# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from autogen.beta import Agent
from autogen.beta.mcp import build_ask_tool, build_server_info
from autogen.beta.testing import TestConfig


class Weather(BaseModel):
    city: str
    temp_c: float


class TestServerInfo:
    def test_defaults_from_agent(self) -> None:
        agent = Agent("greeter", "You are a greeter.", config=TestConfig("hi"))

        name, version, instructions = build_server_info(agent)

        assert name == "greeter"
        assert version == "0.1.0"
        assert instructions == "You are a greeter."

    def test_overrides(self) -> None:
        agent = Agent("greeter", "You are a greeter.", config=TestConfig("hi"))

        name, version, instructions = build_server_info(agent, name="custom", version="2.0.0", instructions="override")

        assert (name, version, instructions) == ("custom", "2.0.0", "override")

    def test_no_prompt_has_none_instructions(self) -> None:
        agent = Agent("bare", config=TestConfig("hi"))

        _, _, instructions = build_server_info(agent)

        assert instructions is None


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

        tool = build_ask_tool(agent)

        assert tool.outputSchema is None

    def test_output_schema_from_response_schema(self) -> None:
        agent = Agent("weather", config=TestConfig("hi"), response_schema=Weather)

        tool = build_ask_tool(agent)

        assert tool.outputSchema is not None
        assert tool.outputSchema["type"] == "object"
        assert set(tool.outputSchema["properties"]) == {"city", "temp_c"}
