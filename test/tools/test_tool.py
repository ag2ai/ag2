# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from autogen import AssistantAgent, UserProxyAgent
from autogen.tools import Tool


class TestTool:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        def f(x: str) -> str:
            return x + "!"

        self.tool = Tool(name="test_tool", description="A test tool", func_or_tool=f)

    def test_init(self) -> None:
        assert self.tool.name == "test_tool"
        assert self.tool.description == "A test tool"

    def test_register_for_llm(self) -> None:
        config_list = [{"api_type": "openai", "model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]

        agent = AssistantAgent(
            name="agent",
            llm_config={"config_list": config_list},
        )

        self.tool.register_for_llm(agent=agent)

        expected_tools = [
            {
                "type": "function",
                "function": {
                    "description": "A test tool",
                    "name": "test_tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string", "description": "x"}},
                        "required": ["x"],
                    },
                },
            }
        ]

        assert agent.llm_config["tools"] == expected_tools  # type: ignore[index]

    def test_register_for_execution(self) -> None:
        user_proxy = UserProxyAgent(
            name="user",
        )

        self.tool.register_for_execution(user_proxy)
        assert user_proxy.can_execute_function("test_tool")
        assert user_proxy.function_map["test_tool"]("Hello") == "Hello!"

    def test__call__(self) -> None:
        assert self.tool("Hello") == "Hello!"

    def test_parameters_json_schema_round_trip(self) -> None:
        # Pin issue #1814: parameters_json_schema must survive Tool-from-Tool copies
        # (which is the path _create_tool_if_needed takes when rewrapping a tool for
        # context_variables dependency injection in group/swarm flows).
        custom_schema = {
            "type": "object",
            "properties": {"x": {"type": "string", "enum": ["a", "b"]}},
            "required": ["x"],
        }

        def f(x: str) -> str:
            return x + "!"

        original = Tool(name="enum_tool", description="enum-using tool", func_or_tool=f, parameters_json_schema=custom_schema)
        assert original.parameters_json_schema == custom_schema
        assert original._func_schema is not None
        assert original._func_schema["function"]["parameters"] == custom_schema

        # Copy without explicit parameters_json_schema must carry the original through.
        copied = Tool(func_or_tool=original)
        assert copied.parameters_json_schema == custom_schema
        assert copied._func_schema is not None
        assert copied._func_schema["function"]["parameters"] == custom_schema

        # Explicit override on copy still wins.
        override_schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        overridden = Tool(func_or_tool=original, parameters_json_schema=override_schema)
        assert overridden.parameters_json_schema == override_schema
        assert overridden._func_schema is not None
        assert overridden._func_schema["function"]["parameters"] == override_schema
