# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

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


class TestRemoveParamsFromSignature:
    def f_without_the_param(  # type: ignore[misc]
        a: int,  # noqa: N805
    ) -> int:
        return a

    def f_with_the_param(  # type: ignore[misc]
        a: int,  # noqa: N805
        context_variables: dict[str, Any] = {},
    ) -> int:
        return a

    def test_remove_params(self) -> None:
        final_schema = {
            "description": "A test tool",
            "name": "test_tool",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer", "description": "a"}},
                "required": ["a"],
            },
        }

        # Test removing parameters that don't exist
        test_tool = Tool(
            name="test_tool", description="A test tool", func_or_tool=TestRemoveParamsFromSignature.f_without_the_param
        )
        test_tool.hide_parameters(["context_variables"])
        assert test_tool.function_schema == final_schema

        # Test removing parameter that does exist
        test_tool = Tool(
            name="test_tool", description="A test tool", func_or_tool=TestRemoveParamsFromSignature.f_with_the_param
        )
        test_tool.hide_parameters(["context_variables"])
        assert test_tool.function_schema == final_schema
