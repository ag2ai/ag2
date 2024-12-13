# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
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
            return x

        self.tool = Tool(name="test_tool", description="A test tool", func=f)

    def test_init(self) -> None:
        assert self.tool.name == "test_tool"
        assert self.tool.description == "A test tool"

    def test_register_for_llm(self) -> None:
        config_list = [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]

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