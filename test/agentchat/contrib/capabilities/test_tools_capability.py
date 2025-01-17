# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen import AssistantAgent
from autogen.agentchat.contrib.capabilities.tools_capability import ToolsCapability
from autogen.tools import Tool


@pytest.fixture
def add_tools():
    def add(x: int, y: int) -> int:
        return x + y

    return Tool(
        name="add_function",
        description="Provide add function to two argument and return sum.",
        func_or_tool=add,
    )


@pytest.fixture
def test_agent():
    return AssistantAgent(
        name="test_agent",
        llm_config={
            "config_list": [{"model": "gpt-4O", "api_key": "sk-proj-ABC"}],
        },
    )


class TestToolsCapability:
    def test_add_capability(self, add_tools, test_agent) -> None:
        # Arrange
        tools_capability = ToolsCapability(tool_list=[add_tools])
        assert "tools" not in test_agent.llm_config
        # Act
        tools_capability.add_to_agent(agent=test_agent)
        # Assert
        assert len(test_agent.llm_config["tools"]) == 1
