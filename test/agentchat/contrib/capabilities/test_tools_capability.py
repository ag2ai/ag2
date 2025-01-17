# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen import AssistantAgent
from autogen.agentchat.contrib.capabilities.tools_capability import ToolSpecs, ToolsCapability


@pytest.fixture
def add_tool_spec():
    def add(x: int, y: int) -> int:
        return x + y

    return ToolSpecs(
        tool_func=add,
        tool_description="Provide add function to two argument and return sum.",
        tool_name="add_function",
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
    def test_add_capability(self, add_tool_spec, test_agent) -> None:
        # Arrange
        tools_capability = ToolsCapability(tool_list=[add_tool_spec])
        assert "tools" not in test_agent.llm_config
        # Act
        tools_capability.add_to_agent(agent=test_agent)
        # Assert
        assert len(test_agent.llm_config["tools"]) == 1
