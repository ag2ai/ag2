# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.agentchat import ConversableAgent
from autogen.tools import Toolkit, tool

from ..conftest import Credentials


class TestToolkit:
    @pytest.fixture
    def toolkit(self) -> Toolkit:
        @tool(description="This is f1")
        def f1() -> None:
            pass

        @tool(description="This is f2")
        def f2() -> None:
            pass

        return Toolkit([f1, f2])

    def test_len(self, toolkit: Toolkit) -> None:
        assert len(toolkit) == 2

    def test_get_tool(self, toolkit: Toolkit) -> None:
        tool = toolkit.get_tool("f1")
        assert tool.description == "This is f1"

        with pytest.raises(ValueError, match="Tool 'f3' not found in Toolkit."):
            toolkit.get_tool("f3")

    def test_remove_tool(self, toolkit: Toolkit) -> None:
        toolkit.remove_tool("f1")
        with pytest.raises(ValueError, match="Tool 'f1' not found in Toolkit."):
            toolkit.get_tool("f1")

    def test_set_tool(self, toolkit: Toolkit) -> None:
        @tool(description="This is f3")
        def f3() -> None:
            pass

        toolkit.set_tool(f3)
        assert len(toolkit) == 3
        f3_tool = toolkit.get_tool("f3")
        assert f3_tool.description == "This is f3"

    def test_register_for_execution(self, toolkit: Toolkit) -> None:
        agent = ConversableAgent(
            name="test_agent",
        )
        toolkit.register_for_execution(agent)
        assert len(agent.function_map) == 2

    def test_register_for_llm(self, toolkit: Toolkit, mock_credentials: Credentials) -> None:
        agent = ConversableAgent(name="test_agent", llm_config=mock_credentials.llm_config)
        toolkit.register_for_llm(agent)
        expected_schema = [
            {
                "type": "function",
                "function": {
                    "description": "This is f1",
                    "name": "f1",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "description": "This is f2",
                    "name": "f2",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]
        assert agent.llm_config["tools"] == expected_schema  # type: ignore[index]
