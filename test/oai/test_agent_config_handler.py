# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

from unittest.mock import MagicMock

from pydantic import BaseModel

from autogen.oai.agent_config_handler import agent_config_parser


class TestAgentConfigParser:
    """Test suite for agent_config_parser function."""

    def test_agent_without_response_format_attribute(self):
        """Test agent that doesn't have response_format attribute."""
        agent = MagicMock()
        # Remove response_format if it exists
        if hasattr(agent, "response_format"):
            delattr(agent, "response_format")

        result = agent_config_parser(agent)
        assert isinstance(result, dict)
        assert result == {}

    def test_agent_with_response_format_none(self):
        """Test agent with response_format set to None."""
        agent = MagicMock()
        agent.response_format = None

        result = agent_config_parser(agent)
        assert isinstance(result, dict)
        assert result == {}

    def test_agent_with_response_format_dict(self):
        """Test agent with response_format as a dictionary."""
        agent = MagicMock()
        response_format = {"type": "json_object"}
        agent.response_format = response_format

        result = agent_config_parser(agent)
        assert isinstance(result, dict)
        assert result == {"response_format": response_format}

    def test_agent_with_response_format_pydantic_model(self):
        """Test agent with response_format as a Pydantic BaseModel."""

        class TestModel(BaseModel):
            name: str
            age: int

        agent = MagicMock()
        agent.response_format = TestModel

        result = agent_config_parser(agent)
        assert isinstance(result, dict)
        assert result["response_format"] == TestModel

    def test_agent_with_response_format_string(self):
        """Test agent with response_format as a string."""
        agent = MagicMock()
        response_format = "json_object"
        agent.response_format = response_format

        result = agent_config_parser(agent)
        assert isinstance(result, dict)
        assert result == {"response_format": response_format}

    def test_agent_with_response_format_complex_dict(self):
        """Test agent with response_format as a complex dictionary."""
        agent = MagicMock()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}},
            },
        }
        agent.response_format = response_format

        result = agent_config_parser(agent)
        assert isinstance(result, dict)
        assert result == {"response_format": response_format}

    def test_agent_with_response_format_empty_dict(self):
        """Test agent with response_format as an empty dictionary."""
        agent = MagicMock()
        agent.response_format = {}

        result = agent_config_parser(agent)
        # Empty dict is not None, so it should be included
        assert isinstance(result, dict)
        assert result == {"response_format": {}}

    def test_agent_with_response_format_false(self):
        """Test agent with response_format set to False (falsy but not None)."""
        agent = MagicMock()
        agent.response_format = False

        result = agent_config_parser(agent)
        # False is not None, so it should be included
        assert isinstance(result, dict)
        assert result == {"response_format": False}

    def test_agent_with_response_format_zero(self):
        """Test agent with response_format set to 0 (falsy but not None)."""
        agent = MagicMock()
        agent.response_format = 0

        result = agent_config_parser(agent)
        # 0 is not None, so it should be included
        assert isinstance(result, dict)
        assert result == {"response_format": 0}

    def test_real_conversable_agent_without_response_format(self):
        """Test with a real ConversableAgent instance without response_format."""
        from autogen.agentchat.conversable_agent import ConversableAgent

        agent = ConversableAgent(name="test_agent", llm_config=False)
        # Ensure response_format doesn't exist or is None
        if hasattr(agent, "response_format"):
            agent.response_format = None

        result = agent_config_parser(agent)
        assert isinstance(result, dict)
        assert result == {}

    def test_real_conversable_agent_with_response_format(self):
        """Test with a real ConversableAgent instance with response_format."""
        from autogen.agentchat.conversable_agent import ConversableAgent

        agent = ConversableAgent(name="test_agent", llm_config=False)
        response_format = {"type": "json_object"}
        agent.response_format = response_format

        result = agent_config_parser(agent)
        assert isinstance(result, dict)
        assert result == {"response_format": response_format}

    def test_real_conversable_agent_with_pydantic_response_format(self):
        """Test with a real ConversableAgent instance with Pydantic BaseModel response_format."""
        from autogen.agentchat.conversable_agent import ConversableAgent

        # Define a nested Pydantic model similar to what's used in other tests
        class Step(BaseModel):
            explanation: str
            output: str

        class MathReasoning(BaseModel):
            steps: list[Step]
            final_answer: str

        agent = ConversableAgent(name="test_agent", llm_config=False)
        agent.response_format = MathReasoning

        result = agent_config_parser(agent)
        assert isinstance(result, dict)
        assert result["response_format"] == MathReasoning
        # Verify it's the correct class
        assert issubclass(result["response_format"], BaseModel)
        # Verify the model structure
        assert hasattr(result["response_format"], "model_json_schema")
