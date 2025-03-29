# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.group.context_str import ContextStr
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.llm_condition import (
    ContextStrLLMCondition,
    StringLLMCondition,
)


class TestLLMCondition:
    def test_protocol_raise_not_implemented(self) -> None:
        """Test that the LLMCondition protocol raises NotImplementedError when implemented without override."""

        # Create a class that implements the protocol but doesn't override get_prompt
        class TestImpl:
            def get_prompt(self, agent, messages):
                raise NotImplementedError("Requires subclasses to implement.")

        impl = TestImpl()
        with pytest.raises(NotImplementedError) as excinfo:
            impl.get_prompt(None, [])
        assert "Requires subclasses to implement" in str(excinfo.value)


class TestStringLLMCondition:
    def test_init(self) -> None:
        """Test initialisation with a prompt string."""
        prompt = "Is this a test?"
        condition = StringLLMCondition(prompt=prompt)
        assert condition.prompt == prompt

    def test_get_prompt(self) -> None:
        """Test get_prompt returns the static prompt string."""
        prompt = "Is this a test?"
        condition = StringLLMCondition(prompt=prompt)

        # Agent and messages are not used
        mock_agent = MagicMock()
        messages = [{"role": "user", "content": "Hello"}]

        result = condition.get_prompt(mock_agent, messages)
        assert result == prompt

    def test_get_prompt_ignores_agent_and_messages(self) -> None:
        """Test get_prompt ignores the agent and messages arguments."""
        prompt = "Is this a test?"
        condition = StringLLMCondition(prompt=prompt)

        # Different agents and messages should produce identical results
        mock_agent1 = MagicMock()
        mock_agent2 = MagicMock()
        messages1 = [{"role": "user", "content": "Hello"}]
        messages2 = [{"role": "user", "content": "Different content"}]

        result1 = condition.get_prompt(mock_agent1, messages1)
        result2 = condition.get_prompt(mock_agent2, messages2)

        assert result1 == result2 == prompt


class TestContextStrLLMCondition:
    def test_init(self) -> None:
        """Test initialisation with a ContextStr object."""
        context_str = MagicMock(spec=ContextStr)
        condition = ContextStrLLMCondition(context_str=context_str)
        assert condition.context_str == context_str

    @patch.object(ContextStr, "format")
    def test_get_prompt(self, mock_format) -> None:
        """Test get_prompt calls format on the ContextStr with the agent's context variables."""
        # Mock ContextStr and its format method
        mock_context_str = MagicMock(spec=ContextStr)
        mock_format.return_value = "Formatted prompt with value=42"
        mock_context_str.format = mock_format

        condition = ContextStrLLMCondition(context_str=mock_context_str)

        # Set up mock agent with context variables
        mock_agent = MagicMock()
        mock_context_vars = ContextVariables(data={"value": 42})
        mock_agent.context_variables = mock_context_vars

        # Messages are not used
        messages = [{"role": "user", "content": "Hello"}]

        result = condition.get_prompt(mock_agent, messages)

        # Verify format was called with context variables
        mock_format.assert_called_once_with(mock_context_vars)
        assert result == "Formatted prompt with value=42"

    def test_get_prompt_with_real_context_str(self) -> None:
        """Test get_prompt using a real ContextStr with variable substitution."""
        # Create a real ContextStr
        template = "Is the value of x equal to {x}?"
        context_str = ContextStr(template)

        condition = ContextStrLLMCondition(context_str=context_str)

        # Set up mock agent with context variables
        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(data={"x": 42})

        # Messages are not used
        messages = [{"role": "user", "content": "Hello"}]

        result = condition.get_prompt(mock_agent, messages)
        assert result == "Is the value of x equal to 42?"

    def test_get_prompt_with_multiple_variables(self) -> None:
        """Test get_prompt with a ContextStr containing multiple variables."""
        # Create a real ContextStr with multiple variables
        template = "User {name} has account type {account_type} with balance {balance}."
        context_str = ContextStr(template)

        condition = ContextStrLLMCondition(context_str=context_str)

        # Set up mock agent with context variables
        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(
            data={"name": "Alice", "account_type": "Premium", "balance": "$100.50"}
        )

        # Messages are not used
        messages = [{"role": "user", "content": "Hello"}]

        result = condition.get_prompt(mock_agent, messages)
        assert result == "User Alice has account type Premium with balance $100.50."

    def test_get_prompt_with_missing_variables(self) -> None:
        """Test get_prompt raises KeyError when variables are missing from context variables."""
        # Create a real ContextStr with a variable that might be missing
        template = "User {name} has access level {access_level}."
        context_str = ContextStr(template)

        condition = ContextStrLLMCondition(context_str=context_str)

        # Set up mock agent with incomplete context variables
        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(data={"name": "Bob"})  # access_level is missing

        # Messages are not used
        messages = [{"role": "user", "content": "Hello"}]

        # The format should raise a KeyError
        with pytest.raises(KeyError) as excinfo:
            condition.get_prompt(mock_agent, messages)

        assert "access_level" in str(excinfo.value)
