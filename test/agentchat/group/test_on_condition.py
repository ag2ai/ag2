# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from autogen.agentchat.group.available_condition import StringAvailableCondition
from autogen.agentchat.group.context_str import ContextStr
from autogen.agentchat.group.llm_condition import ContextStrLLMCondition, StringLLMCondition
from autogen.agentchat.group.on_condition import OnCondition
from autogen.agentchat.group.transition_target import AgentTarget, TransitionTarget


class TestOnCondition:
    def test_init(self) -> None:
        """Test initialisation with basic values."""
        target = MagicMock(spec=TransitionTarget)
        condition = MagicMock(spec=StringLLMCondition)
        available = MagicMock(spec=StringAvailableCondition)

        on_condition = OnCondition(target=target, condition=condition, available=available)

        assert on_condition.target == target
        assert on_condition.condition == condition
        assert on_condition.available == available

    def test_init_with_none_available(self) -> None:
        """Test initialisation with None available."""
        target = MagicMock(spec=TransitionTarget)
        condition = MagicMock(spec=StringLLMCondition)

        on_condition = OnCondition(target=target, condition=condition, available=None)

        assert on_condition.target == target
        assert on_condition.condition == condition
        assert on_condition.available is None

    def test_init_with_string_llm_condition(self) -> None:
        """Test initialisation with StringLLMCondition."""
        target = MagicMock(spec=TransitionTarget)
        condition = StringLLMCondition(prompt="Is this a valid condition?")

        on_condition = OnCondition(target=target, condition=condition)

        assert on_condition.target == target
        assert on_condition.condition == condition
        assert on_condition.condition.prompt == "Is this a valid condition?"

    def test_init_with_context_str_llm_condition(self) -> None:
        """Test initialisation with ContextStrLLMCondition."""
        target = MagicMock(spec=TransitionTarget)
        context_str = ContextStr("Is the value of x equal to {x}?")
        condition = ContextStrLLMCondition(context_str=context_str)

        on_condition = OnCondition(target=target, condition=condition)

        assert on_condition.target == target
        assert on_condition.condition == condition
        assert on_condition.condition.context_str == context_str

    def test_init_with_agent_target(self) -> None:
        """Test initialisation with AgentTarget."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        condition = StringLLMCondition(prompt="Is this a valid condition?")

        on_condition = OnCondition(target=target, condition=condition)

        assert on_condition.target == target
        assert on_condition.target.agent_name == "test_agent"

    def test_init_with_string_available_condition(self) -> None:
        """Test initialisation with string available condition."""
        target = MagicMock(spec=TransitionTarget)
        condition = StringLLMCondition(prompt="Is this a valid condition?")
        available = StringAvailableCondition(context_variable="is_available")

        on_condition = OnCondition(target=target, condition=condition, available=available)

        assert on_condition.available == available
        assert on_condition.available.context_variable == "is_available"

    def test_init_with_context_expression_available(self) -> None:
        """Test initialisation with ContextExpression available."""
        target = MagicMock(spec=TransitionTarget)
        condition = StringLLMCondition(prompt="Is this a valid condition?")
        available = MagicMock()
        available.evaluate = MagicMock()

        on_condition = OnCondition(target=target, condition=condition, available=available)

        assert on_condition.available == available

    @patch("autogen.agentchat.group.on_condition.LLMCondition.__subclasshook__")
    def test_condition_get_prompt(self, mock_subclasshook) -> None:
        """Test that condition.get_prompt is called correctly."""
        target = MagicMock(spec=TransitionTarget)
        condition = MagicMock()
        condition.get_prompt = MagicMock(return_value="Prompt text")

        on_condition = OnCondition(target=target, condition=condition)

        mock_agent = MagicMock()
        messages = [{"role": "user", "content": "Hello"}]

        # Call get_prompt through the condition
        result = on_condition.condition.get_prompt(mock_agent, messages)

        # Verify the mock was called correctly
        condition.get_prompt.assert_called_once_with(mock_agent, messages)
        assert result == "Prompt text"

    @patch("autogen.agentchat.group.on_condition.AvailableCondition.__subclasshook__")
    def test_available_is_available(self, mock_subclasshook) -> None:
        """Test that available.is_available is called correctly."""
        target = MagicMock(spec=TransitionTarget)
        condition = MagicMock()
        available = MagicMock()
        available.is_available = MagicMock(return_value=True)

        on_condition = OnCondition(target=target, condition=condition, available=available)

        mock_agent = MagicMock()
        messages = [{"role": "user", "content": "Hello"}]

        # Call is_available through the available
        result = on_condition.available.is_available(mock_agent, messages)

        # Verify the mock was called correctly
        available.is_available.assert_called_once_with(mock_agent, messages)
        assert result is True
