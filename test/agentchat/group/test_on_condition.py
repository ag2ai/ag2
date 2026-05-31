# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from autogen.agentchat.group.available_condition import StringAvailableCondition
from autogen.agentchat.group.context_str import ContextStr
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.llm_condition import ContextStrLLMCondition, StringLLMCondition
from autogen.agentchat.group.on_condition import OnCondition
from autogen.agentchat.group.targets.transition_target import (
    AgentNameTarget,
    AgentTarget,
    NestedChatTarget,
    TransitionTarget,
)


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
        assert isinstance(on_condition.condition, StringLLMCondition)
        assert on_condition.condition.prompt == "Is this a valid condition?"

    def test_init_with_context_str_llm_condition(self) -> None:
        """Test initialisation with ContextStrLLMCondition."""
        target = MagicMock(spec=TransitionTarget)
        context_str = ContextStr(template="Is the value of x equal to {x}?")
        condition = ContextStrLLMCondition(context_str=context_str)

        on_condition = OnCondition(target=target, condition=condition)

        assert on_condition.target == target
        assert on_condition.condition == condition
        assert isinstance(on_condition.condition, ContextStrLLMCondition)
        assert on_condition.condition.context_str == context_str

    def test_init_with_agent_target(self) -> None:
        """Test initialisation with AgentTarget."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        condition = StringLLMCondition(prompt="Is this a valid condition?")

        on_condition = OnCondition(target=target, condition=condition)

        assert on_condition.target == target
        assert isinstance(on_condition.target, AgentTarget)
        assert on_condition.target.agent_name == "test_agent"

    def test_init_with_string_available_condition(self) -> None:
        """Test initialisation with string available condition."""
        target = MagicMock(spec=TransitionTarget)
        condition = StringLLMCondition(prompt="Is this a valid condition?")
        available = StringAvailableCondition(context_variable="is_available")

        on_condition = OnCondition(target=target, condition=condition, available=available)

        assert on_condition.available == available
        assert isinstance(on_condition.available, StringAvailableCondition)
        assert on_condition.available.context_variable == "is_available"

    def test_init_with_context_expression_available(self) -> None:
        """Test initialisation with ContextExpression available."""
        target = MagicMock(spec=TransitionTarget)
        condition = StringLLMCondition(prompt="Is this a valid condition?")
        available = StringAvailableCondition(context_variable="is_logged_in")

        on_condition = OnCondition(target=target, condition=condition, available=available)

        assert on_condition.available == available

    @patch("autogen.agentchat.group.on_condition.LLMCondition.__subclasshook__")
    def test_condition_get_prompt(self, mock_subclasshook: MagicMock) -> None:
        """Test that condition.get_prompt is called correctly."""
        target = MagicMock(spec=TransitionTarget)
        condition = StringLLMCondition(prompt="Prompt text")

        on_condition = OnCondition(target=target, condition=condition)

        mock_agent = MagicMock()
        messages = [{"role": "user", "content": "Hello"}]

        # Call get_prompt through the condition
        result = on_condition.condition.get_prompt(mock_agent, messages)

        # Verify the mock was called correctly
        assert result == "Prompt text"

    @patch("autogen.agentchat.group.on_condition.AvailableCondition.__subclasshook__")
    def test_available_is_available(self, mock_subclasshook: MagicMock) -> None:
        """Test that available.is_available is called correctly."""
        target = MagicMock(spec=TransitionTarget)
        condition = StringLLMCondition(prompt="Test Prompt")
        available = StringAvailableCondition(context_variable="is_available")
        # available.is_available = MagicMock(return_value=True)

        on_condition = OnCondition(target=target, condition=condition, available=available)

        mock_agent = MagicMock()
        messages = [{"role": "user", "content": "Hello"}]

        # Call is_available through the available
        assert on_condition.available is not None
        result = on_condition.available.is_available(mock_agent, messages)

        # Verify the mock was called correctly
        assert result is True

    def test_has_target_type(self) -> None:
        """Test the has_target_type method with various target types."""
        # Test with AgentTarget
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        condition = StringLLMCondition(prompt="Should we transfer to test_agent?")

        on_condition = OnCondition(target=target, condition=condition)

        # Should match the correct type
        assert on_condition.has_target_type(AgentTarget) is True

        # Should not match other types
        assert on_condition.has_target_type(AgentNameTarget) is False
        assert on_condition.has_target_type(NestedChatTarget) is False

        # Test with AgentNameTarget
        target_agent_name = AgentNameTarget(agent_name="test_agent")
        on_condition = OnCondition(target=target_agent_name, condition=condition)

        assert on_condition.has_target_type(AgentNameTarget) is True
        assert on_condition.has_target_type(AgentTarget) is False

        # Test with NestedChatTarget
        target_nested_chat = NestedChatTarget(nested_chat_config={"chat_queue": []})
        on_condition = OnCondition(target=target_nested_chat, condition=condition)

        assert on_condition.has_target_type(NestedChatTarget) is True
        assert on_condition.has_target_type(AgentTarget) is False

    def test_target_requires_wrapping(self) -> None:
        """Test the target_requires_wrapping method with different target types."""
        condition = StringLLMCondition(prompt="Just a test condition")

        # Test with AgentTarget (should not require wrapping)
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)

        on_condition = OnCondition(target=target, condition=condition)
        assert on_condition.target_requires_wrapping() is False

        # Test with NestedChatTarget (should require wrapping)
        target_nested_chat = NestedChatTarget(nested_chat_config={"chat_queue": []})

        on_condition = OnCondition(target=target_nested_chat, condition=condition)
        assert on_condition.target_requires_wrapping() is True

    def test_llm_function_name_handling(self) -> None:
        """Test setting and getting the llm_function_name."""
        target = MagicMock(spec=TransitionTarget)
        condition = StringLLMCondition(prompt="Test prompt")

        # Initialize with no function name
        on_condition = OnCondition(target=target, condition=condition)
        assert on_condition.llm_function_name is None

        # Set the function name
        function_name = "transfer_to_agent1"
        on_condition.llm_function_name = function_name
        assert on_condition.llm_function_name == function_name

    def test_integration_with_real_components(self) -> None:
        """Test integration of OnCondition with real components."""
        # Create a real agent target
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)

        # Create a real LLM condition
        condition = StringLLMCondition(prompt="Should we transfer to {agent_name}?")

        # Create a real available condition
        available = StringAvailableCondition(context_variable="is_transfer_allowed")

        # Create the OnCondition
        on_condition = OnCondition(
            target=target, condition=condition, available=available, llm_function_name="transfer_to_test_agent"
        )

        # Set up context variables
        mock_agent_for_eval = MagicMock()
        mock_agent_for_eval.context_variables = ContextVariables(
            data={"is_transfer_allowed": True, "agent_name": "test_agent"}
        )

        # Test available condition
        assert isinstance(on_condition.available, StringAvailableCondition)
        assert on_condition.available.is_available(mock_agent_for_eval, []) is True

        # Test getting the prompt
        mock_agent = MagicMock(spec="ConversableAgent")
        prompt = on_condition.condition.get_prompt(mock_agent, [])
        assert prompt == "Should we transfer to {agent_name}?"
