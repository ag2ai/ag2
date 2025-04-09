# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import copy
from unittest.mock import MagicMock

import pytest

from autogen.agentchat.group.context_condition import StringContextCondition
from autogen.agentchat.group.handoffs import Handoffs
from autogen.agentchat.group.llm_condition import StringLLMCondition
from autogen.agentchat.group.on_condition import OnCondition
from autogen.agentchat.group.on_context_condition import OnContextCondition
from autogen.agentchat.group.targets.transition_target import (
    AgentNameTarget,
    AgentTarget,
    NestedChatTarget,
    TransitionTarget,
)


class TestHandoffs:
    @pytest.fixture
    def mock_agent_target(self) -> AgentTarget:
        """Create a mock AgentTarget for testing."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        return AgentTarget(agent=mock_agent)

    @pytest.fixture
    def mock_agent_name_target(self) -> AgentNameTarget:
        """Create a mock AgentNameTarget for testing."""
        return AgentNameTarget(agent_name="test_agent")

    @pytest.fixture
    def mock_nested_chat_target(self) -> NestedChatTarget:
        """Create a mock NestedChatTarget for testing."""
        nested_chat_config = {"chat_queue": ["agent1", "agent2"], "use_async": True}
        return NestedChatTarget(nested_chat_config=nested_chat_config)

    @pytest.fixture
    def mock_on_context_condition(self, mock_agent_target: AgentTarget) -> OnContextCondition:
        """Create a mock OnContextCondition for testing."""
        condition = StringContextCondition(variable_name="test_condition")
        return OnContextCondition(target=mock_agent_target, condition=condition)

    @pytest.fixture
    def mock_on_condition(self, mock_agent_target: AgentTarget) -> OnCondition:
        """Create a mock OnCondition for testing."""
        condition = StringLLMCondition(prompt="Is this a test?")
        return OnCondition(target=mock_agent_target, condition=condition)

    @pytest.fixture
    def mock_on_context_condition_require_wrapping(self, mock_agent_target: AgentTarget) -> OnContextCondition:
        """Create a mock OnContextCondition for testing."""
        condition = StringContextCondition(variable_name="test_condition")
        nested_chat_config = {"chat_queue": ["agent1", "agent2"], "use_async": True}
        return OnContextCondition(target=NestedChatTarget(nested_chat_config=nested_chat_config), condition=condition)

    @pytest.fixture
    def mock_on_condition_require_wrapping(self, mock_agent_target: AgentTarget) -> OnCondition:
        """Create a mock OnCondition for testing."""
        condition = StringLLMCondition(prompt="Is this a test?")
        nested_chat_config = {"chat_queue": ["agent1", "agent2"], "use_async": True}
        return OnCondition(target=NestedChatTarget(nested_chat_config=nested_chat_config), condition=condition)

    @pytest.fixture
    def mock_after_work(self, mock_agent_target: AgentTarget) -> TransitionTarget:
        """Create a mock AfterWork for testing."""
        return mock_agent_target

    def test_init_empty(self) -> None:
        """Test initialization with no conditions."""
        handoffs = Handoffs()
        assert handoffs.context_conditions == []
        assert handoffs.llm_conditions == []
        assert handoffs.after_work is None

    def test_init_with_conditions(
        self,
        mock_on_context_condition: OnContextCondition,
        mock_on_condition: OnCondition,
        mock_after_work: TransitionTarget,
    ) -> None:
        """Test initialization with conditions."""
        handoffs = Handoffs(
            context_conditions=[mock_on_context_condition],
            llm_conditions=[mock_on_condition],
            after_work=mock_after_work,
        )
        assert handoffs.context_conditions == [mock_on_context_condition]
        assert handoffs.llm_conditions == [mock_on_condition]
        assert handoffs.after_work == mock_after_work

    def test_add_context_condition(self, mock_on_context_condition: OnContextCondition) -> None:
        """Test adding a single context condition."""
        handoffs = Handoffs()
        result = handoffs.add_context_condition(mock_on_context_condition)

        assert handoffs.context_conditions == [mock_on_context_condition]
        assert result == handoffs  # Method should return self for chaining

    def test_add_context_condition_invalid_type(self) -> None:
        """Test adding an invalid type to add_context_condition raises TypeError."""
        handoffs = Handoffs()

        with pytest.raises(TypeError) as excinfo:
            handoffs.add_context_condition("not a context condition")  # type: ignore

        assert "Expected an OnContextCondition instance" in str(excinfo.value)

    def test_add_context_conditions(self, mock_on_context_condition: OnContextCondition) -> None:
        """Test adding multiple context conditions."""
        handoffs = Handoffs()
        condition1 = mock_on_context_condition

        # Create a second mock condition
        condition2 = MagicMock(spec=OnContextCondition)

        result = handoffs.add_context_conditions([condition1, condition2])

        assert handoffs.context_conditions == [condition1, condition2]
        assert result == handoffs  # Method should return self for chaining

    def test_add_context_conditions_invalid_type(self) -> None:
        """Test adding invalid types to add_context_conditions raises TypeError."""
        handoffs = Handoffs()

        with pytest.raises(TypeError) as excinfo:
            handoffs.add_context_conditions(["not a context condition"])  # type: ignore

        assert "All conditions must be of type OnContextCondition" in str(excinfo.value)

    def test_add_llm_condition(self, mock_on_condition: OnCondition) -> None:
        """Test adding a single LLM condition."""
        handoffs = Handoffs()
        result = handoffs.add_llm_condition(mock_on_condition)

        assert handoffs.llm_conditions == [mock_on_condition]
        assert result == handoffs  # Method should return self for chaining

    def test_add_llm_condition_invalid_type(self) -> None:
        """Test adding an invalid type to add_llm_condition raises TypeError."""
        handoffs = Handoffs()

        with pytest.raises(TypeError) as excinfo:
            handoffs.add_llm_condition("not an llm condition")  # type: ignore

        assert "Expected an OnCondition instance" in str(excinfo.value)

    def test_add_llm_conditions(self, mock_on_condition: OnCondition) -> None:
        """Test adding multiple LLM conditions."""
        handoffs = Handoffs()
        condition1 = mock_on_condition

        # Create a second mock condition
        condition2 = MagicMock(spec=OnCondition)

        result = handoffs.add_llm_conditions([condition1, condition2])

        assert handoffs.llm_conditions == [condition1, condition2]
        assert result == handoffs  # Method should return self for chaining

    def test_add_llm_conditions_invalid_type(self) -> None:
        """Test adding invalid types to add_llm_conditions raises TypeError."""
        handoffs = Handoffs()

        with pytest.raises(TypeError) as excinfo:
            handoffs.add_llm_conditions(["not an llm condition"])  # type: ignore

        assert "All conditions must be of type OnCondition" in str(excinfo.value)

    def test_set_after_work(self, mock_after_work: TransitionTarget) -> None:
        """Test setting an AfterWork condition."""
        handoffs = Handoffs()
        result = handoffs.set_after_work(mock_after_work)

        assert handoffs.after_work == mock_after_work
        assert result == handoffs  # Method should return self for chaining

    def test_set_after_work_invalid_type(self) -> None:
        """Test setting an invalid type as after_work raises TypeError."""
        handoffs = Handoffs()

        with pytest.raises(TypeError) as excinfo:
            handoffs.set_after_work("not a transition target")  # type: ignore

        assert "Expected a TransitionTarget instance" in str(excinfo.value)

    def test_set_after_work_multiple_times(self, mock_after_work: TransitionTarget) -> None:
        """Test that setting after_work multiple times overrides the previous value."""
        handoffs = Handoffs()

        # Create a second mock target
        mock_target2 = MagicMock(spec=TransitionTarget)

        # Set after_work twice
        handoffs.set_after_work(mock_after_work)
        handoffs.set_after_work(mock_target2)

        # Only the second value should be kept
        assert handoffs.after_work == mock_target2

    def test_add_on_context_condition(self, mock_on_context_condition: OnContextCondition) -> None:
        """Test adding an OnContextCondition using the generic add method."""
        handoffs = Handoffs()
        result = handoffs.add(mock_on_context_condition)

        assert handoffs.context_conditions == [mock_on_context_condition]
        assert result == handoffs  # Method should return self for chaining

    def test_add_on_condition(self, mock_on_condition: OnCondition) -> None:
        """Test adding an OnCondition using the generic add method."""
        handoffs = Handoffs()
        result = handoffs.add(mock_on_condition)

        assert handoffs.llm_conditions == [mock_on_condition]
        assert result == handoffs  # Method should return self for chaining

    def test_add_invalid_type(self) -> None:
        """Test adding an invalid type raises TypeError."""
        handoffs = Handoffs()

        with pytest.raises(TypeError) as excinfo:
            handoffs.add("not a valid condition")  # type: ignore[call-overload]

        assert "Unsupported condition type" in str(excinfo.value)

    def test_add_many(self, mock_on_context_condition: OnContextCondition, mock_on_condition: OnCondition) -> None:
        """Test adding multiple conditions using the add_many method."""
        handoffs = Handoffs()
        result = handoffs.add_many([mock_on_context_condition, mock_on_condition])

        assert handoffs.context_conditions == [mock_on_context_condition]
        assert handoffs.llm_conditions == [mock_on_condition]
        assert result == handoffs  # Method should return self for chaining

    def test_add_many_invalid_type(self) -> None:
        """Test adding an invalid type using add_many raises TypeError."""
        handoffs = Handoffs()

        with pytest.raises(TypeError) as excinfo:
            handoffs.add_many(["not a valid condition"])  # type: ignore[list-item]

        assert "Unsupported condition type" in str(excinfo.value)

    def test_add_empty_lists(self) -> None:
        """Test adding empty lists of conditions."""
        handoffs = Handoffs()

        result = handoffs.add_context_conditions([])
        assert handoffs.context_conditions == []
        assert result == handoffs  # Method should return self for chaining

        result = handoffs.add_llm_conditions([])
        assert handoffs.llm_conditions == []
        assert result == handoffs  # Method should return self for chaining

        result = handoffs.add_many([])
        assert handoffs.context_conditions == []
        assert handoffs.llm_conditions == []
        assert result == handoffs  # Method should return self for chaining

    def test_clear(
        self,
        mock_on_context_condition: OnContextCondition,
        mock_on_condition: OnCondition,
        mock_after_work: TransitionTarget,
    ) -> None:
        """Test clearing all conditions."""
        handoffs = Handoffs(
            context_conditions=[mock_on_context_condition],
            llm_conditions=[mock_on_condition],
            after_work=mock_after_work,
        )

        result = handoffs.clear()

        assert handoffs.context_conditions == []
        assert handoffs.llm_conditions == []
        assert handoffs.after_work is None
        assert result == handoffs  # Method should return self for chaining

    def test_adding_after_clear(
        self,
        mock_on_context_condition: OnContextCondition,
        mock_on_condition: OnCondition,
        mock_after_work: TransitionTarget,
    ) -> None:
        """Test adding conditions after clearing."""
        handoffs = Handoffs(
            context_conditions=[mock_on_context_condition],
            llm_conditions=[mock_on_condition],
            after_work=mock_after_work,
        )

        # Clear and then add new conditions
        handoffs.clear()

        new_context_condition = MagicMock(spec=OnContextCondition)
        new_llm_condition = MagicMock(spec=OnCondition)
        new_after_work = MagicMock(spec=TransitionTarget)

        handoffs.add_context_condition(new_context_condition)
        handoffs.add_llm_condition(new_llm_condition)
        handoffs.set_after_work(new_after_work)

        assert handoffs.context_conditions == [new_context_condition]
        assert handoffs.llm_conditions == [new_llm_condition]
        assert handoffs.after_work == new_after_work

    def test_get_llm_conditions_by_target_type(
        self, mock_on_condition: OnCondition, mock_agent_target: AgentTarget
    ) -> None:
        """Test getting LLM conditions by target type."""
        handoffs = Handoffs(llm_conditions=[mock_on_condition])

        result = handoffs.get_llm_conditions_by_target_type(AgentTarget)

        assert result == [mock_on_condition]

        result = handoffs.get_llm_conditions_by_target_type(NestedChatTarget)

        assert result == []

    def test_get_context_conditions_by_target_type(
        self, mock_on_context_condition: OnContextCondition, mock_agent_target: AgentTarget
    ) -> None:
        """Test getting context conditions by target type."""
        handoffs = Handoffs(context_conditions=[mock_on_context_condition])

        result = handoffs.get_context_conditions_by_target_type(AgentTarget)

        assert result == [mock_on_context_condition]

        result = handoffs.get_context_conditions_by_target_type(NestedChatTarget)

        assert result == []

    def test_get_llm_conditions_requiring_wrapping(
        self, mock_on_condition: OnCondition, mock_on_condition_require_wrapping: OnCondition
    ) -> None:
        """Test getting LLM conditions that require wrapping."""
        handoffs = Handoffs(llm_conditions=[mock_on_condition])

        result = handoffs.get_llm_conditions_requiring_wrapping()

        assert result == []

        handoffs = Handoffs(llm_conditions=[mock_on_condition_require_wrapping])

        result = handoffs.get_llm_conditions_requiring_wrapping()

        assert result == [mock_on_condition_require_wrapping]

    def test_get_context_conditions_requiring_wrapping(
        self,
        mock_on_context_condition: OnContextCondition,
        mock_on_context_condition_require_wrapping: OnContextCondition,
    ) -> None:
        """Test getting context conditions that require wrapping."""
        handoffs = Handoffs(context_conditions=[mock_on_context_condition])

        result = handoffs.get_context_conditions_requiring_wrapping()

        assert result == []

        handoffs = Handoffs(context_conditions=[mock_on_context_condition_require_wrapping])

        result = handoffs.get_context_conditions_requiring_wrapping()

        assert result == [mock_on_context_condition_require_wrapping]

    def test_set_llm_function_names(self, mock_on_condition: OnCondition) -> None:
        """Test setting LLM function names."""
        # Create a target with a known normalized name
        mock_target = MagicMock(spec=TransitionTarget)
        mock_target.normalized_name.return_value = "test_target"

        # Update the mock_on_condition's target
        mock_on_condition.target = mock_target

        mock_on_condition_two = copy.copy(mock_on_condition)

        handoffs = Handoffs(llm_conditions=[mock_on_condition, mock_on_condition_two])

        handoffs.set_llm_function_names()

        # Function names should be set with index (1-based)
        assert mock_on_condition.llm_function_name == "transfer_to_test_target_1"

        # Second condition should have index 2
        assert mock_on_condition_two.llm_function_name == "transfer_to_test_target_2"

    def test_set_llm_function_names_empty(self) -> None:
        """Test setting LLM function names with an empty list."""
        handoffs = Handoffs()

        # Should not raise any errors
        handoffs.set_llm_function_names()

        # No changes to verify since the list is empty
        assert handoffs.llm_conditions == []

    def test_set_llm_function_names_complex(self) -> None:
        """Test setting LLM function names with multiple targets of same type."""
        handoffs = Handoffs()

        # Create 3 targets with the same normalized name
        for i in range(3):
            mock_target = MagicMock(spec=TransitionTarget)
            mock_target.normalized_name.return_value = "same_target"

            mock_condition = MagicMock(spec=OnCondition)
            mock_condition.target = mock_target

            handoffs.add_llm_condition(mock_condition)

        handoffs.set_llm_function_names()

        # Check that all function names are unique
        function_names = [condition.llm_function_name for condition in handoffs.llm_conditions]
        assert len(function_names) == 3
        assert function_names == ["transfer_to_same_target_1", "transfer_to_same_target_2", "transfer_to_same_target_3"]

    def test_adding_duplicate_conditions(
        self, mock_on_context_condition: OnContextCondition, mock_on_condition: OnCondition
    ) -> None:
        """Test adding duplicate conditions."""
        handoffs = Handoffs()

        # Add the same condition twice
        handoffs.add_context_condition(mock_on_context_condition)
        handoffs.add_context_condition(mock_on_context_condition)

        # Both should be added (no duplicate detection)
        assert len(handoffs.context_conditions) == 2
        assert handoffs.context_conditions[0] is mock_on_context_condition
        assert handoffs.context_conditions[1] is mock_on_context_condition

        # Similarly for LLM conditions
        handoffs.add_llm_condition(mock_on_condition)
        handoffs.add_llm_condition(mock_on_condition)

        assert len(handoffs.llm_conditions) == 2
        assert handoffs.llm_conditions[0] is mock_on_condition
        assert handoffs.llm_conditions[1] is mock_on_condition

    def test_method_chaining(
        self,
        mock_on_context_condition: OnContextCondition,
        mock_on_condition: OnCondition,
        mock_after_work: TransitionTarget,
    ) -> None:
        """Test method chaining with multiple operations."""
        handoffs = Handoffs()

        # Chain multiple operations
        result = (
            handoffs.add_context_condition(mock_on_context_condition)
            .add_llm_condition(mock_on_condition)
            .set_after_work(mock_after_work)
        )

        # Verify the operations were applied
        assert handoffs.context_conditions == [mock_on_context_condition]
        assert handoffs.llm_conditions == [mock_on_condition]
        assert handoffs.after_work == mock_after_work

        # Verify the result is the handoffs instance for chaining
        assert result == handoffs
