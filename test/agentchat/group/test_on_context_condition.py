# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from autogen.agentchat.group.available_condition import StringAvailableCondition
from autogen.agentchat.group.context_condition import ExpressionContextCondition, StringContextCondition
from autogen.agentchat.group.context_expression import ContextExpression
from autogen.agentchat.group.on_context_condition import OnContextCondition
from autogen.agentchat.group.targets.transition_target import AgentTarget, TransitionTarget


class TestOnContextCondition:
    def test_init(self) -> None:
        """Test initialisation with basic values."""
        target = MagicMock(spec=TransitionTarget)
        condition = MagicMock(spec=StringContextCondition)
        available = MagicMock(spec=StringAvailableCondition)

        on_context_condition = OnContextCondition(target=target, condition=condition, available=available)

        assert on_context_condition.target == target
        assert on_context_condition.condition == condition
        assert on_context_condition.available == available

    def test_init_with_none_available(self) -> None:
        """Test initialisation with None available."""
        target = MagicMock(spec=TransitionTarget)
        condition = MagicMock(spec=StringContextCondition)

        on_context_condition = OnContextCondition(target=target, condition=condition, available=None)

        assert on_context_condition.target == target
        assert on_context_condition.condition == condition
        assert on_context_condition.available is None

    def test_init_with_string_context_condition(self) -> None:
        """Test initialisation with StringContextCondition."""
        target = MagicMock(spec=TransitionTarget)
        condition = StringContextCondition(variable_name="is_valid")

        on_context_condition = OnContextCondition(target=target, condition=condition)

        assert on_context_condition.target == target
        assert on_context_condition.condition == condition
        assert on_context_condition.condition is not None
        assert isinstance(on_context_condition.condition, StringContextCondition)
        assert on_context_condition.condition.variable_name == "is_valid"

    def test_init_with_expression_context_condition(self) -> None:
        """Test initialisation with ExpressionContextCondition."""
        target = MagicMock(spec=TransitionTarget)
        expression = ContextExpression("${var1} and ${var2}")
        condition = ExpressionContextCondition(expression=expression)

        on_context_condition = OnContextCondition(target=target, condition=condition)

        assert on_context_condition.target == target
        assert on_context_condition.condition == condition
        assert on_context_condition.condition is not None
        assert isinstance(on_context_condition.condition, ExpressionContextCondition)
        assert on_context_condition.condition.expression == expression

    def test_init_with_agent_target(self) -> None:
        """Test initialisation with AgentTarget."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        condition = StringContextCondition(variable_name="is_valid")

        on_context_condition = OnContextCondition(target=target, condition=condition)

        assert on_context_condition.target == target
        assert on_context_condition.target is not None
        assert isinstance(on_context_condition.target, AgentTarget)
        assert on_context_condition.target.agent_name == "test_agent"

    def test_init_with_string_available_condition(self) -> None:
        """Test initialisation with string available condition."""
        target = MagicMock(spec=TransitionTarget)
        condition = StringContextCondition(variable_name="is_valid")
        available = StringAvailableCondition(context_variable="is_available")

        on_context_condition = OnContextCondition(target=target, condition=condition, available=available)

        assert on_context_condition.available == available
        assert on_context_condition.available is not None
        assert isinstance(on_context_condition.available, StringAvailableCondition)
        assert on_context_condition.available.context_variable == "is_available"

    def test_init_with_context_expression_available(self) -> None:
        """Test initialisation with ContextExpression available."""
        target = MagicMock(spec=TransitionTarget)
        condition = StringContextCondition(variable_name="is_valid")
        available = StringAvailableCondition(context_variable="is_logged_in")

        on_context_condition = OnContextCondition(target=target, condition=condition, available=available)

        assert on_context_condition.available == available

    def test_condition_evaluate(self) -> None:
        """Test that condition.evaluate is called correctly."""
        target = MagicMock(spec=TransitionTarget)
        condition = StringContextCondition(variable_name="is_valid")

        on_context_condition = OnContextCondition(target=target, condition=condition)

        mock_context_variables = MagicMock()

        # Call evaluate through the condition
        result = on_context_condition.condition.evaluate(mock_context_variables)

        # Verify the mock was called correctly
        assert result is True

    def test_available_is_available(self) -> None:
        """Test that available.is_available is called correctly."""
        target = MagicMock(spec=TransitionTarget)
        condition = StringContextCondition(variable_name="Test Variable")
        available = StringAvailableCondition(context_variable="is_logged_in")

        on_context_condition = OnContextCondition(target=target, condition=condition, available=available)

        mock_agent = MagicMock()
        messages = [{"role": "user", "content": "Hello"}]

        # Call is_available through the available
        assert on_context_condition.available is not None
        result = on_context_condition.available.is_available(mock_agent, messages)

        assert result is True
