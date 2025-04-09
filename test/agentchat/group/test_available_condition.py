# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.group.available_condition import (
    AvailableCondition,
    ExpressionAvailableCondition,
    StringAvailableCondition,
)
from autogen.agentchat.group.context_expression import ContextExpression
from autogen.agentchat.group.context_variables import ContextVariables


class TestAvailableCondition:
    def test_protocol_raise_not_implemented(self) -> None:
        """Test that the AvailableCondition protocol raises NotImplementedError when implemented without override."""

        # Create a class that implements the protocol but doesn't override is_available
        class TestImpl(AvailableCondition):
            pass

        impl = TestImpl()
        mock_agent = MagicMock(spec="ConversableAgent")
        with pytest.raises(NotImplementedError) as excinfo:
            impl.is_available(mock_agent, [])
        assert "Requires subclasses to implement" in str(excinfo.value)


class TestStringAvailableCondition:
    def test_init(self) -> None:
        """Test initialisation with a context variable name."""
        var_name = "test_variable"
        condition = StringAvailableCondition(context_variable=var_name)
        assert condition.context_variable == var_name

    def test_init_with_extra_data(self) -> None:
        """Test initialisation with extra data."""
        var_name = "test_variable"
        extra_data = {"extra_key": "extra_value"}
        condition = StringAvailableCondition(context_variable=var_name, **extra_data)
        assert condition.context_variable == var_name

        # Pydantic v2 doesn't store extra attributes directly on the model
        # This would have to be checked if using a custom method to store extra attributes

    def test_is_available_with_true_value(self) -> None:
        """Test is_available returns True when the context variable is truthy."""
        var_name = "test_variable"
        condition = StringAvailableCondition(context_variable=var_name)

        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(data={var_name: True})

        result = condition.is_available(mock_agent, [])
        assert result is True

    def test_is_available_with_false_value(self) -> None:
        """Test is_available returns False when the context variable is falsy."""
        var_name = "test_variable"
        condition = StringAvailableCondition(context_variable=var_name)

        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(data={var_name: False})

        result = condition.is_available(mock_agent, [])
        assert result is False

    def test_is_available_with_missing_value(self) -> None:
        """Test is_available returns False when the context variable is missing."""
        var_name = "test_variable"
        condition = StringAvailableCondition(context_variable=var_name)

        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(data={})

        result = condition.is_available(mock_agent, [])
        assert result is False

    def test_is_available_with_none_value(self) -> None:
        """Test is_available returns False when the context variable is None."""
        var_name = "test_variable"
        condition = StringAvailableCondition(context_variable=var_name)

        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(data={var_name: None})

        result = condition.is_available(mock_agent, [])
        assert result is False

    def test_is_available_with_non_bool_value(self) -> None:
        """Test is_available returns appropriate boolean based on the truthy/falsy nature of the value."""
        var_name = "test_variable"
        condition = StringAvailableCondition(context_variable=var_name)

        mock_agent = MagicMock()

        # Test with non-empty string (truthy)
        mock_agent.context_variables = ContextVariables(data={var_name: "value"})
        result = condition.is_available(mock_agent, [])
        assert result is True

        # Test with empty string (falsy)
        mock_agent.context_variables = ContextVariables(data={var_name: ""})
        result = condition.is_available(mock_agent, [])
        assert result is False

        # Test with 1 (truthy)
        mock_agent.context_variables = ContextVariables(data={var_name: 1})
        result = condition.is_available(mock_agent, [])
        assert result is True

        # Test with 0 (falsy)
        mock_agent.context_variables = ContextVariables(data={var_name: 0})
        result = condition.is_available(mock_agent, [])
        assert result is False

    def test_is_available_with_collection_values(self) -> None:
        """Test is_available with collection types."""
        var_name = "test_variable"
        condition = StringAvailableCondition(context_variable=var_name)

        mock_agent = MagicMock()

        # Test with non-empty list (truthy)
        mock_agent.context_variables = ContextVariables(data={var_name: [1, 2, 3]})
        result = condition.is_available(mock_agent, [])
        assert result is True

        # Test with empty list (falsy)
        mock_agent.context_variables = ContextVariables(data={var_name: []})
        result = condition.is_available(mock_agent, [])
        assert result is False

        # Test with non-empty dict (truthy)
        mock_agent.context_variables = ContextVariables(data={var_name: {"key": "value"}})
        result = condition.is_available(mock_agent, [])
        assert result is True

        # Test with empty dict (falsy)
        mock_agent.context_variables = ContextVariables(data={var_name: {}})
        result = condition.is_available(mock_agent, [])
        assert result is False

    def test_messages_parameter_ignored(self) -> None:
        """Test that the messages parameter is ignored."""
        var_name = "test_variable"
        condition = StringAvailableCondition(context_variable=var_name)

        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(data={var_name: True})

        # Different messages should produce the same result
        messages1 = [{"content": "message1"}]
        messages2 = [{"content": "message2"}]

        result1 = condition.is_available(mock_agent, messages1)
        result2 = condition.is_available(mock_agent, messages2)

        assert result1 is True
        assert result2 is True


class TestContextExpressionAvailableCondition:
    def test_init(self) -> None:
        """Test initialisation with a ContextExpression."""
        expression = ContextExpression("${var1} and ${var2}")
        condition = ExpressionAvailableCondition(expression=expression)
        assert condition.expression == expression

    def test_init_with_extra_data(self) -> None:
        """Test initialisation with extra data."""
        expression = ContextExpression("${var1} and ${var2}")
        extra_data = {"extra_key": "extra_value"}
        condition = ExpressionAvailableCondition(expression=expression, **extra_data)
        assert condition.expression == expression

        # Pydantic v2 doesn't store extra attributes directly on the model
        # This would have to be checked if using a custom method to store extra attributes

    @patch("autogen.agentchat.group.context_expression.ContextExpression.evaluate")
    def test_is_available_calls_expression_evaluate(self, mock_evaluate: MagicMock) -> None:
        """Test is_available calls the expression's evaluate method with the agent's context variables."""
        expression = ContextExpression("${var1} and ${var2}")
        condition = ExpressionAvailableCondition(expression=expression)

        mock_agent = MagicMock()
        mock_context_vars = ContextVariables(data={"var1": True, "var2": False})
        mock_agent.context_variables = mock_context_vars

        mock_evaluate.return_value = True

        result = condition.is_available(mock_agent, [])

        mock_evaluate.assert_called_once_with(mock_context_vars)
        assert result is True

    def test_is_available_with_true_expression(self) -> None:
        """Test is_available returns True when the expression evaluates to True."""
        expression = ContextExpression("${var1} and ${var2}")
        condition = ExpressionAvailableCondition(expression=expression)

        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(data={"var1": True, "var2": True})

        result = condition.is_available(mock_agent, [])
        assert result is True

    def test_is_available_with_false_expression(self) -> None:
        """Test is_available returns False when the expression evaluates to False."""
        expression = ContextExpression("${var1} and ${var2}")
        condition = ExpressionAvailableCondition(expression=expression)

        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(data={"var1": True, "var2": False})

        result = condition.is_available(mock_agent, [])
        assert result is False

    def test_is_available_with_complex_expression(self) -> None:
        """Test is_available with a more complex expression."""
        expression = ContextExpression("(${var1} or ${var2}) and not ${var3}")
        condition = ExpressionAvailableCondition(expression=expression)

        mock_agent = MagicMock()

        # Test case: (True or False) and not False = True
        mock_agent.context_variables = ContextVariables(data={"var1": True, "var2": False, "var3": False})
        result = condition.is_available(mock_agent, [])
        assert result is True

        # Test case: (True or False) and not True = False
        mock_agent.context_variables = ContextVariables(data={"var1": True, "var2": False, "var3": True})
        result = condition.is_available(mock_agent, [])
        assert result is False

        # Test case: (False or False) and not False = False
        mock_agent.context_variables = ContextVariables(data={"var1": False, "var2": False, "var3": False})
        result = condition.is_available(mock_agent, [])
        assert result is False

    def test_is_available_with_missing_variables(self) -> None:
        """Test is_available when variables are missing from context."""
        expression = ContextExpression("${var1} and ${var2}")
        condition = ExpressionAvailableCondition(expression=expression)

        mock_agent = MagicMock()
        # var2 is missing
        mock_agent.context_variables = ContextVariables(data={"var1": True})

        with pytest.raises(KeyError):
            condition.is_available(mock_agent, [])

    def test_is_available_with_nested_variables(self) -> None:
        """Test is_available with a nested context expression."""
        # This would typically fail with the current implementation
        # as nested lookups are not supported directly by ContextExpression
        expression_string = "${user.is_premium} and ${user.status} == 'active'"
        with pytest.raises(ValueError, match="Operation type Attribute is not allowed"):
            ContextExpression(expression_string)

    def test_messages_parameter_ignored(self) -> None:
        """Test that the messages parameter is ignored."""
        expression = ContextExpression("${var1} and ${var2}")
        condition = ExpressionAvailableCondition(expression=expression)

        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(data={"var1": True, "var2": True})

        # Different messages should produce the same result
        messages1 = [{"content": "message1"}]
        messages2 = [{"content": "message2"}]

        result1 = condition.is_available(mock_agent, messages1)
        result2 = condition.is_available(mock_agent, messages2)

        assert result1 is True
        assert result2 is True

    def test_is_available_with_comparison_operators(self) -> None:
        """Test is_available with comparison operators."""
        mock_agent = MagicMock()

        # Test greater than
        expression = ContextExpression("${count} > 10")
        condition = ExpressionAvailableCondition(expression=expression)

        mock_agent.context_variables = ContextVariables(data={"count": 15})
        result = condition.is_available(mock_agent, [])
        assert result is True

        mock_agent.context_variables = ContextVariables(data={"count": 5})
        result = condition.is_available(mock_agent, [])
        assert result is False

        # Test less than
        expression = ContextExpression("${count} < 10")
        condition = ExpressionAvailableCondition(expression=expression)

        mock_agent.context_variables = ContextVariables(data={"count": 5})
        result = condition.is_available(mock_agent, [])
        assert result is True

        mock_agent.context_variables = ContextVariables(data={"count": 15})
        result = condition.is_available(mock_agent, [])
        assert result is False

        # Test equality
        expression = ContextExpression("${status} == 'active'")
        condition = ExpressionAvailableCondition(expression=expression)

        mock_agent.context_variables = ContextVariables(data={"status": "active"})
        result = condition.is_available(mock_agent, [])
        assert result is True

        mock_agent.context_variables = ContextVariables(data={"status": "inactive"})
        result = condition.is_available(mock_agent, [])
        assert result is False

        # Test inequality
        expression = ContextExpression("${status} != 'active'")
        condition = ExpressionAvailableCondition(expression=expression)

        mock_agent.context_variables = ContextVariables(data={"status": "inactive"})
        result = condition.is_available(mock_agent, [])
        assert result is True

        mock_agent.context_variables = ContextVariables(data={"status": "active"})
        result = condition.is_available(mock_agent, [])
        assert result is False


class TestAvailableConditionIntegration:
    def test_string_condition_with_real_agent(self) -> None:
        """Test StringAvailableCondition with a more realistic agent setup."""
        # Create a more realistic mock agent
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.context_variables = ContextVariables()

        # Initialise condition
        condition = StringAvailableCondition(context_variable="feature_flag_enabled")

        # Update context in agent
        mock_agent.context_variables.set("feature_flag_enabled", False)
        assert condition.is_available(mock_agent, []) is False

        # Update context again
        mock_agent.context_variables.set("feature_flag_enabled", True)
        assert condition.is_available(mock_agent, []) is True

        # Remove the variable
        mock_agent.context_variables.remove("feature_flag_enabled")
        assert condition.is_available(mock_agent, []) is False

    def test_context_expression_condition_with_real_agent(self) -> None:
        """Test ContextExpressionAvailableCondition with a more realistic agent setup."""
        # Create a more realistic mock agent
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.context_variables = ContextVariables()

        # Initialise condition
        expression = ContextExpression("${is_premium} and ${login_count} > 5")
        condition = ExpressionAvailableCondition(expression=expression)

        # Set initial context (not enough for condition to be true)
        mock_agent.context_variables.set("is_premium", True)
        mock_agent.context_variables.set("login_count", 3)
        assert condition.is_available(mock_agent, []) is False

        # Update context to make condition true
        mock_agent.context_variables.set("login_count", 10)
        assert condition.is_available(mock_agent, []) is True

        # Update context to make condition false again
        mock_agent.context_variables.set("is_premium", False)
        assert condition.is_available(mock_agent, []) is False
