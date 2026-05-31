# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.group.context_condition import (
    ContextCondition,
    ExpressionContextCondition,
    StringContextCondition,
)
from autogen.agentchat.group.context_expression import ContextExpression
from autogen.agentchat.group.context_variables import ContextVariables


class TestContextCondition:
    def test_protocol_raise_not_implemented(self) -> None:
        """Test that the ContextCondition protocol raises NotImplementedError when implemented without override."""

        # Create a class that implements the protocol but doesn't override evaluate
        class TestImpl(ContextCondition):
            pass

        impl = TestImpl()
        mock_cv = MagicMock(spec=ContextVariables)
        with pytest.raises(NotImplementedError) as excinfo:
            impl.evaluate(mock_cv)
        assert "Requires subclasses to implement" in str(excinfo.value)


class TestStringContextCondition:
    def test_init(self) -> None:
        """Test initialisation with a variable name."""
        var_name = "test_variable"
        condition = StringContextCondition(variable_name=var_name)
        assert condition.variable_name == var_name

    def test_init_with_extra_data(self) -> None:
        """Test initialisation with extra data."""
        var_name = "test_variable"
        extra_data = {"extra_key": "extra_value"}
        condition = StringContextCondition(variable_name=var_name, **extra_data)
        assert condition.variable_name == var_name

        # Pydantic v2 doesn't store extra attributes directly on the model
        # This would have to be checked if using a custom method to store extra attributes

    def test_evaluate_with_true_value(self) -> None:
        """Test evaluate returns True when the variable is truthy."""
        var_name = "test_variable"
        condition = StringContextCondition(variable_name=var_name)

        context_vars = ContextVariables(data={var_name: True})
        result = condition.evaluate(context_vars)
        assert result is True

    def test_evaluate_with_false_value(self) -> None:
        """Test evaluate returns False when the variable is falsy."""
        var_name = "test_variable"
        condition = StringContextCondition(variable_name=var_name)

        context_vars = ContextVariables(data={var_name: False})
        result = condition.evaluate(context_vars)
        assert result is False

    def test_evaluate_with_missing_value(self) -> None:
        """Test evaluate returns False when the variable is missing."""
        var_name = "test_variable"
        condition = StringContextCondition(variable_name=var_name)

        context_vars = ContextVariables(data={})
        result = condition.evaluate(context_vars)
        assert result is False

    def test_evaluate_with_none_value(self) -> None:
        """Test evaluate returns False when the variable is None."""
        var_name = "test_variable"
        condition = StringContextCondition(variable_name=var_name)

        context_vars = ContextVariables(data={var_name: None})
        result = condition.evaluate(context_vars)
        assert result is False

    def test_evaluate_with_non_bool_value(self) -> None:
        """Test evaluate returns appropriate boolean based on the truthy/falsy nature of the value."""
        var_name = "test_variable"
        condition = StringContextCondition(variable_name=var_name)

        # Test with non-empty string (truthy)
        context_vars = ContextVariables(data={var_name: "value"})
        result = condition.evaluate(context_vars)
        assert result is True

        # Test with empty string (falsy)
        context_vars = ContextVariables(data={var_name: ""})
        result = condition.evaluate(context_vars)
        assert result is False

        # Test with 1 (truthy)
        context_vars = ContextVariables(data={var_name: 1})
        result = condition.evaluate(context_vars)
        assert result is True

        # Test with 0 (falsy)
        context_vars = ContextVariables(data={var_name: 0})
        result = condition.evaluate(context_vars)
        assert result is False

    def test_evaluate_with_collection_values(self) -> None:
        """Test evaluate with collection types."""
        var_name = "test_variable"
        condition = StringContextCondition(variable_name=var_name)

        # Test with non-empty list (truthy)
        context_vars = ContextVariables(data={var_name: [1, 2, 3]})
        result = condition.evaluate(context_vars)
        assert result is True

        # Test with empty list (falsy)
        context_vars = ContextVariables(data={var_name: []})
        result = condition.evaluate(context_vars)
        assert result is False

        # Test with non-empty dict (truthy)
        context_vars = ContextVariables(data={var_name: {"key": "value"}})
        result = condition.evaluate(context_vars)
        assert result is True

        # Test with empty dict (falsy)
        context_vars = ContextVariables(data={var_name: {}})
        result = condition.evaluate(context_vars)
        assert result is False


class TestExpressionContextCondition:
    def test_init(self) -> None:
        """Test initialisation with a ContextExpression."""
        expression = ContextExpression("${var1} and ${var2}")
        condition = ExpressionContextCondition(expression=expression)
        assert condition.expression == expression

    def test_init_with_extra_data(self) -> None:
        """Test initialisation with extra data."""
        expression = ContextExpression("${var1} and ${var2}")
        extra_data = {"extra_key": "extra_value"}
        condition = ExpressionContextCondition(expression=expression, **extra_data)
        assert condition.expression == expression

    @patch("autogen.agentchat.group.context_expression.ContextExpression.evaluate")
    def test_evaluate_calls_expression_evaluate(self, mock_evaluate: MagicMock) -> None:
        """Test evaluate calls the expression's evaluate method with the context variables."""
        expression = ContextExpression("${var1} and ${var2}")
        condition = ExpressionContextCondition(expression=expression)

        context_vars = ContextVariables(data={"var1": True, "var2": False})
        mock_evaluate.return_value = True

        result = condition.evaluate(context_vars)

        mock_evaluate.assert_called_once_with(context_vars)
        assert result is True

    def test_evaluate_with_true_expression(self) -> None:
        """Test evaluate returns True when the expression evaluates to True."""
        expression = ContextExpression("${var1} and ${var2}")
        condition = ExpressionContextCondition(expression=expression)

        context_vars = ContextVariables(data={"var1": True, "var2": True})
        result = condition.evaluate(context_vars)
        assert result is True

    def test_evaluate_with_false_expression(self) -> None:
        """Test evaluate returns False when the expression evaluates to False."""
        expression = ContextExpression("${var1} and ${var2}")
        condition = ExpressionContextCondition(expression=expression)

        context_vars = ContextVariables(data={"var1": True, "var2": False})
        result = condition.evaluate(context_vars)
        assert result is False

    def test_evaluate_with_complex_expression(self) -> None:
        """Test evaluate with a more complex expression."""
        expression = ContextExpression("(${var1} or ${var2}) and not ${var3}")
        condition = ExpressionContextCondition(expression=expression)

        # Test case: (True or False) and not False = True
        context_vars = ContextVariables(data={"var1": True, "var2": False, "var3": False})
        result = condition.evaluate(context_vars)
        assert result is True

        # Test case: (True or False) and not True = False
        context_vars = ContextVariables(data={"var1": True, "var2": False, "var3": True})
        result = condition.evaluate(context_vars)
        assert result is False

        # Test case: (False or False) and not False = False
        context_vars = ContextVariables(data={"var1": False, "var2": False, "var3": False})
        result = condition.evaluate(context_vars)
        assert result is False

    def test_evaluate_with_missing_variables_raises_keyerror(self) -> None:
        """Test that missing variables raise a KeyError as expected."""
        # When a variable is missing, KeyError should be raised
        expression = ContextExpression("${var1}")
        condition = ExpressionContextCondition(expression=expression)

        # Create empty context
        context_vars = ContextVariables(data={})

        # This should raise KeyError now that we've fixed the implementation
        with pytest.raises(KeyError) as excinfo:
            condition.evaluate(context_vars)

        assert "Missing context variable: 'var1'" in str(excinfo.value)

        # Test with a more complex expression
        expression = ContextExpression("${var1} and ${var2}")
        condition = ExpressionContextCondition(expression=expression)

        # Only var1 is present
        context_vars = ContextVariables(data={"var1": True})

        # Should raise KeyError for missing var2
        with pytest.raises(KeyError) as excinfo:
            condition.evaluate(context_vars)

        assert "Missing context variable: 'var2'" in str(excinfo.value)

    def test_nested_variable_not_supported(self) -> None:
        """Test that nested variable notation is not currently supported."""
        # ContextExpression doesn't support dot notation for nested attributes

        # Attempting to create a ContextExpression with dot notation will fail
        with pytest.raises(ValueError) as excinfo:
            expression = ContextExpression("${user.is_premium}")

        # Verify the specific error mentions Attribute is not allowed
        assert "Attribute is not allowed" in str(excinfo.value)

        # Alternative approach using flattened variables
        expression = ContextExpression("${user_is_premium} and ${user_status} == 'active'")
        condition = ExpressionContextCondition(expression=expression)

        context_vars = ContextVariables(data={"user_is_premium": True, "user_status": "active"})

        # This should work with flattened variable names
        result = condition.evaluate(context_vars)
        assert result is True

    def test_evaluate_with_comparison_operators(self) -> None:
        """Test evaluate with comparison operators."""
        # Test greater than
        expression = ContextExpression("${count} > 10")
        condition = ExpressionContextCondition(expression=expression)

        context_vars = ContextVariables(data={"count": 15})
        result = condition.evaluate(context_vars)
        assert result is True

        context_vars = ContextVariables(data={"count": 5})
        result = condition.evaluate(context_vars)
        assert result is False

        # Test less than
        expression = ContextExpression("${count} < 10")
        condition = ExpressionContextCondition(expression=expression)

        context_vars = ContextVariables(data={"count": 5})
        result = condition.evaluate(context_vars)
        assert result is True

        context_vars = ContextVariables(data={"count": 15})
        result = condition.evaluate(context_vars)
        assert result is False

        # Test equality
        expression = ContextExpression("${status} == 'active'")
        condition = ExpressionContextCondition(expression=expression)

        context_vars = ContextVariables(data={"status": "active"})
        result = condition.evaluate(context_vars)
        assert result is True

        context_vars = ContextVariables(data={"status": "inactive"})
        result = condition.evaluate(context_vars)
        assert result is False

        # Test inequality
        expression = ContextExpression("${status} != 'active'")
        condition = ExpressionContextCondition(expression=expression)

        context_vars = ContextVariables(data={"status": "inactive"})
        result = condition.evaluate(context_vars)
        assert result is True

        context_vars = ContextVariables(data={"status": "active"})
        result = condition.evaluate(context_vars)
        assert result is False

    def test_evaluate_with_numeric_comparisons(self) -> None:
        """Test evaluate with numeric comparisons."""
        # Test greater than or equal
        expression = ContextExpression("${count} >= 10")
        condition = ExpressionContextCondition(expression=expression)

        context_vars = ContextVariables(data={"count": 10})
        result = condition.evaluate(context_vars)
        assert result is True

        context_vars = ContextVariables(data={"count": 9})
        result = condition.evaluate(context_vars)
        assert result is False

        # Test less than or equal
        expression = ContextExpression("${count} <= 10")
        condition = ExpressionContextCondition(expression=expression)

        context_vars = ContextVariables(data={"count": 10})
        result = condition.evaluate(context_vars)
        assert result is True

        context_vars = ContextVariables(data={"count": 11})
        result = condition.evaluate(context_vars)
        assert result is False


class TestContextConditionIntegration:
    def test_string_condition_with_updates(self) -> None:
        """Test StringContextCondition with dynamic context updates."""
        # Initialise condition
        condition = StringContextCondition(variable_name="feature_flag_enabled")

        # Initialise context variables
        context_vars = ContextVariables()

        # Test with missing variable
        assert condition.evaluate(context_vars) is False

        # Add variable and test
        context_vars.set("feature_flag_enabled", False)
        assert condition.evaluate(context_vars) is False

        # Update variable and test
        context_vars.set("feature_flag_enabled", True)
        assert condition.evaluate(context_vars) is True

        # Remove variable and test
        context_vars.remove("feature_flag_enabled")
        assert condition.evaluate(context_vars) is False

    def test_expression_condition_with_updates(self) -> None:
        """Test ExpressionContextCondition with dynamic context updates."""
        # Initialise condition
        expression = ContextExpression("${is_premium} and ${login_count} > 5")
        condition = ExpressionContextCondition(expression=expression)

        # Initialise context variables with all required variables
        context_vars = ContextVariables()
        context_vars.set("is_premium", True)
        context_vars.set("login_count", 3)

        # Initial condition should be false (login_count too low)
        assert condition.evaluate(context_vars) is False

        # Update login_count to make condition true
        context_vars.set("login_count", 10)
        assert condition.evaluate(context_vars) is True

        # Update is_premium to make condition false again
        context_vars.set("is_premium", False)
        assert condition.evaluate(context_vars) is False

        # Now if we remove a variable, it should raise KeyError
        context_vars.remove("login_count")
        with pytest.raises(KeyError) as excinfo:
            condition.evaluate(context_vars)
        assert "Missing context variable: 'login_count'" in str(excinfo.value)

    def test_combine_conditions(self) -> None:
        """Test combining multiple conditions in a custom way."""
        # Create conditions
        string_condition = StringContextCondition(variable_name="is_admin")

        # For ExpressionContextCondition, we need to ensure all variables exist
        expression_condition = ExpressionContextCondition(expression=ContextExpression("${age} >= 18"))

        # Initialise context variables
        context_vars = ContextVariables(data={"is_admin": True, "age": 25})

        # Test both conditions individually
        assert string_condition.evaluate(context_vars) is True
        assert expression_condition.evaluate(context_vars) is True

        # Combine conditions with AND logic
        combined_and_result = string_condition.evaluate(context_vars) and expression_condition.evaluate(context_vars)
        assert combined_and_result is True

        # Update context to make one condition false
        context_vars.set("age", 15)

        # Test conditions again
        assert string_condition.evaluate(context_vars) is True
        assert expression_condition.evaluate(context_vars) is False

        # Combine conditions with AND logic
        combined_and_result = string_condition.evaluate(context_vars) and expression_condition.evaluate(context_vars)
        assert combined_and_result is False

        # Combine conditions with OR logic
        combined_or_result = string_condition.evaluate(context_vars) or expression_condition.evaluate(context_vars)
        assert combined_or_result is True
