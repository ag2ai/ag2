# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from autogen.agentchat.group.context_condition import (
    ExpressionContextCondition,
    StringContextCondition,
)
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.utils import ContextExpression


class TestContextCondition:
    def test_protocol_raise_not_implemented(self) -> None:
        """Test that the ContextCondition protocol raises NotImplementedError when implemented without override."""

        # Create a class that implements the protocol but doesn't override evaluate
        class TestImpl:
            def evaluate(self, context_variables):
                raise NotImplementedError("Requires subclasses to implement.")

        impl = TestImpl()
        with pytest.raises(NotImplementedError) as excinfo:
            impl.evaluate(None)
        assert "Requires subclasses to implement" in str(excinfo.value)


class TestStringContextCondition:
    def test_init(self) -> None:
        """Test initialisation with a variable name."""
        var_name = "test_variable"
        condition = StringContextCondition(variable_name=var_name)
        assert condition.variable_name == var_name

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


class TestExpressionContextCondition:
    def test_init(self) -> None:
        """Test initialisation with a ContextExpression."""
        expression = ContextExpression("${var1} and ${var2}")
        condition = ExpressionContextCondition(expression=expression)
        assert condition.expression == expression

    @patch("autogen.agentchat.utils.ContextExpression.evaluate")
    def test_evaluate_calls_expression_evaluate(self, mock_evaluate) -> None:
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
