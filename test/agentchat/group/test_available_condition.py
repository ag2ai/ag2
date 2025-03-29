# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.group.available_condition import (
    ContextExpressionAvailableCondition,
    StringAvailableCondition,
)
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.utils import ContextExpression


class TestAvailableCondition:
    def test_protocol_raise_not_implemented(self) -> None:
        """Test that the AvailableCondition protocol raises NotImplementedError when implemented without override."""

        # Create a class that implements the protocol but doesn't override is_available
        class TestImpl:
            def is_available(self, agent, messages):
                raise NotImplementedError("Requires subclasses to implement.")

        impl = TestImpl()
        with pytest.raises(NotImplementedError) as excinfo:
            impl.is_available(None, [])
        assert "Requires subclasses to implement" in str(excinfo.value)


class TestStringAvailableCondition:
    def test_init(self) -> None:
        """Test initialisation with a context variable name."""
        var_name = "test_variable"
        condition = StringAvailableCondition(context_variable=var_name)
        assert condition.context_variable == var_name

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


class TestContextExpressionAvailableCondition:
    def test_init(self) -> None:
        """Test initialisation with a ContextExpression."""
        expression = ContextExpression("${var1} and ${var2}")
        condition = ContextExpressionAvailableCondition(expression=expression)
        assert condition.expression == expression

    @patch("autogen.agentchat.utils.ContextExpression.evaluate")
    def test_is_available_calls_expression_evaluate(self, mock_evaluate) -> None:
        """Test is_available calls the expression's evaluate method with the agent's context variables."""
        expression = ContextExpression("${var1} and ${var2}")
        condition = ContextExpressionAvailableCondition(expression=expression)

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
        condition = ContextExpressionAvailableCondition(expression=expression)

        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(data={"var1": True, "var2": True})

        result = condition.is_available(mock_agent, [])
        assert result is True

    def test_is_available_with_false_expression(self) -> None:
        """Test is_available returns False when the expression evaluates to False."""
        expression = ContextExpression("${var1} and ${var2}")
        condition = ContextExpressionAvailableCondition(expression=expression)

        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(data={"var1": True, "var2": False})

        result = condition.is_available(mock_agent, [])
        assert result is False

    def test_is_available_with_complex_expression(self) -> None:
        """Test is_available with a more complex expression."""
        expression = ContextExpression("(${var1} or ${var2}) and not ${var3}")
        condition = ContextExpressionAvailableCondition(expression=expression)

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
