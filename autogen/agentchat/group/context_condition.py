# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

from pydantic import BaseModel

from ..utils import ContextExpression
from .context_variables import ContextVariables

__all__ = ["ContextCondition", "ExpressionContextCondition", "StringContextCondition"]


class ContextCondition(Protocol):
    """Protocol for conditions evaluated directly using context variables."""

    def evaluate(self, context_variables: ContextVariables) -> bool:
        """Evaluate the condition to a boolean result.

        Args:
            context_variables: The context variables to evaluate against

        Returns:
            Boolean result of the condition evaluation
        """
        raise NotImplementedError("Requires subclasses to implement.")


class StringContextCondition(BaseModel):
    """Simple string-based context condition.

    This condition checks if a named context variable exists and is truthy.
    """

    variable_name: str

    def __init__(self, variable_name: str, **data):
        super().__init__(variable_name=variable_name, **data)

    def evaluate(self, context_variables: ContextVariables) -> bool:
        """Check if the named context variable is truthy.

        Args:
            context_variables: The context variables to check against

        Returns:
            True if the variable exists and is truthy, False otherwise
        """
        return bool(context_variables.get(self.variable_name, False))


class ExpressionContextCondition(BaseModel):
    """Complex expression-based context condition.

    This condition evaluates a ContextExpression against the context variables.
    """

    expression: ContextExpression

    def __init__(self, expression: ContextExpression, **data):
        super().__init__(expression=expression, **data)

    def evaluate(self, context_variables: ContextVariables) -> bool:
        """Evaluate the expression against the context variables.

        Args:
            context_variables: The context variables to evaluate against

        Returns:
            Boolean result of the expression evaluation
        """
        return self.expression.evaluate(context_variables)
