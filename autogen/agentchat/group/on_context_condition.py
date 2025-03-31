# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from pydantic import BaseModel

from ...doc_utils import export_module
from .available_condition import AvailableCondition
from .context_condition import ContextCondition
from .transition_target import TransitionTarget

__all__ = [
    "OnContextCondition",
]


@export_module("autogen")
class OnContextCondition(BaseModel):  # noqa: N801
    """Defines a condition for transitioning to another agent or nested chats using context variables and the ContextExpression class.

    This is for context variable-based condition evaluation (does not use the agent's LLM).

    These are evaluated before the OnCondition and AfterWork conditions.

    Args:
        target (TransitionTarget): The transition (essentially an agent) to hand off to.
        condition (ContextCondition): The context variable based condition for transitioning to the target agent.
        available (AvailableCondition): Optional condition to determine if this OnCondition is included for the LLM to evaluate based on context variables using classes like StringAvailableCondition and ContextExpressionAvailableCondition.
    """

    target: TransitionTarget
    condition: ContextCondition
    available: Optional[AvailableCondition] = None

    def has_target_type(self, target_type: TransitionTarget) -> bool:
        """
        Check if the target type matches the specified type.

        Args:
            target_type (str): The target type to check against

        Returns:
            bool: True if the target type matches, False otherwise
        """
        return isinstance(self.target, target_type)

    def target_requires_wrapping(self) -> bool:
        """
        Check if the target requires wrapping in an agent.

        Returns:
            bool: True if the target requires wrapping, False otherwise
        """
        return self.target.needs_agent_wrapper()
