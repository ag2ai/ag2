# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

from ...doc_utils import export_module
from .available_condition import AvailableCondition
from .context_condition import ContextCondition
from .transition_target import TransitionTarget

__all__ = [
    "OnContextCondition",
]


@dataclass
@export_module("autogen")
class OnContextCondition:  # noqa: N801
    """Defines a condition for transitioning to another agent or nested chats using context variables and the ContextExpression class.

    This is for context variable-based condition evaluation (does not use the agent's LLM).

    These are evaluated before the OnCondition and AfterWork conditions.

    Args:
        target (Optional[Union[ConversableAgent, dict[str, Any]]]): The agent to hand off to or the nested chat configuration. Can be a ConversableAgent or a Dict.
            If a Dict, it should follow the convention of the nested chat configuration, with the exception of a carryover configuration which is unique to Swarms.
            Swarm Nested chat documentation: https://docs.ag2.ai/docs/user-guide/advanced-concepts/swarm-deep-dive#registering-handoffs-to-a-nested-chat
        condition (Optional[Union[str, ContextExpression]]): The condition for transitioning to the target agent, evaluated by the LLM.
            If a string, it needs to represent a context variable key and the value will be evaluated as a boolean
            If a ContextExpression, it will evaluate the logical expression against the context variables. If it is True, the transition will occur.
                Can use not, and, or, and comparison operators (>, <, >=, <=, ==, !=).
                Example: ContextExpression("not(${logged_in} and ${is_admin}) or (${guest_checkout})")
                Example with comparison: ContextExpression("${attempts} >= 3 or ${is_premium} == True or ${tier} == 'gold'")
        available (Optional[Union[Callable[[ConversableAgent, list[dict[str, Any]]], bool], str, ContextExpression]]): Optional condition to determine if this OnContextCondition is included for the LLM to evaluate.
            If a string, it will look up the value of the context variable with that name, which should be a bool, to determine whether it should include this condition.
            If a ContextExpression, it will evaluate the logical expression against the context variables. Can use not, and, or, and comparison operators (>, <, >=, <=, ==, !=).
            The Callable signature is:
                def my_available_func(agent: ConversableAgent, messages: list[Dict[str, Any]]) -> bool

    """

    target: TransitionTarget
    condition: ContextCondition
    available: Optional[AvailableCondition] = None

    """
    def __post_init__(self) -> None:
        # Ensure valid types
        if (self.target is not None) and (not isinstance(self.target, (ConversableAgent, dict))):
            raise ValueError("'target' must be a ConversableAgent or a dict")

        # Ensure they have a condition
        if isinstance(self.condition, str):
            if not self.condition.strip():
                raise ValueError("'condition' must be a non-empty string")

            self._context_condition = ContextExpression("${" + self.condition + "}")
        else:
            if not isinstance(self.condition, ContextExpression):
                raise ValueError("'condition' must be a string on ContextExpression")

            self._context_condition = self.condition

        if (self.available is not None) and (
            not (isinstance(self.available, (str, ContextExpression)) or callable(self.available))
        ):
            raise ValueError("'available' must be a callable, a string, or a ContextExpression")
    """
