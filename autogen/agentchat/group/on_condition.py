# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

from ...doc_utils import export_module
from .available_condition import AvailableCondition
from .llm_condition import LLMCondition
from .transition_target import TransitionTarget

__all__ = [
    "OnCondition",
]


@dataclass
@export_module("autogen")
class OnCondition:  # noqa: N801
    """Defines a condition for transitioning to another agent or nested chats.

    This is for LLM-based condition evaluation where these conditions are translated into tools and attached to the agent.

    These are evaluated after the OnCondition conditions but before the AfterWork conditions.

    Args:
        target (Optional[Union[ConversableAgent, dict[str, Any]]]): The agent to hand off to or the nested chat configuration. Can be a ConversableAgent or a Dict.
            If a Dict, it should follow the convention of the nested chat configuration, with the exception of a carryover configuration which is unique to Swarms.
            Swarm Nested chat documentation: https://docs.ag2.ai/docs/user-guide/advanced-concepts/swarm-deep-dive#registering-handoffs-to-a-nested-chat
        condition (Optional[Union[str, ContextStr, Callable[[ConversableAgent, list[dict[str, Any]]], str]]]): The condition for transitioning to the target agent, evaluated by the LLM.
            If a string or Callable, no automatic context variable substitution occurs.
            If a ContextStr, context variable substitution occurs.
            The Callable signature is:
                def my_condition_string(agent: ConversableAgent, messages: list[Dict[str, Any]]) -> str
        available (Optional[Union[Callable[[ConversableAgent, list[dict[str, Any]]], bool], str, ContextExpression]]): Optional condition to determine if this OnCondition is included for the LLM to evaluate.
            If a string, it will look up the value of the context variable with that name, which should be a bool, to determine whether it should include this condition.
            If a ContextExpression, it will evaluate the logical expression against the context variables. Can use not, and, or, and comparison operators (>, <, >=, <=, ==, !=).
                Example: ContextExpression("not(${logged_in} and ${is_admin}) or (${guest_checkout})")
                Example with comparison: ContextExpression("${attempts} >= 3 or ${is_premium} == True or ${tier} == 'gold'")
            The Callable signature is:
                def my_available_func(agent: ConversableAgent, messages: list[Dict[str, Any]]) -> bool
    """

    target: TransitionTarget
    condition: LLMCondition
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
        else:
            if not isinstance(self.condition, ContextStr) and not callable(self.condition):
                raise ValueError("'condition' must be a string, ContextStr, or callable")

        if (self.available is not None) and (
            not (isinstance(self.available, (str, ContextExpression)) or callable(self.available))
        ):
            raise ValueError("'available' must be a callable, a string, or a ContextExpression")'
    """
