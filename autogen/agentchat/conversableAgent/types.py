# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from inspect import signature
from typing import TYPE_CHECKING, Any

from ...doc_utils import export_module

if TYPE_CHECKING:
    from .conversable_agent import ConversableAgent


@dataclass
@export_module("autogen")
class UpdateSystemMessage:
    """Update the agent's system message before they reply

    Args:
        content_updater: The format string or function to update the agent's system message. Can be a format string or a Callable.
            If a string, it will be used as a template and substitute the context variables.
            If a Callable, it should have the signature:
                def my_content_updater(agent: ConversableAgent, messages: List[Dict[str, Any]]) -> str
    """

    content_updater: Callable | str

    def __post_init__(self):
        if isinstance(self.content_updater, str):
            # find all {var} in the string
            vars = re.findall(r"\{(\w+)\}", self.content_updater)
            if len(vars) == 0:
                warnings.warn("Update function string contains no variables. This is probably unintended.")

        elif isinstance(self.content_updater, Callable):
            sig = signature(self.content_updater)
            if len(sig.parameters) != 2:
                raise ValueError(
                    "The update function must accept two parameters of type ConversableAgent and List[Dict[str, Any]], respectively"
                )
            if sig.return_annotation != str:
                raise ValueError("The update function must return a string")
        else:
            raise ValueError("The update function must be either a string or a callable")


@export_module("autogen")
def register_function(
    f: Callable[..., Any],
    *,
    caller: "ConversableAgent",
    executor: "ConversableAgent",
    name: str | None = None,
    description: str,
) -> None:
    """Register a function to be proposed by an agent and executed for an executor.

    This function can be used instead of function decorators `@ConversationAgent.register_for_llm` and
    `@ConversationAgent.register_for_execution`.

    Args:
        f: the function to be registered.
        caller: the agent calling the function, typically an instance of ConversableAgent.
        executor: the agent executing the function, typically an instance of UserProxy.
        name: name of the function. If None, the function name will be used (default: None).
        description: description of the function. The description is used by LLM to decode whether the function
            is called. Make sure the description is properly describing what the function does or it might not be
            called by LLM when needed.

    """
    f = caller.register_for_llm(name=name, description=description)(f)
    executor.register_for_execution(name=name)(f)
