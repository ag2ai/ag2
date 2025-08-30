# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["FunctionTargetResult"]


from pydantic import BaseModel
from typing import TYPE_CHECKING, Any
from .context_variables import ContextVariables
from .targets.transition_target import TransitionTarget

if TYPE_CHECKING:
    from ..conversable_agent import ConversableAgent
    class FunctionTargetMessage(BaseModel):
        content: str
        msg_target: "ConversableAgent"
else:
    class FunctionTargetMessage(BaseModel):
        content: str
        msg_target: Any

class FunctionTargetResult(BaseModel):
    """Result of a function handoff that is used to provide the return message and the target to transition to."""
    messages: list[FunctionTargetMessage] | str | None = None
    target: TransitionTarget
    context_variables: ContextVariables | None = None

    def __str__(self) -> str:
        """The string representation for FunctionTargetResult will be """
        # not implemented yet
        return ""
