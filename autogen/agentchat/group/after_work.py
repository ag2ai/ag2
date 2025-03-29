# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

from .context_str import ContextStr
from .transition_target import TransitionTarget

if TYPE_CHECKING:
    # Avoid circular import
    from ..conversable_agent import ConversableAgent

__all__ = [
    "AfterWork",
    "AfterWorkSelectionMessage",
    "AfterWorkSelectionMessageContextStr",
    "AfterWorkSelectionMessageString",
]


# AfterWorkSelectionMessage protocol and implementations
class AfterWorkSelectionMessage(BaseModel):
    """Base class for all AfterWork selection message types."""

    def get_message(self, agent: "ConversableAgent", messages: list[dict[str, Any]]) -> str:
        """Get the formatted message."""
        raise NotImplementedError("Requires subclasses to implement.")


class AfterWorkSelectionMessageString(AfterWorkSelectionMessage):
    """Selection message that uses a plain string template."""

    template: str

    def __init__(self, template: str, **data):
        super().__init__(template=template, **data)

    def get_message(self, agent: Any, messages: list[dict[str, Any]]) -> str:
        """Get the message string."""
        return self.template


class AfterWorkSelectionMessageContextStr(AfterWorkSelectionMessage):
    """Selection message that uses a ContextStr template."""

    context_str_template: str

    def __init__(self, context_str_template: str, **data):
        super().__init__(context_str_template=context_str_template, **data)

    def get_message(self, agent: "ConversableAgent", messages: list[dict[str, Any]]) -> str:
        """Get the formatted message with context variables substituted."""
        context_str = ContextStr(self.context_str_template)
        return context_str.format(agent.context_variables)


# AfterWork


class AfterWork(BaseModel):
    """Handles the next step in the conversation when an agent doesn't suggest a tool call or a handoff."""

    target: TransitionTarget
    selection_message: Optional[AfterWorkSelectionMessage] = None

    def __init__(
        self,
        target: TransitionTarget,
        selection_message: Optional[AfterWorkSelectionMessage] = None,
        **data,
    ):
        data["target"] = target
        if selection_message is not None:
            data["selection_message"] = selection_message
        super().__init__(**data)
