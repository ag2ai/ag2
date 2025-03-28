# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel

from .context_str import ContextStr

if TYPE_CHECKING:
    # Avoid circular import
    from ..conversable_agent import ConversableAgent
    from ..groupchat import GroupChat
    
__all__ = ["AfterWorkOption"]

AfterWorkOption = Literal["terminate", "revert_to_user", "stay", "group_manager"]

# AfterWorkTarget


class AfterWorkTarget(BaseModel):
    """Base class for all AfterWork targets."""

    def resolve(self, last_speaker: "ConversableAgent", messages: list[dict[str, Any]], groupchat: "GroupChat") -> Any:
        """Resolve to a concrete agent or option."""
        raise NotImplementedError("Requires subclasses to implement.")


class AfterWorkTargetOption(AfterWorkTarget):
    """Target that represents an AfterWorkOption."""

    option: AfterWorkOption

    def __init__(self, option: AfterWorkOption, **data):
        super().__init__(option=option, **data)

    def resolve(
        self, last_speaker: "ConversableAgent", messages: list[dict[str, Any]], groupchat: "GroupChat"
    ) -> AfterWorkOption:
        """Resolve to the option value."""
        return self.option


class AfterWorkTargetAgent(AfterWorkTarget):
    """Target that represents an agent."""

    agent_name: str

    def __init__(self, agent: "ConversableAgent", **data):
        # Store the name from the agent for serialisation
        super().__init__(agent_name=agent.name, **data)

    def resolve(
        self, last_speaker: "ConversableAgent", messages: list[dict[str, Any]], groupchat: "GroupChat"
    ) -> "ConversableAgent":
        """Resolve to the actual agent object."""
        for agent in groupchat.agents:
            if agent.name == self.agent_name:
                return agent
        raise ValueError(f"Agent with name '{self.agent_name}' not found in groupchat")


class AfterWorkTargetAgentName(AfterWorkTarget):
    """Target that represents an agent name directly."""

    agent_name: str

    def __init__(self, agent_name: str, **data):
        super().__init__(agent_name=agent_name, **data)

    def resolve(self, last_speaker: "ConversableAgent", messages: list[dict[str, Any]], groupchat: "GroupChat") -> str:
        """Resolve to the agent name string."""
        return self.agent_name


# AfterWorkSelectionMessage


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

    target: AfterWorkTarget
    selection_message: Optional[AfterWorkSelectionMessage] = None

    def __init__(
        self,
        target_arg: AfterWorkTarget = None,
        selection_message_arg: Optional[AfterWorkSelectionMessage] = None,
        **data,
    ):
        if target_arg is not None:
            data["target"] = target_arg
        if selection_message_arg is not None:
            data["selection_message"] = selection_message_arg
        super().__init__(**data)
