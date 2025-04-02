# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from ..after_work import AfterWork
from ..context_variables import ContextVariables
from .pattern import Pattern

if TYPE_CHECKING:
    from ...conversable_agent import ConversableAgent
    from ...groupchat import GroupChat, GroupChatManager
    from ..group_tool_executor import GroupToolExecutor


class GenericPattern(Pattern):
    """GenericPattern implements no pattern as such, used when a pattern isn't really defined."""

    def prepare_group_chat(
        self,
        max_rounds: int,
        messages: Union[list[dict[str, Any]], str],
    ) -> Tuple[
        list["ConversableAgent"],
        list["ConversableAgent"],
        Optional["ConversableAgent"],
        ContextVariables,
        "ConversableAgent",
        AfterWork,
        "GroupToolExecutor",
        "GroupChat",
        "GroupChatManager",
        list[dict[str, Any]],
        Any,
        list[str],
        list[Any],
    ]:
        """Prepare the group chat for organic agent selection.

        Ensures that:
        1. The group manager has a valid LLM config
        2. All agents have appropriate descriptions for the group manager to use

        Args:
            max_rounds: Maximum number of conversation rounds.
            messages: Initial message(s) to start the conversation.

        Returns:
            Tuple containing all necessary components for the group chat.
        """
        # Use the parent class's implementation to prepare the agents and group chat
        (
            agents,
            wrapped_agents,
            user_agent,
            context_variables,
            initial_agent,
            _,
            tool_executor,
            groupchat,
            manager,
            processed_messages,
            last_agent,
            group_agent_names,
            temp_user_list,
        ) = super().prepare_group_chat(
            max_rounds=max_rounds,
            messages=messages,
        )

        # Return all components with our group_after_work
        return (
            agents,
            wrapped_agents,
            user_agent,
            context_variables,
            initial_agent,
            self.after_work,
            tool_executor,
            groupchat,
            manager,
            processed_messages,
            last_agent,
            group_agent_names,
            temp_user_list,
        )
