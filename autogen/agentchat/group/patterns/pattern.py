# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

# Patterns of agent orchestrations
# Uses the group chat or the agents' handoffs to create a pattern

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union

from ..context_variables import ContextVariables
from ..group_utils import (
    create_group_manager,
    create_group_transition,
    link_agents_to_group_manager,
    prepare_group_agents,
    process_initial_messages,
    setup_context_variables,
)
from ..targets.transition_target import TerminateTarget, TransitionTarget

if TYPE_CHECKING:
    from ...agent import Agent
    from ...conversable_agent import ConversableAgent
    from ...groupchat import GroupChat, GroupChatManager
    from ..group_tool_executor import GroupToolExecutor


class Pattern:
    """Base class for all orchestration patterns.

    Patterns provide a reusable way to define how agents interact within a group chat.
    Each pattern encapsulates the logic for setting up agents, configuring handoffs,
    and determining the flow of conversation.
    """

    def __init__(
        self,
        initial_agent: "ConversableAgent",
        agents: list["ConversableAgent"],
        user_agent: Optional["ConversableAgent"] = None,
        group_manager_args: Optional[dict[str, Any]] = None,
        context_variables: Optional[ContextVariables] = None,
        group_after_work: Optional[TransitionTarget] = None,
        exclude_transit_message: bool = True,
        summary_method: Optional[Union[str, Callable[..., Any]]] = "last_msg",
    ):
        """Initialize the pattern with the required components.

        Args:
            initial_agent: The first agent to speak in the group chat.
            agents: List of all agents participating in the chat.
            user_agent: Optional user proxy agent.
            group_manager_args: Optional arguments for the GroupChatManager.
            context_variables: Initial context variables for the chat.
            group_after_work: Default after work transition behavior when no specific next agent is determined.
            exclude_transit_message: Whether to exclude transit messages from the conversation.
            summary_method: Method for summarizing the conversation.
        """
        self.initial_agent = initial_agent
        self.agents = agents
        self.user_agent = user_agent
        self.group_manager_args = group_manager_args or {}
        self.context_variables = context_variables or ContextVariables()
        self.group_after_work = group_after_work if group_after_work is not None else TerminateTarget()
        self.exclude_transit_message = exclude_transit_message
        self.summary_method = summary_method

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
        TransitionTarget,
        "GroupToolExecutor",
        "GroupChat",
        "GroupChatManager",
        list[dict[str, Any]],
        "ConversableAgent",
        list[str],
        list["Agent"],
    ]:
        """Prepare the group chat for orchestration.

        This is the main method called by initiate_group_chat to set up the pattern.
        Subclasses should override this method to implement pattern-specific behavior.

        Args:
            max_rounds: Maximum number of conversation rounds.
            messages: Initial message(s) to start the conversation.

        Returns:
            Tuple containing:
            - List of agents involved in the group chat
            - List of wrapped agents
            - User agent, if applicable
            - Context variables for the group chat
            - Initial agent for the group chat
            - Group-level after work transition for the group chat
            - Tool executor for the group chat
            - GroupChat instance
            - GroupChatManager instance
            - Processed messages
            - Last agent to speak
            - List of group agent names
            - List of temporary user agents
        """
        from ...groupchat import GroupChat

        # Prepare the agents using the existing helper function
        tool_executor, wrapped_agents = prepare_group_agents(
            self.agents, self.context_variables, self.exclude_transit_message
        )

        # Process the initial messages BEFORE creating the GroupChat
        # This will create a temporary user agent if needed
        processed_messages, last_agent, group_agent_names, temp_user_list = process_initial_messages(
            messages, self.user_agent, self.agents, wrapped_agents
        )

        # Create transition function (has enclosed state for initial agent)
        group_transition = create_group_transition(
            initial_agent=self.initial_agent,
            tool_execution=tool_executor,
            group_agent_names=group_agent_names,
            user_agent=self.user_agent,
            group_after_work=self.group_after_work,
        )

        # Create the group chat - now we use temp_user_list if no user_agent
        groupchat = GroupChat(
            agents=[tool_executor]
            + self.agents
            + wrapped_agents
            + ([self.user_agent] if self.user_agent else temp_user_list),
            messages=[],
            max_round=max_rounds,
            speaker_selection_method=group_transition,
        )

        # Create the group manager
        manager = create_group_manager(groupchat, self.group_manager_args, self.agents)

        # Point all agent's context variables to this function's context_variables
        setup_context_variables(tool_executor, self.agents, manager, self.context_variables)

        # Link all agents with the GroupChatManager to allow access to the group chat
        link_agents_to_group_manager(groupchat.agents, manager)

        return (
            self.agents,
            wrapped_agents,
            self.user_agent,
            self.context_variables,
            self.initial_agent,
            self.group_after_work,
            tool_executor,
            groupchat,
            manager,
            processed_messages,
            last_agent,
            group_agent_names,
            temp_user_list,
        )  # type: ignore[return-value]
