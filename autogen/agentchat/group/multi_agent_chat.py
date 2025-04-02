# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Union

from ...doc_utils import export_module
from ..chat import ChatResult
from .context_variables import ContextVariables
from .group_utils import cleanup_temp_user_messages

if TYPE_CHECKING:
    from ..agent import Agent
    from .patterns.pattern import Pattern

__all__ = [
    "a_initiate_group_chat",
    "initiate_group_chat",
]


@export_module("autogen")
def initiate_group_chat(
    pattern: "Pattern",
    messages: Union[list[dict[str, Any]], str],
    max_rounds: int = 20,
) -> tuple[ChatResult, ContextVariables, "Agent"]:
    """Initialize and run a group chat using a pattern for configuration.

    Args:
        pattern: Pattern object that encapsulates the chat configuration.
        messages: Initial message(s).
        max_rounds: Maximum number of conversation rounds.

    Returns:
        ChatResult:         Conversations chat history.
        ContextVariables:   Updated Context variables.
        "ConversableAgent":   Last speaker.
    """
    # Let the pattern prepare the group chat and all its components
    # Only passing the necessary parameters that aren't already in the pattern
    (
        agents,
        wrapped_agents,
        user_agent,
        context_variables,
        initial_agent,
        group_after_work,
        tool_execution,
        groupchat,
        manager,
        processed_messages,
        last_agent,
        group_agent_names,
        temp_user_list,
    ) = pattern.prepare_group_chat(
        max_rounds=max_rounds,
        messages=messages,
    )

    # Start or resume the conversation
    if len(processed_messages) > 1:
        last_agent, last_message = manager.resume(messages=processed_messages)
        clear_history = False
    else:
        last_message = processed_messages[0]
        clear_history = True

    if last_agent is None:
        raise ValueError("No agent selected to start the conversation")

    chat_result = last_agent.initiate_chat(
        manager,
        message=last_message,
        clear_history=clear_history,
        summary_method=pattern.summary_method,
    )

    cleanup_temp_user_messages(chat_result)

    return chat_result, context_variables, manager.last_speaker


@export_module("autogen.agentchat")
async def a_initiate_group_chat(
    pattern: "Pattern",
    messages: Union[list[dict[str, Any]], str],
    max_rounds: int = 20,
) -> tuple[ChatResult, ContextVariables, "Agent"]:
    """Async version of initiate_group_chat."""
    raise NotImplementedError("This function is not implemented yet")
