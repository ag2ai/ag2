# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Optional, Union

from ..agentchat.groupchat import GroupChat, GroupChatManager
from ..agentchat.termination import MaxTurns, TerminateProtocol
from ..llm_config import LLMConfig

if TYPE_CHECKING:
    from ..agentchat import Agent, ChatResult, LLMMessageType
    from .run_pattern import RunPatternProtocol

__all__ = ["GroupRunPattern"]


class GroupRunPattern(GroupChatManager):
    def __init__(
        self,
        *agents: Iterable["Agent"],
        terminate_on: TerminateProtocol = MaxTurns(10),
        llm_config: Optional[Union[LLMConfig, dict[str, Any], Literal[False]]] = None,
    ) -> None:
        super().__init__(is_termination_msg=terminate_on.is_termination_message, llm_config=llm_config)
        self._agents = agents

    def run(
        self,
        message: str,
        messages: list["LLMMessageType"],
        summary_method: Optional[Union[str, Callable[..., Any]]],
    ) -> "ChatResult":
        """Run a group chat with the provided agents and messages.

        Args:
            agents: The agents participating in the group chat.
            message: Initial message to start the group chat.
            messages: The messages to use in the group chat.
            max_turns: The maximum number of turns in the group chat.
            summary_method (str or callable): a method to get a summary from the chat. Default is DEFAULT_SUMMARY_METHOD, i.e., "last_msg".
                Supported strings are "last_msg" and "reflection_with_llm":
                    - when set to "last_msg", it returns the last message of the dialog as the summary.
                    - when set to "reflection_with_llm", it returns a summary extracted using an llm client.
                        `llm_config` must be set in either the recipient or sender.

                A callable summary_method should take the recipient and sender agent in a chat as input and return a string of summary. E.g.,

                ```python
                def my_summary_method(
                    sender: ConversableAgent,
                    recipient: ConversableAgent,
                    summary_args: dict,
                ):
                    return recipient.last_message(sender)["content"]
                ```

        Returns:
            A ChatResult object.
        """
        groupchat = GroupChat(
            agents=self._agents,
            messages=messages,
            speaker_selection_method="auto",
        )

        self.initialize_groupchat(groupchat)

        for agent in self._agents:
            for tool in agent.tools:
                tool.register_for_execution(agent)

        return self._agents[0].initiate_chat(
            recipient=self,
            message=message,
            summary_method=summary_method,
        )

    async def a_run(
        self,
        message: str,
        messages: list["LLMMessageType"],
        summary_method: Optional[Union[str, Callable[..., Any]]],
    ) -> "ChatResult":
        """Run a group chat with the provided agents and messages.

        Args:
            agents: The agents participating in the group chat.
            message: Initial message to start the group chat.
            messages: The messages to use in the group chat.
            max_turns: The maximum number of turns in the group chat.
            summary_method (str or callable): a method to get a summary from the chat. Default is DEFAULT_SUMMARY_METHOD, i.e., "last_msg".
                Supported strings are "last_msg" and "reflection_with_llm":
                    - when set to "last_msg", it returns the last message of the dialog as the summary.
                    - when set to "reflection_with_llm", it returns a summary extracted using an llm client.
                        `llm_config` must be set in either the recipient or sender.

                A callable summary_method should take the recipient and sender agent in a chat as input and return a string of summary. E.g.,

                ```python
                def my_summary_method(
                    sender: ConversableAgent,
                    recipient: ConversableAgent,
                    summary_args: dict,
                ):
                    return recipient.last_message(sender)["content"]
                ```

        Returns:
            A ChatResult object.
        """
        groupchat = GroupChat(
            agents=self._agents,
            messages=messages,
            speaker_selection_method="auto",
        )

        self.initialize_groupchat(groupchat)

        for agent in self._agents:
            for tool in agent.tools:
                tool.register_for_execution(agent)

        return await self._agents[0].a_initiate_chat(
            recipient=self,
            message=message,
            summary_method=summary_method,
        )


if TYPE_CHECKING:

    def check_group_run_pattern_implements_run_pattern_protocol(x: GroupRunPattern) -> RunPatternProtocol:
        return x
