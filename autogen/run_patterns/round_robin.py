# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Union

from ..agentchat.groupchat import GroupChat, GroupChatManager
from ..agentchat.termination import MaxTurns, TerminateProtocol
from ..doc_utils import export_module

if TYPE_CHECKING:
    from ..agentchat import Agent, ChatResult, LLMMessageType
    from .run_pattern import RunPatternProtocol


@export_module("autogen.run_patterns")
class RoundRobinRunPattern(GroupChatManager):
    def __init__(
        self,
        *agents: Iterable["Agent"],
        terminate_on: TerminateProtocol = MaxTurns(10),
    ) -> None:
        super().__init__(is_termination_msg=terminate_on.is_termination_message)
        self._agents = agents

    def run(
        self,
        message: str,
        messages: list["LLMMessageType"],
        summary_method: Optional[Union[str, Callable[..., Any]]],
    ) -> "ChatResult":
        groupchat = GroupChat(
            agents=self._agents,
            messages=messages,
            speaker_selection_method="round_robin",
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
        *agents: "Agent",
        message: str,
        messages: list["LLMMessageType"],
        max_turns: int,
        summary_method: Optional[Union[str, Callable[..., Any]]],
    ) -> "ChatResult":
        groupchat = GroupChat(
            agents=agents,
            messages=messages,
            max_round=max_turns,
            speaker_selection_method="round_robin",
        )

        self.initialize_groupchat(groupchat)

        for agent in self._agents:
            for tool in agent.tools:
                tool.register_for_execution(agent)

        return await agents[0].a_initiate_chat(
            recipient=self,
            message=message,
            summary_method=summary_method,
        )


if TYPE_CHECKING:

    def check_group_run_pattern_implements_run_pattern_protocol(x: RoundRobinRunPattern) -> RunPatternProtocol:
        return x
