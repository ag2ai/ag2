# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from ..agentchat import ConversableAgent
from ..agentchat.contrib.swarm_agent import AfterWorkOption, a_initiate_swarm_chat, initiate_swarm_chat
from ..agentchat.groupchat import GroupChat
from ..doc_utils import export_module
from ..llm_config import LLMConfig

if TYPE_CHECKING:
    from ..agentchat import Agent, ChatResult, LLMMessageType
    from .run_pattern import RunPatternProtocol


@export_module("autogen.run_patterns")
class SwarmRunPattern:
    def __init__(
        self,
        *agents: "Agent",
        llm_config: Optional[Union[LLMConfig, dict[str, str]]] = None,
        context_variables: Optional[dict[str, Any]] = None,
        after_work: Optional[
            Union[
                AfterWorkOption,
                Callable[
                    [ConversableAgent, list[dict[str, Any]], GroupChat], Union[AfterWorkOption, ConversableAgent, str]
                ],
            ]
        ] = AfterWorkOption.TERMINATE,
        exclude_transit_message: bool = True,
    ):
        self._agents = agents
        if llm_config is None:
            llm_config = LLMConfig.get_current_llm_config()

        self.llm_config = llm_config
        self.after_work = after_work
        self.context_variables = context_variables
        self.exclude_transit_message = exclude_transit_message

    def run(
        self,
        message: str,
        messages: list["LLMMessageType"],
        summary_method: Optional[Union[str, Callable[..., Any]]],
    ) -> "ChatResult":
        result, _, _ = initiate_swarm_chat(
            initial_agent=self._agents[0],
            agents=list(self._agents),
            messages=message if len(messages) == 0 else [*messages, {"role": "user", "content": message}],
            swarm_manager_args={"llm_config": self.llm_config},
            after_work=self.after_work,
            context_variables=self.context_variables,
            exclude_transit_message=self.exclude_transit_message,
            summary_method=summary_method,
        )

        return result

    async def a_run(
        self,
        *agents: "Agent",
        message: str,
        messages: list["LLMMessageType"],
        max_turns: int,
        summary_method: Optional[Union[str, Callable[..., Any]]],
    ) -> "ChatResult":
        result, _, _ = await a_initiate_swarm_chat(
            initial_agent=agents[0],
            agents=list(agents),
            messages=message if len(messages) == 0 else [*messages, message],
            max_rounds=max_turns,
            swarm_manager_args={"llm_config": self.llm_config},
            after_work=self.after_work,
            context_variables=self.context_variables,
            exclude_transit_message=self.exclude_transit_message,
            summary_method=summary_method,
        )

        return result


if TYPE_CHECKING:

    def check_run_pattern_manager_implements_run_pattern_protocol(x: SwarmRunPattern) -> RunPatternProtocol:
        return x
