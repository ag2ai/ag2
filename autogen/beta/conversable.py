# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from autogen import ConversableAgent
from autogen.oai.client import OpenAIWrapper

from .agent import Agent, AgentReply
from .tools.final import ClientTool


class ConversableAdapter(ConversableAgent):
    def __init__(self, agent: Agent) -> None:
        super().__init__(agent.name)

        self.__agent = agent
        self.__conversation: AgentReply | None = None
        self.__client_tools: list[ClientTool] = []
        self.__llm_config: dict[str, Any] = {}

        self.replace_reply_func(
            ConversableAgent.generate_oai_reply,
            ConversableAdapter.generate_conversable_reply,
        )
        self.replace_reply_func(
            ConversableAgent.a_generate_oai_reply,
            ConversableAdapter.a_generate_conversable_reply,
        )

    def generate_conversable_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: ConversableAgent | None = None,
        config: OpenAIWrapper | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support synchronous reply generation")

    async def a_generate_conversable_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: ConversableAgent | None = None,
        config: OpenAIWrapper | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        if messages is None:
            messages = self._oai_messages[sender]

        request = messages[-1]["content"]

        if not self.__conversation:
            r = self.__conversation = await self.__agent.ask(
                request,
                variables=self.context_variables.data,
                tools=self.__client_tools,
            )
        else:
            r = self.__conversation = await self.__conversation.ask(
                request,
                variables=self.context_variables.data,
                tools=self.__client_tools,
            )

        if vars := r.context.variables:
            self.context_variables.update(vars)
            if sender:
                sender.context_variables.update(vars)

        result = r.response.to_api() | {"name": self.name}

        return True, result

    def update_tool_signature(
        self,
        tool_sig: str | dict[str, Any],
        is_remove: bool,
        silent_override: bool = False,
    ) -> None:
        self.__llm_config = self._update_tool_config(
            self.__llm_config,
            tool_sig=tool_sig,
            is_remove=is_remove,
            silent_override=silent_override,
        )
        self.__client_tools = [ClientTool(t) for t in self.__llm_config.get("tools", [])]
