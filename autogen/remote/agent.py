# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import httpx

from autogen import Agent, ConversableAgent
from autogen.oai.client import OpenAIWrapper

from .protocol import AgentBusMessage


class HTTPRemoteAgent(ConversableAgent):
    def __init__(self, url: str, name: str) -> None:
        self.url = url

        super().__init__(name, silent=True)

        # Replace `generate_oai_reply` by `generate_remote_reply`
        # Save `_reply_func_list` order to execute tools & functions locally
        self._reply_func_list.pop()  # remove ConversableAgent.a_generate_oai_reply
        self._reply_func_list.pop()  # remove ConversableAgent.generate_oai_reply

        self.register_reply(
            [Agent, None],
            HTTPRemoteAgent.generate_remote_reply,
            position=len(self._reply_func_list),
        )
        self.register_reply(
            [Agent, None],
            HTTPRemoteAgent.a_generate_remote_reply,
            position=len(self._reply_func_list),
            ignore_async_in_sync_chat=True,
        )

    def generate_remote_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: Agent | None = None,
        config: OpenAIWrapper | None = None,
    ) -> tuple[bool, str | dict[str, Any] | None]:
        if messages is None:
            messages = self._oai_messages[sender]

        with httpx.Client() as client:
            # initiate remote procedure
            task_response = client.post(
                f"{self.url}/{self.name}",
                content=AgentBusMessage(
                    messages=messages,
                    context=self.context_variables.data,
                ).model_dump_json(),
                timeout=30,
            )

            task_id = task_response.json()

            # wait for remote task complete
            while (
                reply_response := client.get(
                    f"{self.url}/{self.name}/{task_id}",
                    timeout=30,
                )
            ).status_code == 425:
                pass

        if reply := self._process_remote_reply(reply_response):
            # TODO: support multiple messages response for remote chat history
            return True, reply.messages[-1]

        return True, None

    async def a_generate_remote_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: Agent | None = None,
        config: OpenAIWrapper | None = None,
    ) -> tuple[bool, str | dict[str, Any] | None]:
        if messages is None:
            messages = self._oai_messages[sender]

        async with httpx.AsyncClient() as client:
            # initiate remote procedure
            task_response = await client.post(
                f"{self.url}/{self.name}",
                content=AgentBusMessage(
                    messages=messages,
                    context=self.context_variables.data,
                ).model_dump_json(),
                timeout=30,
            )

            task_id = task_response.json()

            # wait for remote task complete
            while (
                reply_response := await client.get(
                    f"{self.url}/{self.name}/{task_id}",
                    timeout=30,
                )
            ).status_code == 425:
                pass

        if reply := self._process_remote_reply(reply_response):
            # TODO: support multiple messages response for remote chat history
            return True, reply.messages[-1]

        return True, None

    def _process_remote_reply(self, reply_response: httpx.Response) -> AgentBusMessage | None:
        if reply_response.status_code == 204:
            return None

        try:
            serialized_message = AgentBusMessage.model_validate_json(reply_response.content)
        except Exception as e:
            raise ValueError(f"Remote client error: {reply_response}, {reply_response.content}") from e

        return serialized_message
