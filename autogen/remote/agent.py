# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, cast

import httpx

from autogen import ConversableAgent
from autogen.agentchat.group import ContextVariables
from autogen.doc_utils import export_module
from autogen.oai.client import OpenAIWrapper

from .protocol import AgentBusMessage


@export_module("autogen.remote")
class HTTPRemoteAgent(ConversableAgent):
    def __init__(
        self,
        url: str,
        name: str,
        *,
        silent: bool = False,
        client: httpx.AsyncClient | httpx.Client | None = None,
    ) -> None:
        self.url = url

        self._httpx_client = client

        super().__init__(name, silent=silent)

        self.replace_reply_func(
            ConversableAgent.generate_oai_reply,
            HTTPRemoteAgent.generate_remote_reply,
        )
        self.replace_reply_func(
            ConversableAgent.a_generate_oai_reply,
            HTTPRemoteAgent.a_generate_remote_reply,
        )

    def generate_remote_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: ConversableAgent | None = None,
        config: OpenAIWrapper | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        if messages is None:
            messages = self._oai_messages[sender]

        client = cast(httpx.Client, self._httpx_client) or httpx.Client()

        with client:
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
            if sender:
                context_variables = ContextVariables(reply.context)
                sender.context_variables.update(context_variables.to_dict())
            # TODO: support multiple messages response for remote chat history
            return True, reply.messages[-1]

        return True, None

    async def a_generate_remote_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: ConversableAgent | None = None,
        config: OpenAIWrapper | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        if messages is None:
            messages = self._oai_messages[sender]

        client = cast(httpx.AsyncClient, self._httpx_client) or httpx.AsyncClient()

        async with client:
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
            if sender:
                context_variables = ContextVariables(reply.context)
                sender.context_variables.update(context_variables.to_dict())
            # TODO: support multiple messages response for remote chat history
            return True, reply.messages[-1]

        return True, None

    def _process_remote_reply(self, reply_response: httpx.Response) -> AgentBusMessage | None:
        if reply_response.status_code == 204:
            return None

        try:
            serialized_message = AgentBusMessage.model_validate_json(reply_response.content)
        except Exception as e:
            raise ValueError(f"Remote client error: {reply_response}, {reply_response.content!r}") from e

        return serialized_message
