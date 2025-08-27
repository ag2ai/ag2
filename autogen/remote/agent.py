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
        # TODO: how to notify agent about chat terminated?
        self._reply_func_list.pop()  # remove ConversableAgent.a_generate_oai_reply
        self._reply_func_list.pop()  # remove ConversableAgent.generate_oai_reply

        self.register_reply(
            [Agent, None],
            HTTPRemoteAgent.generate_remote_reply,
            position=len(self._reply_func_list),  # set latest
        )

    def generate_remote_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: Agent | None = None,
        config: OpenAIWrapper | None = None,
    ) -> tuple[bool, str | dict[str, Any] | None]:
        if messages is None:
            messages = self._oai_messages[sender]

        reply_response = httpx.post(
            f"{self.url}/{self.name}",
            content=AgentBusMessage(messages=messages).model_dump_json(),
            timeout=30,
        )

        if reply_response.status_code == 204:
            return True, None

        try:
            serialized_message = AgentBusMessage.model_validate_json(reply_response.content)
        except Exception as e:
            raise ValueError(f"Remote client error: {reply_response}, {reply_response.content}") from e

        # TODO: support multiple messages response for remote chat history
        return True, serialized_message.messages[0]
