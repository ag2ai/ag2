# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import AgentCard, DataPart, Message, Part, Role, TaskState
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, EXTENDED_AGENT_CARD_PATH
from a2a.utils.message import get_data_parts, get_text_parts

from autogen import ConversableAgent
from autogen.agentchat.group import ContextVariables
from autogen.doc_utils import export_module
from autogen.oai.client import OpenAIWrapper
from autogen.remote.protocol import RequestMessage, ResponseMessage

logger = logging.getLogger(__name__)


@export_module("autogen.remote.a2a")
class A2ARemoteAgent(ConversableAgent):
    def __init__(
        self,
        url: str,
        name: str,
        *,
        silent: bool = False,
        client: httpx.AsyncClient | None = None,
        client_config: ClientConfig | None = None,
    ) -> None:
        self.url = url
        self._httpx_client = client
        self._client_config = client_config or ClientConfig()

        super().__init__(name, silent=silent)

        self.__llm_config: dict[str, Any] = {}

        self.replace_reply_func(
            ConversableAgent.generate_oai_reply,
            A2ARemoteAgent.generate_remote_reply,
        )
        self.replace_reply_func(
            ConversableAgent.a_generate_oai_reply,
            A2ARemoteAgent.a_generate_remote_reply,
        )

    def generate_remote_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: ConversableAgent | None = None,
        config: OpenAIWrapper | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        raise NotImplementedError("generate_remote_reply is not implemented")

    async def a_generate_remote_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: ConversableAgent | None = None,
        config: OpenAIWrapper | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        if messages is None:
            messages = self._oai_messages[sender]

        self._client_config.httpx_client = self._httpx_client or httpx.AsyncClient(timeout=30)

        async with self._client_config.httpx_client:
            agent_card = await self._get_agent_card()
            agent_client = ClientFactory(self._client_config).create(agent_card)

            async for event in agent_client.send_message(
                Message(
                    role=Role.user,
                    parts=[
                        Part(
                            root=DataPart(
                                data=RequestMessage(
                                    messages=messages,
                                    context=self.context_variables.data,
                                    client_tools=self.__llm_config.get("tools", []),
                                ).model_dump(),
                            )
                        )
                    ],
                    message_id=uuid4().hex,
                    context_id=uuid4().hex,
                )
            ):
                if isinstance(event, Message):
                    return True, {"content": " ".join(get_text_parts(event.parts))}

                task, _ = event

                if task.status.state == TaskState.submitted:
                    continue

                if task.status.state == TaskState.input_required:
                    raise NotImplementedError

                if task.status.state in (
                    TaskState.completed,
                    TaskState.failed,
                    TaskState.canceled,
                    TaskState.rejected,
                ):
                    if not task.history or not (parts := task.history[-1].parts):
                        return True, None

                    reply = ResponseMessage.model_validate(get_data_parts(parts)[-1])

                    if sender:
                        context_variables = ContextVariables(reply.context)
                        sender.context_variables.update(context_variables.to_dict())

                    return True, reply.messages[-1]

        return True, None

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

    async def _get_agent_card(
        self,
        auth_http_kwargs: dict[str, Any] | None = None,
    ) -> AgentCard:
        client = self._httpx_client or httpx.AsyncClient(timeout=30)

        resolver = A2ACardResolver(httpx_client=client, base_url=self.url)

        card: AgentCard | None = None

        try:
            logger.info(f"Attempting to fetch public agent card from: {self.url}{AGENT_CARD_WELL_KNOWN_PATH}")
            card = await resolver.get_agent_card()

            if card.supports_authenticated_extended_card:
                try:
                    card = await resolver.get_agent_card(
                        relative_card_path=EXTENDED_AGENT_CARD_PATH,
                        http_kwargs=auth_http_kwargs,
                    )
                except Exception as e_extended:
                    logger.warning(
                        f"Failed to fetch extended agent card: {e_extended}. Will proceed with public card.",
                        exc_info=True,
                    )

        except Exception as e:
            raise RuntimeError("Failed to fetch the public agent card. Cannot continue.") from e

        if card.url == "http://magic-useless-url/":
            card.url = self.url

        return card
