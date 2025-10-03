# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClientHTTPError, ClientConfig, ClientFactory
from a2a.types import AgentCard, Message, Task, TaskQueryParams, TaskState
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, EXTENDED_AGENT_CARD_PATH, PREV_AGENT_CARD_WELL_KNOWN_PATH
from a2a.utils.message import get_message_text

from autogen import ConversableAgent
from autogen.agentchat.group import ContextVariables
from autogen.doc_utils import export_module
from autogen.oai.client import OpenAIWrapper
from autogen.remote.protocol import RequestMessage, ResponseMessage

from .utils import request_message_to_a2a, response_message_from_a2a

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
        self.__agent_card: AgentCard | None = None

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

        if not self.__agent_card:
            self.__agent_card = await self._get_agent_card()

        reply: ResponseMessage | None = None

        request_message = RequestMessage(
            messages=messages,
            context=self.context_variables.data,
            client_tools=self.__llm_config.get("tools", []),
        )

        context_id = uuid4().hex

        self._client_config.httpx_client = self._httpx_client or httpx.AsyncClient(timeout=30)
        async with self._client_config.httpx_client:
            agent_client = ClientFactory(self._client_config).create(self.__agent_card)

            async for event in agent_client.send_message(
                request_message_to_a2a(request_message, context_id),
            ):
                if isinstance(event, Message):
                    return True, {"content": get_message_text(event)}

                task, _ = event

                if _is_task_completed(task):
                    reply = response_message_from_a2a(task.artifacts)
                    return self._apply_reply(reply, sender)

                while True:
                    task = await agent_client.get_task(TaskQueryParams(id=task.id))

                    if _is_task_completed(task):
                        reply = response_message_from_a2a(task.artifacts)
                        return self._apply_reply(reply, sender)

                    await asyncio.sleep(1)

        return self._apply_reply(reply, sender)

    def _apply_reply(
        self, reply: ResponseMessage | None, sender: ConversableAgent | None
    ) -> tuple[bool, dict[str, Any] | None]:
        if not reply:
            return True, None

        if sender and reply.context:
            context_variables = ContextVariables(reply.context)
            sender.context_variables.update(context_variables.to_dict())

        return True, reply.messages[-1]

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

            try:
                card = await resolver.get_agent_card(relative_card_path=AGENT_CARD_WELL_KNOWN_PATH)
            except A2AClientHTTPError as e_public:
                if e_public.status_code == 404:
                    logger.info(
                        f"Attempting to fetch public agent card from: {self.url}{PREV_AGENT_CARD_WELL_KNOWN_PATH}"
                    )
                    card = await resolver.get_agent_card(relative_card_path=PREV_AGENT_CARD_WELL_KNOWN_PATH)
                else:
                    raise e_public

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


def _is_task_completed(task: Task) -> bool:
    return task.status.state in (
        TaskState.completed,
        TaskState.failed,
        TaskState.canceled,
        TaskState.rejected,
    )
