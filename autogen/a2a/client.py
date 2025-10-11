# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from pprint import pformat
from typing import Any, cast
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClientHTTPError, Client, ClientConfig, ClientEvent, ClientFactory
from a2a.types import AgentCard, Message, Task, TaskQueryParams, TaskState
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, EXTENDED_AGENT_CARD_PATH, PREV_AGENT_CARD_WELL_KNOWN_PATH

from autogen import ConversableAgent
from autogen.agentchat.group import ContextVariables
from autogen.doc_utils import export_module
from autogen.oai.client import OpenAIWrapper
from autogen.remote.protocol import RequestMessage, ResponseMessage

from .errors import A2aAgentNotFoundError, A2aClientError
from .utils import request_message_to_a2a, response_message_from_a2a, response_message_from_a2a_message

logger = logging.getLogger(__name__)


@export_module("autogen.a2a")
class A2aRemoteAgent(ConversableAgent):
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
            A2aRemoteAgent.generate_remote_reply,
        )
        self.replace_reply_func(
            ConversableAgent.a_generate_oai_reply,
            A2aRemoteAgent.a_generate_remote_reply,
        )

    def generate_remote_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: ConversableAgent | None = None,
        config: OpenAIWrapper | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support synchronous reply generation")

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

        initial_message = request_message_to_a2a(
            request_message=RequestMessage(
                messages=messages,
                context=self.context_variables.data,
                client_tools=self.__llm_config.get("tools", []),
            ),
            context_id=uuid4().hex,
        )

        self._client_config.httpx_client = self._httpx_client or httpx.AsyncClient(timeout=30)
        async with self._client_config.httpx_client:
            agent_client = ClientFactory(self._client_config).create(self.__agent_card)

            if self.__agent_card.capabilities.streaming:
                reply = await self._ask_streaming(agent_client, initial_message)
                return self._apply_reply(reply, sender)

            else:
                reply = await self._ask_polling(agent_client, initial_message)
                return self._apply_reply(reply, sender)

        return True, None

    def _apply_reply(
        self, reply: ResponseMessage | None, sender: ConversableAgent | None
    ) -> tuple[bool, dict[str, Any] | None]:
        if not reply:
            return True, None

        if sender and reply.context:
            context_variables = ContextVariables(reply.context)
            self.context_variables.update(context_variables.to_dict())
            sender.context_variables.update(context_variables.to_dict())

        return True, reply.messages[-1]

    async def _ask_streaming(self, client: Client, message: Message) -> ResponseMessage | None:
        try:
            async for event in client.send_message(message):
                result, task = self._process_event(event)
                if not task:
                    return result
        except httpx.ConnectError as e:
            if not self.__agent_card:
                raise A2aClientError("Failed to connect to the agent: agent card not found") from e
            raise A2aClientError(f"Failed to connect to the agent: {pformat(self.__agent_card.model_dump())}") from e
        return None

    async def _ask_polling(self, client: Client, message: Message) -> ResponseMessage | None:
        try:
            async for event in client.send_message(message):
                result, started_task = self._process_event(event)
                if not started_task:
                    return result
                break
        except httpx.ConnectError as e:
            if not self.__agent_card:
                raise A2aClientError("Failed to connect to the agent: agent card not found") from e
            raise A2aClientError(f"Failed to connect to the agent: {pformat(self.__agent_card.model_dump())}") from e

        started_task = cast(Task, started_task)
        while True:
            try:
                task = await client.get_task(TaskQueryParams(id=started_task.id))
            except httpx.ConnectError as e:
                if not self.__agent_card:
                    raise A2aClientError("Failed to connect to the agent: agent card not found") from e
                raise A2aClientError(
                    f"Failed to connect to the agent: {pformat(self.__agent_card.model_dump())}"
                ) from e

            if _is_task_completed(task):
                return response_message_from_a2a(task.artifacts)

            await asyncio.sleep(1)

    def _process_event(self, event: ClientEvent | Message) -> tuple[ResponseMessage | None, Task | None]:
        if isinstance(event, Message):
            return response_message_from_a2a_message(event), None

        task, _ = event
        if _is_task_completed(task):
            return response_message_from_a2a(task.artifacts), None

        return None, task

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
            raise A2aAgentNotFoundError(self.name) from e

        return card


def _is_task_completed(task: Task) -> bool:
    if task.status.state is TaskState.failed:
        raise A2aClientError(f"Task failed: {pformat(task.model_dump())}")

    if task.status.state is TaskState.rejected:
        raise A2aClientError(f"Task rejected: {pformat(task.model_dump())}")

    return task.status.state in (
        TaskState.completed,
        TaskState.canceled,
    )
