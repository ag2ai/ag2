# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from collections.abc import Sequence
from pprint import pformat
from typing import Any, cast
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClientHTTPError, Client, ClientCallInterceptor, ClientConfig, ClientEvent
from a2a.client import ClientFactory as A2AClientFactory
from a2a.types import AgentCard, Message, Task, TaskIdParams, TaskQueryParams, TaskState
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, EXTENDED_AGENT_CARD_PATH, PREV_AGENT_CARD_WELL_KNOWN_PATH
from typing_extensions import Self

from autogen import ConversableAgent
from autogen.agentchat.group import ContextVariables
from autogen.doc_utils import export_module
from autogen.events.agent_events import TerminationEvent
from autogen.io.base import IOStream
from autogen.oai.client import OpenAIWrapper
from autogen.remote.httpx_client_factory import ClientFactory, EmptyClientFactory
from autogen.remote.protocol import RequestMessage, ResponseMessage

from .errors import A2aAgentNotFoundError, A2aClientError
from .utils import (
    request_message_to_a2a,
    response_message_from_a2a_message,
    response_message_from_a2a_task,
)

logger = logging.getLogger(__name__)


@export_module("autogen.a2a")
class A2aRemoteAgent(ConversableAgent):
    """`a2a-sdk`-based client for handling asynchronous communication with an A2A server.

    It has fully-compatible with original `ConversableAgent` API, so you can easily integrate
    remote A2A agents to existing collaborations.

    Args:
        url: The URL of the A2A server to connect to.
        name: A unique identifier for this client instance.
        silent: whether to print the message sent. If None, will use the value of silent in each function.
        client: An optional HTTPX client instance factory.
        client_config: A2A Client configuration options.
        max_reconnects: Maximum number of reconnection attempts before giving up.
        polling_interval: Time in seconds between polling operations. Works for A2A Servers doesn't support streaming.
        interceptors: A list of interceptors to use for the client.
    """

    def __init__(
        self,
        url: str,
        name: str,
        *,
        silent: bool | None = None,
        client: ClientFactory | None = None,
        client_config: ClientConfig | None = None,
        interceptors: Sequence[ClientCallInterceptor] = (),
        max_reconnects: int = 3,
        polling_interval: float = 0.5,
    ) -> None:
        self.url = url  # make it public for backward compatibility

        self._httpx_client_factory = client or EmptyClientFactory()
        self._card_resolver = A2ACardResolver(
            httpx_client=self._httpx_client_factory(),
            base_url=url,
        )

        self._max_reconnects = max_reconnects
        self._polling_interval = polling_interval

        super().__init__(name, silent=silent)

        self.__llm_config: dict[str, Any] = {}

        self._client_config = client_config or ClientConfig()
        self._interceptors = list(interceptors)
        self._agent_card: AgentCard | None = None

        self.replace_reply_func(
            ConversableAgent.generate_oai_reply,
            A2aRemoteAgent.generate_remote_reply,
        )
        self.replace_reply_func(
            ConversableAgent.a_generate_oai_reply,
            A2aRemoteAgent.a_generate_remote_reply,
        )

    @classmethod
    def from_card(
        cls,
        card: AgentCard,
        *,
        silent: bool | None = None,
        client: ClientFactory | None = None,
        client_config: ClientConfig | None = None,
        max_reconnects: int = 3,
        polling_interval: float = 0.5,
        interceptors: Sequence[ClientCallInterceptor] = (),
    ) -> Self:
        """Creates an A2aRemoteAgent instance from an existing AgentCard.

        This method allows you to instantiate an A2aRemoteAgent directly using a pre-existing
        AgentCard, such as one retrieved from a discovery service or constructed manually.
        The resulting agent will use the data from the given card and avoid redundant card
        fetching. The agent's registryURL is set to "UNKNOWN" since it is assumed to be derived
        from the card.

        Args:
            card: The agent card containing metadata and configuration for the remote agent.
            silent: whether to print the message sent. If None, will use the value of silent in each function.
            client: An optional HTTPX client instance factory.
            client_config: A2A Client configuration options.
            max_reconnects: Maximum number of reconnection attempts before giving up.
            polling_interval: Time in seconds between polling operations. Works for A2A Servers doesn't support streaming.
            interceptors: A list of interceptors to use for the client.

        Returns:
            Self: An instance of the A2aRemoteAgent configured with the provided card.
        """
        instance = cls(
            url="UNKNOWN",
            name=card.name,
            silent=silent,
            client=client,
            client_config=client_config,
            max_reconnects=max_reconnects,
            polling_interval=polling_interval,
            interceptors=interceptors,
        )
        instance._agent_card = card
        return instance

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

        if not self._agent_card:
            self._agent_card = await self._get_agent_card()

        context_id = uuid4().hex

        self._client_config.httpx_client = self._httpx_client_factory()
        async with self._client_config.httpx_client:
            agent_client = A2AClientFactory(self._client_config).create(
                self._agent_card,
                interceptors=self._interceptors,
            )

            while True:
                initial_message = request_message_to_a2a(
                    request_message=RequestMessage(
                        messages=messages,
                        context=self.context_variables.data,
                        client_tools=self.__llm_config.get("tools", []),
                    ),
                    context_id=context_id,
                )

                if self._agent_card.capabilities.streaming:
                    reply = await self._ask_streaming(agent_client, initial_message)
                else:
                    reply = await self._ask_polling(agent_client, initial_message)

                if not reply:
                    return True, None

                messages = reply.messages
                if reply.input_required is not None:
                    user_input = await self.a_get_human_input(prompt=f"Input for `{self.name}`\n{reply.input_required}")

                    if user_input == "exit":
                        IOStream.get_default().send(
                            TerminationEvent(
                                termination_reason="User requested to end the conversation",
                                sender=self,
                                recipient=sender,
                            )
                        )
                        return True, None

                    messages.append({"content": user_input, "role": "user"})
                    continue

                if sender and reply.context:
                    context_variables = ContextVariables(reply.context)
                    self.context_variables.update(context_variables.to_dict())
                    sender.context_variables.update(context_variables.to_dict())

                return True, reply.messages[-1]

    async def _ask_streaming(self, client: Client, message: Message) -> ResponseMessage | None:
        started_task: Task | None = None
        try:
            async for event in client.send_message(message):
                result, started_task = self._process_event(event)
                if not started_task:
                    return result

        except (httpx.ConnectError, A2AClientHTTPError) as e:
            if not started_task:
                if not self._agent_card:
                    raise A2aClientError("Failed to connect to the agent: agent card not found") from e
                raise A2aClientError(f"Failed to connect to the agent: {pformat(self._agent_card.model_dump())}") from e

        connection_attemps, started_task = 1, cast(Task, started_task)
        while connection_attemps < self._max_reconnects:
            try:
                async for event in client.resubscribe(TaskIdParams(id=started_task.id)):
                    result, task = self._process_event(event)
                    if not task:
                        return result

            except (httpx.ConnectError, A2AClientHTTPError) as e:
                connection_attemps += 1
                if connection_attemps >= self._max_reconnects:
                    if not self._agent_card:
                        raise A2aClientError("Failed to connect to the agent: agent card not found") from e
                    raise A2aClientError(
                        f"Failed to connect to the agent: {pformat(self._agent_card.model_dump())}"
                    ) from e

        return None

    async def _ask_polling(self, client: Client, message: Message) -> ResponseMessage | None:
        started_task: Task | None = None
        try:
            async for event in client.send_message(message):
                result, started_task = self._process_event(event)
                if not started_task:
                    return result
                break

        except (httpx.ConnectError, A2AClientHTTPError) as e:
            if not started_task:
                if not self._agent_card:
                    raise A2aClientError("Failed to connect to the agent: agent card not found") from e
                raise A2aClientError(f"Failed to connect to the agent: {pformat(self._agent_card.model_dump())}") from e

        connection_attemps, started_task = 1, cast(Task, started_task)
        while connection_attemps < self._max_reconnects:
            try:
                task = await client.get_task(TaskQueryParams(id=started_task.id))

            except (httpx.ConnectError, A2AClientHTTPError) as e:
                connection_attemps += 1
                if connection_attemps >= self._max_reconnects:
                    if not self._agent_card:
                        raise A2aClientError("Failed to connect to the agent: agent card not found") from e
                    raise A2aClientError(
                        f"Failed to connect to the agent: {pformat(self._agent_card.model_dump())}"
                    ) from e

            else:
                if _is_task_completed(task):
                    return response_message_from_a2a_task(task)

                await asyncio.sleep(self._polling_interval)

        return None

    def _process_event(self, event: ClientEvent | Message) -> tuple[ResponseMessage | None, Task | None]:
        if isinstance(event, Message):
            return response_message_from_a2a_message(event), None

        task, _ = event
        if _is_task_completed(task):
            return response_message_from_a2a_task(task), None

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
        card: AgentCard | None = None

        try:
            logger.info(
                f"Attempting to fetch public agent card from: {self._card_resolver.base_url}{AGENT_CARD_WELL_KNOWN_PATH}"
            )

            try:
                card = await self._card_resolver.get_agent_card(relative_card_path=AGENT_CARD_WELL_KNOWN_PATH)
            except A2AClientHTTPError as e_public:
                if e_public.status_code == 404:
                    logger.info(
                        f"Attempting to fetch public agent card from: {self._card_resolver.base_url}{PREV_AGENT_CARD_WELL_KNOWN_PATH}"
                    )
                    card = await self._card_resolver.get_agent_card(relative_card_path=PREV_AGENT_CARD_WELL_KNOWN_PATH)
                else:
                    raise e_public

            if card.supports_authenticated_extended_card:
                try:
                    card = await self._card_resolver.get_agent_card(
                        relative_card_path=EXTENDED_AGENT_CARD_PATH,
                        http_kwargs=auth_http_kwargs,
                    )
                except Exception as e_extended:
                    logger.warning(
                        f"Failed to fetch extended agent card: {e_extended}. Will proceed with public card.",
                        exc_info=True,
                    )

        except Exception as e:
            raise A2aAgentNotFoundError(f"{self.name}: {self._card_resolver.base_url}") from e

        return card


def _is_task_completed(task: Task) -> bool:
    if task.status.state is TaskState.failed:
        raise A2aClientError(f"Task failed: {pformat(task.model_dump())}")

    if task.status.state is TaskState.rejected:
        raise A2aClientError(f"Task rejected: {pformat(task.model_dump())}")

    return task.status.state in (
        TaskState.completed,
        TaskState.canceled,
        TaskState.input_required,
    )
