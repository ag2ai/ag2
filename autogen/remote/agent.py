# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, cast

import httpx

from autogen import ConversableAgent
from autogen.agentchat.group import ContextVariables
from autogen.doc_utils import export_module
from autogen.oai.client import OpenAIWrapper

from .errors import RemoteAgentError, RemoteAgentNotFoundError
from .protocol import RequestMessage, ResponseMessage
from .retry import EmptyRetryPolicy, RetryPolicy


@export_module("autogen.remote")
class HTTPRemoteAgent(ConversableAgent):
    def __init__(
        self,
        url: str,
        name: str,
        *,
        silent: bool = False,
        client: httpx.AsyncClient | httpx.Client | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.url = url
        self.retry_policy: RetryPolicy = retry_policy or EmptyRetryPolicy
        self._httpx_client = client

        super().__init__(name, silent=silent)

        self.__llm_config: dict[str, Any] = {}

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

        client = cast(httpx.Client, self._httpx_client) or httpx.Client(timeout=30)

        retry_policy = self.retry_policy()

        task_id: Any = None
        with client:
            while True:
                with retry_policy:
                    if task_id is None:
                        # initiate remote procedure
                        task_id = self._process_create_remote_task_response(
                            client.post(
                                f"{self.url}/{self.name}",
                                content=RequestMessage(
                                    messages=messages,
                                    context=self.context_variables.data,
                                    client_tools=self.__llm_config.get("tools", []),
                                ).model_dump_json(),
                            )
                        )

                    reply_response = client.get(f"{self.url}/{self.name}/{task_id}")

                    if reply_response.status_code in (200, 204):  # valid answer codes
                        break

                    if reply_response.status_code == 425:  # task still in progress
                        continue

                    if reply_response.status_code == 404:
                        task_id = None  # recreate task due remote agent lost it
                        continue

                    raise RemoteAgentError(f"Remote client error: {reply_response}, {reply_response.content!r}")

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

        client = cast(httpx.AsyncClient, self._httpx_client) or httpx.AsyncClient(timeout=30)

        retry_policy = self.retry_policy()

        task_id: Any = None
        async with client:
            while True:
                with retry_policy:
                    if task_id is None:
                        # initiate remote procedure
                        task_id = self._process_create_remote_task_response(
                            await client.post(
                                f"{self.url}/{self.name}",
                                content=RequestMessage(
                                    messages=messages,
                                    context=self.context_variables.data,
                                    client_tools=self.__llm_config.get("tools", []),
                                ).model_dump_json(),
                            )
                        )

                    reply_response = await client.get(f"{self.url}/{self.name}/{task_id}")

                    if reply_response.status_code in (200, 204):  # valid answer codes
                        break

                    if reply_response.status_code == 425:  # task still in progress
                        continue

                    if reply_response.status_code == 404:
                        task_id = None  # recreate task due remote agent lost it
                        continue

                    raise RemoteAgentError(f"Remote client error: {reply_response}, {reply_response.content!r}")

        if reply := self._process_remote_reply(reply_response):
            if sender:
                context_variables = ContextVariables(reply.context)
                sender.context_variables.update(context_variables.to_dict())
            # TODO: support multiple messages response for remote chat history
            return True, reply.messages[-1]

        return True, None

    def _process_create_remote_task_response(self, response: httpx.Response) -> Any:
        if response.status_code == 404:
            raise RemoteAgentNotFoundError(self.name)

        if response.status_code != 202:
            raise RemoteAgentError(f"Remote client error: {response}, {response.content!r}")

        return response.json()

    def _process_remote_reply(self, reply_response: httpx.Response) -> ResponseMessage | None:
        if reply_response.status_code == 204:
            return None

        try:
            serialized_message = ResponseMessage.model_validate_json(reply_response.content)

        except Exception as e:
            raise RemoteAgentError(f"Remote client error: {reply_response}, {reply_response.content!r}") from e

        return serialized_message

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
