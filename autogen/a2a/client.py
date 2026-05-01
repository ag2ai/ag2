# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from pprint import pformat
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, Client, ClientCallInterceptor, ClientConfig
from a2a.client import ClientFactory as A2AClientFactory
from a2a.client.errors import A2AClientError
from a2a.compat.v0_3.conversions import (
    to_compat_agent_card,
    to_compat_message,
    to_compat_task,
    to_compat_task_artifact_update_event,
    to_compat_task_status_update_event,
    to_core_agent_card,
    to_core_message,
)
from a2a.compat.v0_3.types import (
    AgentCard,
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)
from a2a.types import GetTaskRequest, SendMessageRequest, StreamResponse, SubscribeToTaskRequest
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from typing_extensions import Self

from autogen import ConversableAgent
from autogen.agentchat.remote import RequestMessage, ResponseMessage
from autogen.doc_utils import export_module
from autogen.events.agent_events import TerminationEvent
from autogen.io.base import IOStream
from autogen.oai.client import OpenAIWrapper

from .client_factory import ClientFactory, EmptyClientFactory
from .errors import A2aAgentNotFoundError, A2aClientError
from .utils import (
    request_message_to_a2a,
    response_message_from_a2a_message,
    response_message_from_a2a_task,
    update_artifact_to_streaming,
)

logger = logging.getLogger(__name__)

# Stream event yielded by ``_ask_streaming`` / ``_ask_polling``.
# Mirrors the legacy a2a-sdk shape: either a standalone ``Message`` event,
# or a ``(Task, status/artifact-update)`` pair sharing a common task object.
ClientEvent = tuple[Task, TaskStatusUpdateEvent | TaskArtifactUpdateEvent | None]


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
        # ``accepted_output_modes`` is required on the wire by v0.3-format
        # servers (e.g. fasta2a, used by pydantic-ai). a2a-sdk's default of
        # ``[]`` causes those servers to reject the request as malformed.
        if not self._client_config.accepted_output_modes:
            self._client_config.accepted_output_modes = ["text"]
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
        extra_parts: list[Any] | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        if messages is None:
            messages = self._oai_messages[sender]

        if not self._agent_card:
            self._agent_card = await self._get_agent_card()

        context_id = uuid4().hex

        self._client_config.httpx_client = self._httpx_client_factory()
        async with self._client_config.httpx_client:
            agent_client = A2AClientFactory(self._client_config).create(
                to_core_agent_card(self._agent_card),
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
                    extra_parts=extra_parts,
                )

                if self._agent_card.capabilities.streaming:
                    a2a_stream = self._ask_streaming(agent_client, initial_message)
                else:
                    a2a_stream = self._ask_polling(agent_client, initial_message)

                io_stream = IOStream.get_default()

                reply: ResponseMessage | None = None
                # do not break stream to let HTTPX client close correctly
                async for a2a_event in a2a_stream:
                    if isinstance(a2a_event, Message):
                        reply = response_message_from_a2a_message(a2a_event)

                    else:
                        task, ev = a2a_event
                        if isinstance(ev, TaskArtifactUpdateEvent):
                            for e in update_artifact_to_streaming(ev):
                                io_stream.send(e)

                        if _is_task_completed(task):
                            reply = response_message_from_a2a_task(task)

                if not reply:
                    return True, None

                messages = reply.messages
                if reply.input_required is not None:
                    user_input = await self.a_get_human_input(prompt=f"Input for `{self.name}`\n{reply.input_required}")

                    if user_input == "exit":
                        io_stream.send(
                            TerminationEvent(
                                termination_reason="User requested to end the conversation",
                                sender=self,
                                recipient=sender,
                            )
                        )
                        return True, None

                    messages.append({"content": user_input, "role": "user"})
                    continue

                if reply.context:
                    self.context_variables.update(reply.context)
                    if sender:
                        sender.context_variables.update(reply.context)

                return True, reply.messages[-1]

    def _connection_error(self, exc: Exception) -> "A2aClientError":
        if not self._agent_card:
            return A2aClientError(f"Failed to connect to the agent: agent card not found. {exc}")
        return A2aClientError(
            f"Failed to connect to the agent {self._agent_card.name!r} at {self._agent_card.url}: {exc}"
        )

    async def _ask_streaming(self, client: Client, message: Message) -> AsyncIterator[ClientEvent | Message]:
        request = SendMessageRequest(message=to_core_message(message))
        started_task: Task | None = None
        completed = False
        try:
            async for compat_event in _adapt_stream_response(client.send_message(request)):
                if isinstance(compat_event, Message):
                    yield compat_event
                    completed = True
                else:
                    task, _ev = compat_event
                    started_task = task
                    yield compat_event
                    completed = _is_task_completed(task)

        except (httpx.ConnectError, A2AClientError) as e:
            if not started_task:
                raise self._connection_error(e) from e

        if not completed:
            if not started_task:
                raise self._connection_error(RuntimeError("stream ended without producing a task"))

            connection_attempts = 1
            while not completed and connection_attempts < self._max_reconnects:
                try:
                    raw = client.subscribe(SubscribeToTaskRequest(id=started_task.id))
                    async for compat_event in _adapt_stream_response(raw, current_task=started_task):
                        if isinstance(compat_event, Message):
                            yield compat_event
                            completed = True
                        else:
                            task, _ev = compat_event
                            started_task = task
                            yield compat_event
                            completed = _is_task_completed(task)

                except (httpx.ConnectError, A2AClientError) as e:
                    connection_attempts += 1
                    if connection_attempts >= self._max_reconnects:
                        raise self._connection_error(e) from e

    async def _ask_polling(self, client: Client, message: Message) -> AsyncIterator[ClientEvent | Message]:
        request = SendMessageRequest(message=to_core_message(message))
        started_task: Task | None = None
        completed = False
        try:
            async for compat_event in _adapt_stream_response(client.send_message(request)):
                if isinstance(compat_event, Message):
                    yield compat_event
                    completed = True
                else:
                    task, _ev = compat_event
                    started_task = task
                    yield compat_event
                    if _is_task_completed(task):
                        completed = True

        except (httpx.ConnectError, A2AClientError) as e:
            if not started_task:
                raise self._connection_error(e) from e

        if not completed:
            if not started_task:
                raise self._connection_error(RuntimeError("stream ended without producing a task"))

            connection_attempts = 1
            while not completed and connection_attempts < self._max_reconnects:
                try:
                    task = to_compat_task(await client.get_task(GetTaskRequest(id=started_task.id)))
                    completed = _is_task_completed(task)

                except (httpx.ConnectError, A2AClientError) as e:
                    connection_attempts += 1
                    if connection_attempts >= self._max_reconnects:
                        raise self._connection_error(e) from e

                else:
                    yield task, None
                    await asyncio.sleep(self._polling_interval)

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
        try:
            logger.info(f"Attempting to fetch public agent card from: {self.url}{AGENT_CARD_WELL_KNOWN_PATH}")
            proto_card = await self._card_resolver.get_agent_card(relative_card_path=AGENT_CARD_WELL_KNOWN_PATH)
        except Exception as e:
            raise A2aAgentNotFoundError(f"{self.name}: {self.url}") from e

        return to_compat_agent_card(proto_card)


async def _adapt_stream_response(
    raw: AsyncIterator[StreamResponse],
    *,
    current_task: Task | None = None,
) -> AsyncIterator[ClientEvent | Message]:
    """Translate proto ``StreamResponse`` events into the legacy compat-typed shape.

    A2A 1.0 emits standalone status/artifact update events keyed only by
    ``task_id``; the legacy autogen consumer expects ``(Task, event)`` pairs that
    share a single task object whose ``status`` and ``artifacts`` mirror the
    latest updates. We keep a rolling ``current_task`` and patch its
    ``status`` (from status_update events) and ``artifacts`` (from
    artifact_update events) as they arrive so ``response_message_from_a2a_task``
    can read the accumulated state when the task hits a terminal status.
    """
    async for resp in raw:
        kind = resp.WhichOneof("payload")
        if kind == "message":
            yield to_compat_message(resp.message)
        elif kind == "task":
            current_task = to_compat_task(resp.task)
            yield current_task, None
        elif kind == "status_update":
            status_event = to_compat_task_status_update_event(resp.status_update)
            if current_task is not None:
                current_task.status = status_event.status
            assert current_task is not None, "status_update received before task event"
            yield current_task, status_event
        elif kind == "artifact_update":
            artifact_event = to_compat_task_artifact_update_event(resp.artifact_update)
            assert current_task is not None, "artifact_update received before task event"
            _merge_artifact_update(current_task, artifact_event)
            yield current_task, artifact_event


def _merge_artifact_update(task: Task, event: TaskArtifactUpdateEvent) -> None:
    """Apply an ``artifact_update`` to ``task.artifacts`` in place.

    A2A 1.0 streams artifacts as deltas: ``append=True`` extends the parts of
    a previously seen artifact, ``append=False`` (or unset) replaces it.
    Without this merge ``task.artifacts`` stays empty across the stream and
    the consumer thinks the agent produced no reply when the task completes.
    """
    if task.artifacts is None:
        task.artifacts = []

    incoming = event.artifact
    for existing in task.artifacts:
        if existing.artifact_id == incoming.artifact_id:
            if event.append:
                existing.parts = [*existing.parts, *incoming.parts]
            else:
                existing.parts = list(incoming.parts)
            if incoming.metadata is not None:
                existing.metadata = incoming.metadata
            if incoming.name:
                existing.name = incoming.name
            return

    task.artifacts.append(incoming)


def _is_event_completed(event: ClientEvent | Message) -> bool:
    if isinstance(event, Message):
        return True
    return _is_task_completed(event[0])


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
