# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

try:
    import faststream.nats  # noqa: F401
except ImportError as e:
    raise ImportError('NATS Transport is not installed. Please install it with:\npip install "faststream[nats]"') from e

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Annotated, Any, TypeAlias
from uuid import uuid4

from a2a.client import ClientCallInterceptor, ClientConfig
from a2a.client.errors import A2AClientTimeoutError
from a2a.client.middleware import ClientCallContext
from a2a.client.transports.base import ClientTransport
from a2a.server.request_handlers import RequestHandler
from a2a.types import (
    AgentCard,
    AgentInterface,
    GetTaskPushNotificationConfigParams,
    Message,
    MessageSendParams,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskStatusUpdateEvent,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, PREV_AGENT_CARD_WELL_KNOWN_PATH
from faststream import Header
from faststream.asgi import AsgiFastStream, AsgiResponse
from faststream.nats import NatsBroker
from pydantic import Field, TypeAdapter

MessageType: TypeAlias = Message | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
parser = TypeAdapter[MessageType](MessageType)


REPLY_HEADER = "a2a-reply"


class NatsInterface(AgentInterface):
    transport: str = "nats"
    url: str = Field(
        ...,
        description="The subject agent is listening to.",
        examples=["agent-name.subject"],
    )


def make_nats_app(
    *,
    broker: "NatsBroker",
    subject: str,
    request_handler: RequestHandler,
    card: AgentCard,
) -> Callable[..., Any]:
    @broker.subscriber(subject)
    async def process_a2a_request(
        params: MessageSendParams,
        reply_target: Annotated[str, Header(REPLY_HEADER)],
    ) -> None:
        async for event in request_handler.on_message_send_stream(params):
            await broker.publish(event, reply_target)

        # Final message to indicate that the request is complete
        await broker.publish(None, reply_target)

    # serve agent cards as regular HTTP routes
    card_json = AsgiResponse(
        card.model_dump_json(exclude_none=True, exclude_defaults=True).encode(),
        headers={"content-type": "application/json"},
    )

    return AsgiFastStream(
        broker,
        asgi_routes=[
            (AGENT_CARD_WELL_KNOWN_PATH, card_json),
            (PREV_AGENT_CARD_WELL_KNOWN_PATH, card_json),
        ],
    )


class NatsTransport:
    """NATS transport for A2A client.

    Follows `a2a.client.client_factory.TransportProducer` protocol."""

    def __init__(
        self,
        broker: NatsBroker,
        *,
        timeout: float | None = 30,
    ) -> None:
        self.broker = broker
        self.timeout = timeout

    def __call__(
        self,
        card: AgentCard,
        url: str,
        client_config: ClientConfig,
        interceptors: list[ClientCallInterceptor],
    ) -> ClientTransport:
        return _NatsTransportImpl(
            self.broker,
            self.timeout,
            card,
            url,
            client_config,
            interceptors,
        )


class _NatsTransportImpl(ClientTransport):
    def __init__(
        self,
        broker: NatsBroker,
        timeout: float | None,
        card: AgentCard,
        url: str,
        client_config: ClientConfig,
        interceptors: list[ClientCallInterceptor],
    ) -> None:
        self.broker = broker
        self.timeout = timeout
        self.addr = str(uuid4())
        self.targer_subject = url

    async def send_message_streaming(
        self,
        request: MessageSendParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> AsyncGenerator[MessageType, None]:
        async with self.broker as br:
            subscriber = br.subscriber(self.addr)
            await br.start()

            await br.publish(
                request,
                self.targer_subject,
                headers={REPLY_HEADER: self.addr},
            )

            iterator = subscriber.__aiter__()
            while True:
                try:
                    msg = await asyncio.wait_for(
                        iterator.__anext__(),
                        timeout=self.timeout,
                    )
                except asyncio.TimeoutError as e:
                    raise A2AClientTimeoutError(f"Timeout waiting for message from {self.addr}") from e

                if not msg.body:
                    break
                yield parser.validate_json(msg.body)

    async def resubscribe(
        self,
        request: TaskIdParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> AsyncGenerator[MessageType, None]:
        async with self.broker as br:
            subscriber = br.subscriber(self.addr)

            iterator = subscriber.__aiter__()
            while True:
                try:
                    msg = await asyncio.wait_for(
                        iterator.__anext__(),
                        timeout=self.timeout,
                    )
                except asyncio.TimeoutError as e:
                    raise A2AClientTimeoutError(f"Timeout waiting for message from {self.addr}") from e

                if not msg.body:
                    break
                yield parser.validate_json(msg.body)

    async def send_message(
        self,
        request: MessageSendParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> Task | Message:
        raise NotImplementedError

    async def get_task(
        self,
        request: TaskQueryParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> Task:
        raise NotImplementedError

    async def cancel_task(
        self,
        request: TaskIdParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> Task:
        raise NotImplementedError

    async def set_task_callback(
        self,
        request: TaskPushNotificationConfig,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> TaskPushNotificationConfig:
        raise NotImplementedError

    async def get_task_callback(
        self,
        request: GetTaskPushNotificationConfigParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> TaskPushNotificationConfig:
        raise NotImplementedError

    async def get_card(
        self,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
        signature_verifier: Callable[[AgentCard], None] | None = None,
    ) -> AgentCard:
        raise NotImplementedError

    async def close(self) -> None:
        pass
