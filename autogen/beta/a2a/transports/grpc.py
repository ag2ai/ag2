# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

import grpc
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.request_handlers.grpc_handler import GrpcHandler
from a2a.server.tasks import (
    InMemoryTaskStore,
    PushNotificationConfigStore,
    PushNotificationSender,
    TaskStore,
)
from a2a.types import AgentCard, a2a_pb2_grpc

from ._http import ExtendedCardModifier


def default_grpc_channel_factory(url: str) -> grpc.aio.Channel:
    """Default insecure ``grpc.aio.Channel`` factory.

    Strips ``grpc://`` / ``grpc+insecure://`` scheme prefixes if present —
    some servers declare interface URLs with a scheme even though gRPC
    addresses are conventionally bare ``host:port``. The SDK passes the
    raw ``AgentInterface.url`` here, so we normalise.

    Insecure only — TLS lands in a follow-up alongside cert handling.
    """
    for prefix in ("grpc+insecure://", "grpc://"):
        if url.startswith(prefix):
            url = url[len(prefix) :]
            break
    return grpc.aio.insecure_channel(url)


def build_grpc_server(
    *,
    agent_executor: AgentExecutor,
    agent_card: AgentCard,
    bind: str,
    extended_agent_card: AgentCard | None = None,
    extended_card_modifier: "ExtendedCardModifier | None" = None,
    task_store: TaskStore | None = None,
    push_config_store: PushNotificationConfigStore | None = None,
    push_sender: PushNotificationSender | None = None,
    options: Sequence[tuple[str, Any]] = (),
) -> grpc.aio.Server:
    """Assemble a ``grpc.aio.Server`` exposing the A2A service for an agent.

    Counterpart to ``build_jsonrpc_asgi`` / ``build_rest_asgi`` for the
    gRPC transport. Wires up the same ``DefaultRequestHandlerV2`` pipeline
    behind the SDK's ``GrpcHandler`` servicer and binds it to ``bind``
    (e.g. ``"0.0.0.0:50051"``).

    The returned server is **not** started — callers do
    ``await server.start()`` and ``await server.wait_for_termination()``
    themselves so they can compose with other async lifecycles.

    Insecure binding only in this iteration; TLS support is a follow-up
    that will land alongside cert handling.
    """
    if extended_agent_card is not None:
        agent_card.capabilities.extended_agent_card = True
    if push_config_store is not None:
        agent_card.capabilities.push_notifications = True

    store = task_store or InMemoryTaskStore()
    handler = DefaultRequestHandlerV2(
        agent_executor=agent_executor,
        task_store=store,
        agent_card=agent_card,
        extended_agent_card=extended_agent_card,
        extended_card_modifier=extended_card_modifier,
        push_config_store=push_config_store,
        push_sender=push_sender,
    )
    server = grpc.aio.server(options=list(options) if options else None)
    servicer = GrpcHandler(handler)
    a2a_pb2_grpc.add_A2AServiceServicer_to_server(servicer, server)
    server.add_insecure_port(bind)
    return server
