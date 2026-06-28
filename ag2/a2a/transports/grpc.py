# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

import grpc
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers.grpc_handler import GrpcHandler
from a2a.server.tasks import (
    PushNotificationConfigStore,
    PushNotificationSender,
    TaskStore,
)
from a2a.types import AgentCard, a2a_pb2_grpc

from ._common import ExtendedCardModifier, build_default_handler, clone_card_with_capabilities


def default_grpc_channel_factory(url: str) -> grpc.aio.Channel:
    """Insecure ``grpc.aio.Channel`` factory; strips ``grpc(+insecure)://`` prefix."""
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
    extended_card_modifier: ExtendedCardModifier | None = None,
    task_store: TaskStore | None = None,
    push_config_store: PushNotificationConfigStore | None = None,
    push_sender: PushNotificationSender | None = None,
    options: Sequence[tuple[str, Any]] = (),
) -> grpc.aio.Server:
    """``grpc.aio.Server`` exposing A2A service; caller starts/awaits it."""
    agent_card = clone_card_with_capabilities(
        agent_card,
        extended=extended_agent_card is not None,
        push=push_config_store is not None,
    )
    handler = build_default_handler(
        agent_executor=agent_executor,
        agent_card=agent_card,
        extended_agent_card=extended_agent_card,
        extended_card_modifier=extended_card_modifier,
        task_store=task_store,
        push_config_store=push_config_store,
        push_sender=push_sender,
    )
    server = grpc.aio.server(options=list(options) if options else None)
    a2a_pb2_grpc.add_A2AServiceServicer_to_server(GrpcHandler(handler), server)
    server.add_insecure_port(bind)
    return server
