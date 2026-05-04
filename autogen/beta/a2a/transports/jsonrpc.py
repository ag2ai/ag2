# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import (
    InMemoryTaskStore,
    PushNotificationConfigStore,
    PushNotificationSender,
    TaskStore,
)
from a2a.types import AgentCard
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import BaseRoute

from ._http import (
    DEFAULT_AGENT_CARD_PATH,
    LEGACY_AGENT_CARD_PATH,
    CardModifier,
    ExtendedCardModifier,
    fetch_card,
    make_a2a_client,
    make_httpx_client,
)

if TYPE_CHECKING:
    from a2a.server.agent_execution import AgentExecutor

__all__ = (
    "DEFAULT_AGENT_CARD_PATH",
    "LEGACY_AGENT_CARD_PATH",
    "CardModifier",
    "ExtendedCardModifier",
    "build_jsonrpc_asgi",
    "build_jsonrpc_routes",
    "fetch_card",
    "make_a2a_client",
    "make_httpx_client",
)


def build_jsonrpc_routes(
    *,
    handler: DefaultRequestHandlerV2,
    rpc_url: str = "/",
) -> list[BaseRoute]:
    """Return Starlette routes that dispatch JSON-RPC 2.0 requests to ``handler``."""
    return list(create_jsonrpc_routes(handler, rpc_url=rpc_url))


def build_jsonrpc_asgi(
    *,
    agent_executor: "AgentExecutor",
    agent_card: AgentCard,
    extended_agent_card: AgentCard | None = None,
    card_modifier: CardModifier | None = None,
    extended_card_modifier: ExtendedCardModifier | None = None,
    middlewares: Sequence[tuple[type, Mapping[str, Any]]] = (),
    task_store: TaskStore | None = None,
    push_config_store: PushNotificationConfigStore | None = None,
    push_sender: PushNotificationSender | None = None,
    rpc_url: str = "/",
    card_url: str = DEFAULT_AGENT_CARD_PATH,
    legacy_card_url: str | None = LEGACY_AGENT_CARD_PATH,
) -> Starlette:
    """Assemble a Starlette ASGI app exposing JSON-RPC routes for an A2A agent.

    This is the server-side counterpart to ``make_a2a_client``. It wires up
    ``DefaultRequestHandlerV2`` with our ``AgentExecutor`` and registers the
    JSON-RPC dispatcher plus the agent-card discovery endpoint.

    ``extended_agent_card``, when supplied, flips ``capabilities.extended_agent_card``
    on the public card and is served via the JSON-RPC ``GetExtendedAgentCard``
    method (the SDK 1.x flow — there is no separate REST endpoint for it).

    ``card_modifier`` is awaited per-request before serving the public
    card; ``extended_card_modifier`` is awaited per-request before
    serving the extended card.

    ``legacy_card_url`` registers a second card route on the v0.x path
    for backward compatibility. Pass ``None`` to disable.

    ``middlewares`` is a list of ``(class, kwargs)`` pairs applied to
    the resulting Starlette app at runtime — accumulated by
    ``A2AServer.add_middleware``.
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
    routes: list[BaseRoute] = build_jsonrpc_routes(handler=handler, rpc_url=rpc_url)
    routes.extend(
        create_agent_card_routes(agent_card, card_modifier=card_modifier, card_url=card_url),
    )
    if legacy_card_url:
        routes.extend(
            create_agent_card_routes(
                agent_card,
                card_modifier=card_modifier,
                card_url=legacy_card_url,
            ),
        )

    starlette_middleware = [Middleware(cls, **dict(kwargs)) for cls, kwargs in middlewares]
    return Starlette(routes=routes, middleware=starlette_middleware)
