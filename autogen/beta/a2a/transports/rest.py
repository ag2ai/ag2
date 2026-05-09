# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.rest_routes import create_rest_routes
from a2a.server.tasks import (
    InMemoryTaskStore,
    PushNotificationConfigStore,
    PushNotificationSender,
    TaskStore,
)
from a2a.types import AgentCard
from starlette.applications import Starlette
from starlette.routing import BaseRoute

from ._card import clone_card_with_capabilities
from ._http import DEFAULT_AGENT_CARD_PATH, LEGACY_AGENT_CARD_PATH
from .jsonrpc import CardModifier, ExtendedCardModifier


def build_rest_routes(
    *,
    handler: DefaultRequestHandlerV2,
    path_prefix: str = "",
) -> list[BaseRoute]:
    """Return Starlette routes that dispatch A2A REST requests to ``handler``.

    REST exposes one route per A2A method (e.g. ``POST /v1/message:send``,
    ``POST /v1/tasks/{id}:cancel``) — different from JSON-RPC which uses a
    single envelope endpoint.
    """
    return list(create_rest_routes(handler, path_prefix=path_prefix))


def build_rest_asgi(
    *,
    agent_executor: AgentExecutor,
    agent_card: AgentCard,
    extended_agent_card: AgentCard | None = None,
    card_modifier: CardModifier | None = None,
    extended_card_modifier: ExtendedCardModifier | None = None,
    task_store: TaskStore | None = None,
    push_config_store: PushNotificationConfigStore | None = None,
    push_sender: PushNotificationSender | None = None,
    path_prefix: str = "",
    card_url: str = DEFAULT_AGENT_CARD_PATH,
    legacy_card_url: str | None = LEGACY_AGENT_CARD_PATH,
) -> Starlette:
    """Assemble a Starlette ASGI app exposing REST routes for an A2A agent.

    The REST counterpart to ``build_jsonrpc_asgi``. Same handler shape
    (``DefaultRequestHandlerV2`` over ``AgentExecutor``), different
    dispatcher — one HTTP route per A2A method instead of a single
    JSON-RPC envelope endpoint.

    ``path_prefix`` lets callers mount REST under e.g. ``/api/v1`` if they
    need it; default is empty (routes live at the application root).

    See ``build_jsonrpc_asgi`` for parameter semantics; everything except
    ``rpc_url``/``path_prefix`` and the dispatcher is identical. Add
    Starlette middleware via ``app.add_middleware(...)`` after building.
    """
    agent_card = clone_card_with_capabilities(
        agent_card,
        extended=extended_agent_card is not None,
        push=push_config_store is not None,
    )

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
    # Card routes MUST precede REST routes: the REST router includes a
    # ``Mount(path='/{tenant}', ...)`` catch-all that swallows any
    # ``/<segment>/...`` path (including ``/.well-known/agent-card.json``,
    # treating ``.well-known`` as the tenant) and returns 404 from its
    # own subroutes before Starlette can fall through to the card route.
    routes: list[BaseRoute] = list(
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
    routes.extend(build_rest_routes(handler=handler, path_prefix=path_prefix))

    return Starlette(routes=routes)
