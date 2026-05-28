# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from a2a.server.agent_execution import AgentExecutor
from a2a.server.routes.rest_routes import create_rest_routes
from a2a.server.tasks import (
    PushNotificationConfigStore,
    PushNotificationSender,
    TaskStore,
)
from a2a.types import AgentCard
from starlette.applications import Starlette
from starlette.routing import BaseRoute

from ._common import (
    DEFAULT_AGENT_CARD_PATH,
    LEGACY_AGENT_CARD_PATH,
    CardModifier,
    ExtendedCardModifier,
    build_card_routes_with_legacy,
    build_default_handler,
    clone_card_with_capabilities,
)


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
    """Starlette ASGI app exposing REST dispatch + agent-card discovery.

    Card routes MUST precede REST routes: the REST router includes a
    ``Mount(path='/{tenant}', ...)`` catch-all that swallows any
    ``/<segment>/...`` path (including ``/.well-known/agent-card.json``).
    """
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
    routes: list[BaseRoute] = build_card_routes_with_legacy(
        agent_card,
        card_modifier=card_modifier,
        card_url=card_url,
        legacy_card_url=legacy_card_url,
    )
    routes.extend(create_rest_routes(handler, path_prefix=path_prefix))
    return Starlette(routes=routes)
