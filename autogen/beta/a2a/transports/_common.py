# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable
from typing import TypeAlias

from a2a.server.agent_execution import AgentExecutor
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.tasks import (
    InMemoryTaskStore,
    PushNotificationConfigStore,
    PushNotificationSender,
    TaskStore,
)
from a2a.types import AgentCard
from starlette.routing import BaseRoute

CardModifier: TypeAlias = Callable[[AgentCard], Awaitable[AgentCard]]
ExtendedCardModifier: TypeAlias = Callable[[AgentCard, ServerCallContext], Awaitable[AgentCard]]

# Legacy v0.x server-side card alias. Kept so pre-v1 clients still discover the card.
LEGACY_AGENT_CARD_PATH = "/.well-known/agent.json"
DEFAULT_AGENT_CARD_PATH = "/.well-known/agent-card.json"


def clone_card_with_capabilities(card: AgentCard, *, extended: bool, push: bool) -> AgentCard:
    """Deep-copy ``card`` with capability flags flipped — never mutate caller's card."""
    new_card = AgentCard()
    new_card.CopyFrom(card)
    if extended:
        new_card.capabilities.extended_agent_card = True
    if push:
        new_card.capabilities.push_notifications = True
    return new_card


def build_default_handler(
    *,
    agent_executor: AgentExecutor,
    agent_card: AgentCard,
    extended_agent_card: AgentCard | None,
    extended_card_modifier: ExtendedCardModifier | None,
    task_store: TaskStore | None,
    push_config_store: PushNotificationConfigStore | None,
    push_sender: PushNotificationSender | None,
) -> DefaultRequestHandlerV2:
    """Build the SDK request handler shared by all transports."""
    return DefaultRequestHandlerV2(
        agent_executor=agent_executor,
        task_store=task_store or InMemoryTaskStore(),
        agent_card=agent_card,
        extended_agent_card=extended_agent_card,
        extended_card_modifier=extended_card_modifier,
        push_config_store=push_config_store,
        push_sender=push_sender,
    )


def build_card_routes_with_legacy(
    agent_card: AgentCard,
    *,
    card_modifier: CardModifier | None,
    card_url: str,
    legacy_card_url: str | None,
) -> list[BaseRoute]:
    """Card routes at v1.x ``card_url`` plus optional v0.x alias at ``legacy_card_url``."""
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
    return routes
