# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Starlette route helpers shared by the JSON-RPC and REST transports.

Kept apart from ``_common.py`` so that the transport-agnostic card and handler
helpers stay importable without ``starlette`` — a gRPC-only server has no HTTP
routes and must not be told to install ``a2a-sdk[http-server]``.
"""

from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.types import AgentCard
from starlette.routing import BaseRoute

from ._common import CardModifier


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
