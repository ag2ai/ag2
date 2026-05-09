# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from a2a.types import AgentCard


def clone_card_with_capabilities(
    card: AgentCard,
    *,
    extended: bool,
    push: bool,
) -> AgentCard:
    """Return a deep copy of ``card`` with capability flags optionally flipped.

    Transport builders advertise extended-card / push support based on the
    wiring they receive, but they must not mutate the user-provided card
    in place — that surprises callers who reuse a single card across
    builds and breaks multi-transport setups where each transport has
    different push wiring.

    Lives in its own module to keep the dependency graph acyclic:
    ``_http`` imports ``default_grpc_channel_factory`` from ``grpc``, so
    ``grpc`` cannot import from ``_http`` at runtime.
    """
    new_card = AgentCard()
    new_card.CopyFrom(card)
    if extended:
        new_card.capabilities.extended_agent_card = True
    if push:
        new_card.capabilities.push_notifications = True
    return new_card
