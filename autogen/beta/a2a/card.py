# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from a2a.types import AgentCapabilities, AgentCard

if TYPE_CHECKING:
    from autogen.beta import Agent


def build_card(agent: "Agent", *, url: str, supports_extended: bool = False) -> AgentCard:
    """Build a default `AgentCard` for the given AG2 beta agent.

    The card carries the agent's name, the joined system prompt as the
    description, and declares streaming support. Skills are left empty —
    callers can supply their own pre-built `AgentCard` to `A2AServer(card=...)`
    if they need richer metadata.

    `supports_extended=True` flips `supports_authenticated_extended_card` so
    A2A clients know to also fetch `/agent/authenticatedExtendedCard`.
    """
    description = "\n".join(agent._system_prompt) if agent._system_prompt else "AG2 beta agent"
    return AgentCard(
        name=agent.name,
        description=description,
        url=url,
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
        supports_authenticated_extended_card=True if supports_extended else None,
    )
