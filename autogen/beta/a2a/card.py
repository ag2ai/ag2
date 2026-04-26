# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from a2a.types import AgentCapabilities, AgentCard

if TYPE_CHECKING:
    from autogen.beta import Agent


_DEFAULT_DESCRIPTION = "AG2 beta agent"
_DEFAULT_VERSION = "0.1.0"
_DEFAULT_INPUT_MODES = ("text",)
_DEFAULT_OUTPUT_MODES = ("text",)


def build_card(agent: "Agent", *, url: str) -> AgentCard:
    """Build a default `AgentCard` for the given AG2 beta agent.

    The card carries the agent's name, the joined system prompt as the
    description, and declares streaming support. Skills are left empty —
    callers can supply their own pre-built `AgentCard` to `A2AServer(card=...)`
    if they need richer metadata.
    """
    description = "\n".join(agent._system_prompt) if agent._system_prompt else _DEFAULT_DESCRIPTION
    return AgentCard(
        name=agent.name,
        description=description,
        url=url,
        version=_DEFAULT_VERSION,
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=list(_DEFAULT_INPUT_MODES),
        default_output_modes=list(_DEFAULT_OUTPUT_MODES),
        skills=[],
    )
