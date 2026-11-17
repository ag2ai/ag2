# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The A2UI rules of one turn, shared by every transport (ADR 0042).

A transport decodes what the renderer sent (:func:`~ag2.a2ui.incoming.read_renderer_input`), opens the turn with
:func:`open_turn`, and runs its agent with the turn's prompt and middleware. The turn decides what the agent is
offered, answers the interactions the server handles itself, and filters what may be sent back. Running the agent
stays with the transport.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ag2.context import Stream
from ag2.middleware.base import Middleware

from ._runtime import A2UIOffer, _A2UIRuntime
from ._types import JsonValue, ServerToClientMessage
from .capabilities import A2UIClientCapabilities, capabilities_to_prompt
from .data_model import data_model_to_prompt
from .events import A2UIMessageEvent
from .incoming import A2UIRendererInput
from .middleware import A2UIInboundMiddleware
from .server_action import run_client_interactions

if TYPE_CHECKING:
    from ag2.agent import Agent


@dataclass(frozen=True, slots=True)
class A2UIServerAnswers:
    """What the server answered on its own: the messages to send, and whether it handled any interaction."""

    messages: list[ServerToClientMessage]
    handled: bool

    def complete_without_agent(self, *, has_input: bool) -> bool:
        """Whether the turn is done without the agent: it only carried interactions the server handled."""
        return self.handled and not has_input


@dataclass(frozen=True, slots=True)
class A2UITurn:
    """One turn's A2UI behaviour: the negotiated offer, the prompt fragments and middleware it adds to the agent's
    run, the server-side answers, and the filter on what reaches the renderer."""

    runtime: _A2UIRuntime
    renderer: A2UIRendererInput
    offer: A2UIOffer

    @property
    def system_prompt(self) -> tuple[str, ...]:
        """The A2UI section of the turn's offer, the renderer's capabilities and its data model, in that order."""
        fragments = (
            self.offer.system_prompt_section,
            _capabilities_prompt(self.renderer.capabilities, self.offer),
            data_model_to_prompt(self.renderer.data_model),
        )
        return tuple(fragment for fragment in fragments if fragment)

    @property
    def middleware(self) -> list[Middleware]:
        """The offer's extraction / validation middleware, plus one surfacing the renderer's interactions."""
        if not self.renderer.interactions:
            return [self.offer.middleware]
        return [self.offer.middleware, Middleware(A2UIInboundMiddleware, interactions=self.renderer.interactions)]

    async def answer_interactions(self, agent: "Agent", variables: dict[str, JsonValue] | None) -> A2UIServerAnswers:
        """Run the interactions the server answers itself (registered clicks, ``callAgentFunction``)."""
        messages, handled = await run_client_interactions(
            self.renderer.interactions,
            self.runtime,
            agent,
            offer=self.offer,
            variables=variables,
        )
        return A2UIServerAnswers(messages, handled)

    def collect(self, stream: Stream) -> list[ServerToClientMessage]:
        """The list the agent's validated A2UI messages on ``stream`` are appended to as they are sent."""
        messages: list[ServerToClientMessage] = []
        stream.where(A2UIMessageEvent).subscribe(A2UIMessageCollector(messages))
        return messages

    def deliverable(self, messages: list[ServerToClientMessage]) -> list[ServerToClientMessage]:
        """The agent's ``messages`` this turn may send; the rest are dropped with a warning."""
        return self.offer.deliverable(messages, source="the agent")


def _capabilities_prompt(caps: A2UIClientCapabilities | None, offer: A2UIOffer) -> str:
    """Render negotiated client capabilities as a per-turn prompt fragment."""
    if not offer.sends_ui or offer.catalog_id is None:
        return ""
    return capabilities_to_prompt(caps, catalog_id=offer.catalog_id)


class A2UIMessageCollector:
    """Appends each ``A2UIMessageEvent``'s message to ``into``; subscribe it to a turn's stream."""

    def __init__(self, into: list[ServerToClientMessage]) -> None:
        self.into = into

    async def __call__(self, event: A2UIMessageEvent) -> None:
        self.into.append(event.message)


def open_turn(runtime: _A2UIRuntime, renderer: A2UIRendererInput) -> A2UITurn:
    """Open a turn for what the renderer sent: offer only the catalogs it supports (none in common: no A2UI)."""
    return A2UITurn(runtime=runtime, renderer=renderer, offer=runtime.offer_for(renderer.capabilities))
