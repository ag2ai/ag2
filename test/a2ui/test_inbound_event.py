# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The inbound middleware surfaces each client→server interaction as an A2UIClientEvent."""

import pytest

from ag2 import Agent
from ag2.a2ui import A2UIClientEvent
from ag2.a2ui.incoming import A2UIIncomingActionResult, parse_incoming_interactions
from ag2.a2ui.middleware import A2UIInboundMiddleware
from ag2.events import BaseEvent
from ag2.stream import MemoryStream
from ag2.testing import TestConfig

_ACTION_ENVELOPE = {
    "version": "v0.9",
    "action": {
        "name": "confirm",
        "surfaceId": "s1",
        "sourceComponentId": "btn",
        "timestamp": "2026-06-15T00:00:00Z",
        "context": {"id": 1},
    },
}


@pytest.mark.asyncio
async def test_inbound_middleware_emits_client_event_per_interaction() -> None:
    # arrange: parse a client action envelope into the typed interactions the
    # transports feed the inbound middleware.
    interactions = parse_incoming_interactions([_ACTION_ENVELOPE])
    assert len(interactions) == 1  # sanity: the envelope classified as an action

    received: list[A2UIClientEvent] = []
    stream = MemoryStream()

    @stream.subscribe
    async def _collect(event: BaseEvent) -> None:
        if isinstance(event, A2UIClientEvent):
            received.append(event)

    agent = Agent(name="ui", config=TestConfig("Confirmed."))

    # act: run a turn with the inbound middleware injected (as the transports do).
    await agent.ask("act", stream=stream, middleware=[A2UIInboundMiddleware(interactions)])

    # assert: one A2UIClientEvent carrying the parsed action result.
    assert len(received) == 1
    interaction = received[0].interaction
    assert isinstance(interaction, A2UIIncomingActionResult)
    assert interaction.action.name == "confirm"
