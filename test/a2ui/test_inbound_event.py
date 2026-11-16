# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Each client→server interaction in a request reaches the agent's observers as an A2UIClientEvent."""

import pytest

from .seam import serve_turn

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
async def test_client_action_emits_one_client_event() -> None:
    turn = await serve_turn("Confirmed.", body={"messages": [], "a2ui": [_ACTION_ENVELOPE]})

    assert len(turn.client_events) == 1
    assert turn.client_events[0].interaction.action.name == "confirm"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_request_without_interactions_emits_no_client_event() -> None:
    turn = await serve_turn("Hello.")

    assert turn.client_events == []
