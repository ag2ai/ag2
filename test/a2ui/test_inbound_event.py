# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Each client→server interaction in a request reaches the agent's observers as an A2UIClientEvent."""

from typing import Any

import pytest

from ag2.events import ModelRequest, TextInput

from .seam import Turn, serve_turn

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


def _turn_text(turn: Turn) -> str:
    """The text of the user turn the model was first called with."""
    request = turn.model_calls[0].messages[-1]
    assert isinstance(request, ModelRequest)
    return " ".join(part.content for part in request.parts if isinstance(part, TextInput))


_ERROR_ENVELOPE_BODY = {"code": "VALIDATION_FAILED", "surfaceId": "s1", "message": "bad layout", "path": "/x"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("protocol_version", "envelope"),
    [
        pytest.param(
            "v0.9",
            {"version": "v0.9", "action": _ACTION_ENVELOPE["action"], "error": _ERROR_ENVELOPE_BODY},
            id="v0.9-action-and-error",
        ),
        pytest.param(
            "v1.0",
            {
                "version": "1.0",
                "action": _ACTION_ENVELOPE["action"],
                "rendererFunctionResponse": {"functionCallId": "fc-7", "value": "Berlin"},
            },
            id="v1.0-action-and-function-response",
        ),
    ],
)
async def test_envelope_with_more_than_one_payload_is_dropped_with_a_warning(
    protocol_version: str, envelope: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    # Upstream: a renderer→agent message carries exactly one of action / callAgentFunction /
    # rendererFunctionResponse / error. An envelope with two is not guessed at.
    with caplog.at_level("WARNING", logger="ag2.a2ui.incoming"):
        turn = await serve_turn(
            "Hello.",
            body={"messages": [], "a2ui": [envelope]},
            protocol_version=protocol_version,
            validate_responses=False,
        )

    assert turn.client_events == []
    assert "confirm" not in _turn_text(turn)
    assert "Skipping A2UI envelope" in caplog.text
    assert "exactly one" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol_version", ["v0.9", "v0.9.1"])
@pytest.mark.parametrize(
    "envelope",
    [
        pytest.param({"rendererFunctionResponse": {"functionCallId": "fc-7", "value": "Berlin"}}, id="response"),
        pytest.param(
            {
                "callAgentFunction": {
                    "surfaceId": "s1",
                    "functionCallId": "fc-1",
                    "callFunction": {"@call": "Berlin", "args": {}},
                },
            },
            id="call",
        ),
    ],
)
async def test_v1_0_only_envelope_is_dropped_before_v1_0(
    protocol_version: str, envelope: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    # v0.9's client_to_server.json has only 'action' and 'error'.
    with caplog.at_level("WARNING", logger="ag2.a2ui.incoming"):
        turn = await serve_turn(
            "Hello.",
            body={"messages": [], "a2ui": [{"version": protocol_version, **envelope}]},
            protocol_version=protocol_version,
            validate_responses=False,
        )

    assert turn.client_events == []
    assert "Berlin" not in _turn_text(turn)
    assert "Skipping A2UI envelope" in caplog.text
    assert "v1.0" in caplog.text
