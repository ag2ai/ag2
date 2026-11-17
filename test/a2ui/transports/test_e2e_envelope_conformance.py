# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Renderer→agent envelopes must conform to the served protocol's ``renderer_to_agent.json``.

A malformed envelope is dropped with a warning, except a malformed v1.0 ``callAgentFunction`` that carries a
``functionCallId``: the renderer is waiting on that id, so the server answers ``INVALID_FUNCTION_CALL``.
"""

import logging
from typing import Any

import pytest

from ag2.a2ui import a2ui_action

from ..seam import serve_turn

_ACTION = {
    "name": "lookup",
    "surfaceId": "s",
    "sourceComponentId": "b",
    "timestamp": "2026-01-01T00:00:00Z",
    "context": {},
}
_CALL = {"surfaceId": "s", "functionCallId": "fc-1", "callFunction": {"@call": "lookup", "args": {}}}


async def _serve(envelope: dict[str, Any], called: list[str]) -> Any:
    @a2ui_action
    def lookup() -> dict[str, Any]:
        called.append("lookup")
        return {"updateDataModel": {"surfaceId": "s", "path": "/v", "value": 1}}

    return await serve_turn(
        "Hi.",
        body={"messages": [{"role": "user", "content": "go"}], "a2ui": [envelope]},
        actions=[lookup],
        protocol_version="v1.0",
    )


@pytest.mark.asyncio
async def test_a_conforming_action_runs_its_handler() -> None:
    called: list[str] = []

    action = {**_ACTION, "metadata": {"extensions": {"client_audit": {"signature": "abc"}}}}

    turn = await _serve({"version": "1.0", "action": action}, called)

    assert called == ["lookup"]
    assert turn.ui == [{"version": "1.0", "updateDataModel": {"surfaceId": "s", "path": "/v", "value": 1}}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("envelope", "problem"),
    [
        ({"version": "v0.9", "action": _ACTION}, "envelope 'version' is 'v0.9'; this server speaks '1.0'"),
        ({"action": _ACTION}, "envelope 'version' is None; this server speaks '1.0'"),
        (
            {"version": "1.0", "action": _ACTION, "extra": 1},
            "envelope has properties besides 'version' and 'action': 'extra'",
        ),
        (
            {"version": "1.0", "action": {**_ACTION, "timestamp": "yesterday"}},
            "'action.timestamp' 'yesterday' is not an RFC 3339 date-time",
        ),
        (
            {"version": "1.0", "action": {key: value for key, value in _ACTION.items() if key != "context"}},
            "'action' needs a 'context' object",
        ),
        (
            {"version": "1.0", "action": _ACTION, "error": {"code": "X", "message": "m", "surfaceId": "s"}},
            "envelope must carry exactly one of",
        ),
        (
            {"version": "1.0", "action": {**_ACTION, "metadata": {"unknown": 1}}},
            "'action.metadata' has unknown properties 'unknown'",
        ),
        (
            {"version": "1.0", "action": {**_ACTION, "metadata": {"extensions": {"bad.key": True}}}},
            "'action.metadata.extensions' keys must be UAX #31 identifiers: 'bad.key'",
        ),
    ],
    ids=[
        "wrong-version",
        "no-version",
        "extra-key",
        "bad-timestamp",
        "no-context",
        "two-payloads",
        "unknown-metadata",
        "extension-key-not-an-identifier",
    ],
)
async def test_a_malformed_envelope_is_dropped(
    envelope: dict[str, Any], problem: str, caplog: pytest.LogCaptureFixture
) -> None:
    called: list[str] = []

    with caplog.at_level(logging.WARNING, logger="ag2.a2ui"):
        turn = await _serve(envelope, called)

    assert called == []
    assert turn.frames == [{"text": "Hi."}]
    assert problem in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "reason"),
    [
        (
            {key: value for key, value in _CALL.items() if key != "surfaceId"},
            "'callAgentFunction' needs a non-empty string 'surfaceId'",
        ),
        ({**_CALL, "extra": 1}, "'callAgentFunction' has unknown properties 'extra'"),
        (
            {**_CALL, "callFunction": {**_CALL["callFunction"], "returnType": "string"}},
            "'callAgentFunction.callFunction' has unknown properties 'returnType'",
        ),
        ({**_CALL, "callFunction": "lookup"}, "'callAgentFunction' needs a 'callFunction' object"),
    ],
    ids=["no-surface-id", "extra-key", "extra-key-in-call", "call-not-an-object"],
)
async def test_a_malformed_function_call_with_an_id_is_answered_invalid(call: dict[str, Any], reason: str) -> None:
    called: list[str] = []

    turn = await _serve({"version": "1.0", "callAgentFunction": call}, called)

    assert called == []
    assert turn.ui == [
        {
            "version": "1.0",
            "agentFunctionResponse": {
                "functionCallId": "fc-1",
                "error": {"code": "INVALID_FUNCTION_CALL", "message": f"Malformed callAgentFunction: {reason}."},
            },
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "problem"),
    [
        (
            {key: value for key, value in _CALL.items() if key != "functionCallId"},
            "'callAgentFunction' needs a string 'functionCallId'",
        ),
        (
            {"functionCallId": "", "callFunction": _CALL["callFunction"]},
            "'callAgentFunction' needs a non-empty string 'surfaceId'",
        ),
    ],
    ids=["no-id", "empty-id"],
)
async def test_a_malformed_function_call_without_an_id_to_answer_is_dropped(
    call: dict[str, Any], problem: str, caplog: pytest.LogCaptureFixture
) -> None:
    called: list[str] = []

    with caplog.at_level(logging.WARNING, logger="ag2.a2ui"):
        turn = await _serve({"version": "1.0", "callAgentFunction": call}, called)

    assert called == []
    assert turn.frames == [{"text": "Hi."}]
    assert problem in caplog.text
