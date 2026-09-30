# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Both AG-UI endpoints answer as the HTTP + SSE binding says.

A run that starts answers `200` with `Content-Type: text/event-stream`; input that
cannot be read is refused with `400` before any stream. The inputs below are the shapes
upstream's `RunAgentInput` conformance fixtures cover (`spec/1.0/fixtures/RunAgentInput/` in
`ag-ui-protocol/ag-ui`), written out here so that nothing has to be copied or kept in step.
"""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from dirty_equals import IsPartialDict

from ag2 import Agent
from ag2.a2ui import A2UIServer
from ag2.a2ui.transports import AgUiTransport
from ag2.ag_ui import NO_HELD_TURN, AGUIStream
from ag2.testing import TestConfig
from test.ag_ui.harness import decode
from test.ag_ui.serving import app_for, run_body

pytestmark = pytest.mark.asyncio

_RUN: dict[str, Any] = {"threadId": "t1", "runId": "r1", "messages": []}

_EVERY_ROLE: list[dict[str, Any]] = [
    {"id": "1", "role": "developer", "content": "be brief"},
    {"id": "2", "role": "system", "content": "you are an agent"},
    {"id": "3", "role": "user", "content": "hello"},
    {
        "id": "4",
        "role": "assistant",
        "content": "hi",
        "toolCalls": [{"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"q":"x"}'}}],
    },
    {"id": "5", "role": "tool", "content": "3 results", "toolCallId": "c1"},
    {"id": "6", "role": "activity", "activityType": "search", "content": {"hits": 3}},
    {"id": "7", "role": "reasoning", "content": "weighing options"},
]

_VALID_INPUTS: dict[str, dict[str, Any]] = {
    "minimal": _RUN,
    "every-role": {**_RUN, "messages": _EVERY_ROLE},
    "full": {
        **_RUN,
        "runId": "r2",
        "parentRunId": "r1",
        "messages": _EVERY_ROLE,
        "tools": [{"name": "search", "description": "Searches", "parameters": {}}],
        "context": [{"description": "locale", "value": "en-GB"}],
        "resume": [{"interruptId": "i1", "status": "resolved", "payload": True}],
    },
    **{
        f"state-{kind}": {**_RUN, "state": value}
        for kind, value in [("array", [1, 2]), ("number", 42), ("object", {"a": 1}), ("string", "text")]
    },
    **{
        f"forwarded-props-{kind}": {**_RUN, "forwardedProps": value}
        for kind, value in [("array", [1, 2]), ("number", 42), ("object", {"a": 1}), ("string", "text")]
    },
}

_INVALID_INPUTS: dict[str, dict[str, Any]] = {
    "messages-missing": {"threadId": "t1", "runId": "r1"},
    "messages-item-not-a-message": {**_RUN, "messages": [42]},
    "tools-item-not-a-tool": {**_RUN, "tools": [42]},
    "context-item-not-a-context": {**_RUN, "context": [42]},
    "resume-item-not-a-resume-entry": {**_RUN, "resume": [42]},
}


def _ag_ui_app() -> Any:
    return app_for(AGUIStream(Agent("test_agent", config=TestConfig("hello"))))


def _a2ui_app() -> Any:
    return A2UIServer(
        Agent("test_agent", config=TestConfig("hello")), transport=AgUiTransport(), validate_responses=False
    )


_APPS = pytest.mark.parametrize("make_app", [_ag_ui_app, _a2ui_app], ids=["AGUIStream", "A2UI"])


async def _post(app: Any, content: bytes) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ag-ui.test") as client:
        return await client.post(
            "/", content=content, headers={"content-type": "application/json", "accept": "text/event-stream"}
        )


@_APPS
async def test_a_run_that_starts_answers_200_as_an_event_stream(make_app: Callable[[], Any]) -> None:
    response = await _post(make_app(), json.dumps(run_body(thread_id="t1", run_id="r1")).encode())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


@_APPS
@pytest.mark.parametrize("body", _VALID_INPUTS.values(), ids=list(_VALID_INPUTS))
async def test_every_valid_input_is_served(make_app: Callable[[], Any], body: dict[str, Any]) -> None:
    response = await _post(make_app(), json.dumps(body).encode())

    assert response.status_code == 200
    [first, *_] = decode(response.text.splitlines())
    if body.get("resume"):
        # Accepted, then refused as a resume: a fresh server holds no interrupt.
        assert first == IsPartialDict({"type": "RUN_ERROR", "code": NO_HELD_TURN})
    else:
        assert first["type"] == "RUN_STARTED"


@_APPS
@pytest.mark.parametrize("body", _INVALID_INPUTS.values(), ids=list(_INVALID_INPUTS))
async def test_every_invalid_input_is_refused_before_any_stream(
    make_app: Callable[[], Any], body: dict[str, Any]
) -> None:
    response = await _post(make_app(), json.dumps(body).encode())

    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert "error" in response.json()


@_APPS
async def test_a_part_whose_source_is_of_an_unknown_kind_is_stripped_and_the_run_served(
    make_app: Callable[[], Any],
) -> None:
    body = run_body(thread_id="t1", run_id="r1")
    body["messages"] = [
        {
            "id": "m1",
            "role": "user",
            "content": [{"type": "text", "text": "look"}, {"type": "image", "source": {"type": "ipfs", "value": "Qm"}}],
        }
    ]

    response = await _post(make_app(), json.dumps(body).encode())

    assert response.status_code == 200
    assert decode(response.text.splitlines())[-1] == IsPartialDict({
        "type": "RUN_FINISHED",
        "outcome": {"type": "success"},
    })


@_APPS
async def test_a_body_that_is_not_json_is_refused(make_app: Callable[[], Any]) -> None:
    response = await _post(make_app(), b"{not json")

    assert response.status_code == 400
