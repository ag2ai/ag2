# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Both AG-UI endpoints answer as the HTTP + SSE binding says.

A run that starts answers `200` with `Content-Type: text/event-stream`; input that
cannot be read is refused with `400` before any stream. The fixtures under
`test/ag_ui/fixtures/run_agent_input/` are copied verbatim from upstream
`ag-ui-protocol/ag-ui` at `024332cb`, `spec/1.0/fixtures/RunAgentInput/`.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from ag2 import Agent
from ag2.a2ui import A2UIServer
from ag2.a2ui.transports import AgUiTransport
from ag2.ag_ui import AGUIStream
from ag2.testing import TestConfig
from test.ag_ui.harness import decode
from test.ag_ui.serving import app_for, run_body

pytestmark = pytest.mark.asyncio

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "run_agent_input"


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
@pytest.mark.parametrize("fixture", sorted((_FIXTURES / "valid").glob("*.json")), ids=lambda p: p.stem)
async def test_every_valid_upstream_input_is_served(make_app: Callable[[], Any], fixture: Path) -> None:
    response = await _post(make_app(), fixture.read_bytes())

    assert response.status_code == 200
    assert decode(response.text.splitlines())[0]["type"] == "RUN_STARTED"


@_APPS
@pytest.mark.parametrize("fixture", sorted((_FIXTURES / "invalid").glob("*.json")), ids=lambda p: p.stem)
async def test_every_invalid_upstream_input_is_refused_before_any_stream(
    make_app: Callable[[], Any], fixture: Path
) -> None:
    response = await _post(make_app(), fixture.read_bytes())

    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert "error" in response.json()


@_APPS
async def test_a_body_that_is_not_json_is_refused(make_app: Callable[[], Any]) -> None:
    response = await _post(make_app(), b"{not json")

    assert response.status_code == 400
