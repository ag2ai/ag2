# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end A2UI REST/SSE round-trips over a real HTTP transport.

Unlike ``test_server.py`` (which drives the ASGI app through Starlette's
synchronous ``TestClient``), these tests POST through ``httpx`` over an
``ASGITransport`` — the same in-memory-but-real-HTTP path the A2A E2E suite
uses — exercising body reading, streaming responses, and status codes end to
end. The LLM is mocked with ``TestConfig`` so the turn is deterministic; the
server is stateless, so each request rebuilds the client from the agent config.
"""

import json
from typing import Any

import httpx
import pytest
from dirty_equals import IsPartialDict

from ag2 import Agent
from ag2.a2ui import A2UIServer, a2ui_action
from ag2.a2ui.transports import RestTransport
from ag2.events import ModelRequest, TextInput
from ag2.testing import TestConfig, TrackingConfig

_CATALOG = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"
_A2UI_RESPONSE = (
    "Here is your UI.\n<a2ui-json>\n"
    f'[{{"version": "v0.9", "createSurface": {{"surfaceId": "s1", "catalogId": "{_CATALOG}"}}}}]\n'
    "</a2ui-json>"
)


def _client(app: Any) -> httpx.AsyncClient:
    """An async HTTP client bound to ``app`` over an in-process ASGI transport."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://a2ui.test")


@pytest.mark.asyncio
class TestE2EJsonl:
    async def test_single_turn_streams_prose_then_surface(self) -> None:
        agent = Agent(name="ui", config=TestConfig(_A2UI_RESPONSE))
        app = A2UIServer(agent, transport=RestTransport(encoding="jsonl"))

        async with _client(app) as client:
            resp = await client.post("/a2ui", json={"messages": [{"role": "user", "content": "show ui"}]})

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        lines = [json.loads(line) for line in resp.text.splitlines() if line]
        assert lines[0] == {"text": "Here is your UI."}
        assert lines[1] == IsPartialDict({"createSurface": {"surfaceId": "s1", "catalogId": _CATALOG}})

    async def test_plain_text_emits_no_surface_frame(self) -> None:
        agent = Agent(name="ui", config=TestConfig("Just text."))
        app = A2UIServer(agent, transport=RestTransport(encoding="jsonl"), validate_responses=False)

        async with _client(app) as client:
            resp = await client.post("/a2ui", json={"messages": [{"role": "user", "content": "hi"}]})

        assert resp.status_code == 200
        lines = [json.loads(line) for line in resp.text.splitlines() if line]
        assert lines == [{"text": "Just text."}]

    async def test_malformed_body_returns_400(self) -> None:
        agent = Agent(name="ui", config=TestConfig(_A2UI_RESPONSE))
        app = A2UIServer(agent, transport=RestTransport(encoding="jsonl"))

        async with _client(app) as client:
            resp = await client.post("/a2ui", content=b"{not json")

        assert resp.status_code == 400
        assert "error" in resp.json()


@pytest.mark.asyncio
async def test_sse_single_turn_streams_text_message_done() -> None:
    agent = Agent(name="ui", config=TestConfig(_A2UI_RESPONSE))
    app = A2UIServer(agent, transport=RestTransport(encoding="sse"))

    async with _client(app) as client:
        resp = await client.post("/a2ui", json={"messages": [{"role": "user", "content": "show ui"}]})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "event: text" in body
    assert '"text": "Here is your UI."' in body
    assert '"createSurface"' in body
    assert "event: done" in body


@pytest.mark.asyncio
async def test_action_round_trip_client_click_executes_server_action() -> None:
    clicked: list[str] = []

    @a2ui_action(description="Add the item to the cart")
    def add_to_basket(good_id: str) -> dict:
        clicked.append(good_id)
        return {"updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 1}}

    # The click runs the registered action on the server; the agent is NOT
    # invoked, and the handler's surface update is streamed back.
    agent = Agent(name="ui", config=TestConfig("AGENT SHOULD NOT RUN"))
    app = A2UIServer(
        agent, actions=[add_to_basket], transport=RestTransport(encoding="jsonl"), validate_responses=False
    )

    async with _client(app) as client:
        resp = await client.post(
            "/a2ui",
            json={
                "messages": [],
                "a2ui": [
                    {
                        "version": "v0.9",
                        "action": {
                            "name": "add_to_basket",
                            "surfaceId": "s1",
                            "sourceComponentId": "btn",
                            "timestamp": "2026-06-15T00:00:00Z",
                            "context": {"good_id": "G1"},
                        },
                    }
                ],
            },
        )

    assert resp.status_code == 200
    assert clicked == ["G1"]
    lines = [json.loads(line) for line in resp.text.splitlines() if line]
    assert lines == [{"version": "v0.9", "updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 1}}]


@pytest.mark.asyncio
async def test_stateless_client_resends_history_each_turn() -> None:
    # The server keeps no state: the client must resend the full
    # conversation, so the trailing user message is the current turn and
    # the prior assistant turn lands in history.
    tracking = TrackingConfig(TestConfig("ack"))
    agent = Agent(name="ui", config=tracking)
    app = A2UIServer(agent, transport=RestTransport(encoding="jsonl"), validate_responses=False)

    async with _client(app) as client:
        resp = await client.post(
            "/a2ui",
            json={
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "ack"},
                    {"role": "user", "content": "second"},
                ]
            },
        )

    assert resp.status_code == 200
    assert [json.loads(line) for line in resp.text.splitlines() if line] == [{"text": "ack"}]
    tracking.mock.assert_called_with(ModelRequest([TextInput("second")]))
