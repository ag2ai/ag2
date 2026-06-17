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

from autogen.beta.a2ui import A2UIAgent, a2ui_action
from autogen.beta.a2ui.rest import A2UIServer
from autogen.beta.events import ModelRequest, TextInput, ToolCallEvent
from autogen.beta.testing import TestConfig, TrackingConfig

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
        agent = A2UIAgent(name="ui", config=TestConfig(_A2UI_RESPONSE))
        app = A2UIServer(agent).build_jsonl_app()

        async with _client(app) as client:
            resp = await client.post("/a2ui", json={"messages": [{"role": "user", "content": "show ui"}]})

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        lines = [json.loads(line) for line in resp.text.splitlines() if line]
        assert lines[0] == {"text": "Here is your UI."}
        assert lines[1] == IsPartialDict({"createSurface": {"surfaceId": "s1", "catalogId": _CATALOG}})

    async def test_plain_text_emits_no_surface_frame(self) -> None:
        agent = A2UIAgent(name="ui", config=TestConfig("Just text."), validate_responses=False)
        app = A2UIServer(agent).build_jsonl_app()

        async with _client(app) as client:
            resp = await client.post("/a2ui", json={"messages": [{"role": "user", "content": "hi"}]})

        assert resp.status_code == 200
        lines = [json.loads(line) for line in resp.text.splitlines() if line]
        assert lines == [{"text": "Just text."}]

    async def test_malformed_body_returns_400(self) -> None:
        agent = A2UIAgent(name="ui", config=TestConfig(_A2UI_RESPONSE))
        app = A2UIServer(agent).build_jsonl_app()

        async with _client(app) as client:
            resp = await client.post("/a2ui", content=b"{not json")

        assert resp.status_code == 400
        assert "error" in resp.json()


@pytest.mark.asyncio
async def test_sse_single_turn_streams_text_message_done() -> None:
    agent = A2UIAgent(name="ui", config=TestConfig(_A2UI_RESPONSE))
    app = A2UIServer(agent).build_sse_app()

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
async def test_action_round_trip_client_click_executes_server_tool() -> None:
    clicked: list[str] = []

    @a2ui_action(description="Schedule all posts for the given time")
    def schedule_posts(time: str) -> str:
        clicked.append(time)
        return f"scheduled {time}"

    # The click envelope is rewritten into a prompt; the mocked LLM then
    # calls the registered tool, and answers with prose once it returns.
    agent = A2UIAgent(
        name="ui",
        validate_responses=False,
        tools=[schedule_posts],
        config=TestConfig(
            ToolCallEvent(name="schedule_posts", arguments='{"time": "2:00 PM"}'),
            "All set.",
        ),
    )
    app = A2UIServer(agent).build_jsonl_app()

    async with _client(app) as client:
        resp = await client.post(
            "/a2ui",
            json={
                "messages": [],
                "a2ui": [
                    {
                        "version": "v0.9",
                        "action": {
                            "name": "schedule_posts",
                            "surfaceId": "s1",
                            "sourceComponentId": "btn",
                            "timestamp": "2026-06-15T00:00:00Z",
                            "context": {"time": "2:00 PM"},
                        },
                    }
                ],
            },
        )

    assert resp.status_code == 200
    assert clicked == ["2:00 PM"]
    lines = [json.loads(line) for line in resp.text.splitlines() if line]
    assert lines == [{"text": "All set."}]


@pytest.mark.asyncio
async def test_stateless_client_resends_history_each_turn() -> None:
    # The server keeps no state: the client must resend the full
    # conversation, so the trailing user message is the current turn and
    # the prior assistant turn lands in history.
    tracking = TrackingConfig(TestConfig("ack"))
    agent = A2UIAgent(name="ui", config=tracking, validate_responses=False)
    app = A2UIServer(agent).build_jsonl_app()

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
