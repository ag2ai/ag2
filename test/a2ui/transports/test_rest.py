# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

from dirty_equals import IsPartialDict
from starlette.testclient import TestClient

from ag2 import Agent
from ag2.a2ui import A2UIServer, a2ui_action
from ag2.a2ui.transports import RestTransport
from ag2.testing import TestConfig

_CATALOG = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"
_A2UI_RESPONSE = (
    "Here is your UI.\n<a2ui-json>\n"
    f'[{{"version": "v0.9", "createSurface": {{"surfaceId": "s1", "catalogId": "{_CATALOG}"}}}}]\n'
    "</a2ui-json>"
)


def _server(
    response: str = _A2UI_RESPONSE,
    *,
    validate: bool = True,
    actions=(),
    transport: RestTransport | None = None,
) -> A2UIServer:
    agent = Agent(name="t", config=TestConfig(response))
    return A2UIServer(
        agent,
        transport=transport or RestTransport(encoding="sse"),
        actions=list(actions),
        validate_responses=validate,
    )


class TestSSEApp:
    def test_streams_prose_and_message(self) -> None:
        client = TestClient(_server())

        resp = client.post("/a2ui", json={"messages": [{"role": "user", "content": "show ui"}]})

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
        assert "event: text" in body
        assert '"text": "Here is your UI."' in body
        assert '"createSurface"' in body
        assert "event: done" in body

    def test_plain_text_no_message_frame(self) -> None:
        client = TestClient(_server("Just text.", validate=False))

        resp = client.post("/a2ui", json={"messages": [{"role": "user", "content": "hi"}]})

        assert resp.status_code == 200
        assert '"text": "Just text."' in resp.text
        assert "createSurface" not in resp.text

    def test_malformed_body_returns_400(self) -> None:
        client = TestClient(_server())

        resp = client.post("/a2ui", content=b"{not json")

        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_custom_path(self) -> None:
        client = TestClient(_server("hi", validate=False, transport=RestTransport(path="/custom")))

        assert client.post("/custom", json={"messages": []}).status_code == 200
        assert client.post("/a2ui", json={"messages": []}).status_code == 404


class TestJSONLApp:
    def test_streams_ndjson_lines(self) -> None:
        client = TestClient(_server(transport=RestTransport(encoding="jsonl")))

        resp = client.post("/a2ui", json={"messages": [{"role": "user", "content": "show ui"}]})

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        lines = [json.loads(line) for line in resp.text.splitlines() if line]
        assert lines[0] == {"text": "Here is your UI."}
        assert lines[1] == IsPartialDict({"createSurface": IsPartialDict({"surfaceId": "s1"})})

    def test_malformed_body_returns_400(self) -> None:
        client = TestClient(_server(transport=RestTransport(encoding="jsonl")))

        resp = client.post("/a2ui", content=b"not json")

        assert resp.status_code == 400


def test_action_click_runs_server_action() -> None:
    # A click on a registered action runs its handler on the server (the agent is
    # not invoked); the handler's surface update is streamed back as the only
    # frame, with no agent prose.
    @a2ui_action(description="Confirm the booking")
    def confirm() -> dict:
        return {"updateDataModel": {"surfaceId": "s1", "path": "/confirmed", "value": True}}

    client = TestClient(_server("AGENT SHOULD NOT RUN", validate=False, actions=[confirm]))

    resp = client.post(
        "/a2ui",
        json={
            "messages": [],
            "a2ui": [
                {
                    "version": "v0.9",
                    "action": {
                        "name": "confirm",
                        "surfaceId": "s1",
                        "sourceComponentId": "btn",
                        "timestamp": "2026-06-15T00:00:00Z",
                        "context": {},
                    },
                }
            ],
        },
    )

    assert resp.status_code == 200
    assert "AGENT SHOULD NOT RUN" not in resp.text
    assert '"updateDataModel"' in resp.text
    assert "/confirmed" in resp.text
