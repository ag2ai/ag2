# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

from starlette.testclient import TestClient

from autogen.beta.a2ui import A2UIAgent, A2UIEventAction
from autogen.beta.a2ui.rest import A2UIServer
from autogen.beta.testing import TestConfig

_CATALOG = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"
_A2UI_RESPONSE = (
    "Here is your UI.\n<a2ui-json>\n"
    f'[{{"version": "v0.9", "createSurface": {{"surfaceId": "s1", "catalogId": "{_CATALOG}"}}}}]\n'
    "</a2ui-json>"
)


def _agent(response: str = _A2UI_RESPONSE, *, validate: bool = True, actions=()) -> A2UIAgent:
    return A2UIAgent(name="t", config=TestConfig(response), validate_responses=validate, tools=actions)


class TestSSEApp:
    def test_streams_prose_and_message(self) -> None:
        app = A2UIServer(_agent()).build_sse_app()
        client = TestClient(app)

        resp = client.post("/a2ui", json={"messages": [{"role": "user", "content": "show ui"}]})

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
        assert "event: text" in body
        assert '"text": "Here is your UI."' in body
        assert '"createSurface"' in body
        assert "event: done" in body

    def test_plain_text_no_message_frame(self) -> None:
        app = A2UIServer(_agent("Just text.", validate=False)).build_sse_app()
        client = TestClient(app)

        resp = client.post("/a2ui", json={"messages": [{"role": "user", "content": "hi"}]})

        assert resp.status_code == 200
        assert '"text": "Just text."' in resp.text
        assert "createSurface" not in resp.text

    def test_malformed_body_returns_400(self) -> None:
        app = A2UIServer(_agent()).build_sse_app()
        client = TestClient(app)

        resp = client.post("/a2ui", content=b"{not json")

        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_custom_path(self) -> None:
        app = A2UIServer(_agent("hi", validate=False)).build_sse_app(path="/custom")
        client = TestClient(app)

        assert client.post("/custom", json={"messages": []}).status_code == 200
        assert client.post("/a2ui", json={"messages": []}).status_code == 404


class TestJSONLApp:
    def test_streams_ndjson_lines(self) -> None:
        app = A2UIServer(_agent()).build_jsonl_app()
        client = TestClient(app)

        resp = client.post("/a2ui", json={"messages": [{"role": "user", "content": "show ui"}]})

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        lines = [json.loads(line) for line in resp.text.splitlines() if line]
        assert lines[0] == {"text": "Here is your UI."}
        assert lines[1]["createSurface"]["surfaceId"] == "s1"

    def test_malformed_body_returns_400(self) -> None:
        app = A2UIServer(_agent()).build_jsonl_app()
        client = TestClient(app)

        resp = client.post("/a2ui", content=b"not json")

        assert resp.status_code == 400


class TestActionRoundTrip:
    def test_action_click_drives_a_turn(self) -> None:
        action = A2UIEventAction(name="confirm", description="Confirm the booking")
        app = A2UIServer(_agent("Confirmed.", validate=False, actions=[action])).build_sse_app()
        client = TestClient(app)

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
        assert '"text": "Confirmed."' in resp.text
