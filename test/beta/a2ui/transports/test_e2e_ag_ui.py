# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end A2UI-over-AG-UI turns through ``A2UIServer(transport=AgUiTransport())``.

The agent's validated A2UI messages must reach the wire as a single AG-UI
``ActivitySnapshotEvent`` with ``activity_type="a2ui-surface"`` and the
operations under ``content["a2ui_operations"]`` — the exact contract
CopilotKit's ``@copilotkit/a2ui-renderer`` consumes. The LLM is mocked with
``TestConfig`` so the turn is deterministic. Turns are driven over a real
in-process HTTP transport (``httpx`` + ``ASGITransport``) since the server is
itself the ASGI app.
"""

import json
from typing import Any

import httpx
import pytest
from ag_ui.core import RunAgentInput
from dirty_equals import IsPartialDict

from autogen.beta import Agent
from autogen.beta.a2ui import A2UIServer, a2ui_action
from autogen.beta.a2ui.transports import AgUiTransport
from autogen.beta.events import ModelRequest, TextInput, ToolCallEvent
from autogen.beta.testing import TestConfig, TrackingConfig

_CATALOG = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"
_A2UI_RESPONSE = (
    "Here is your UI.\n<a2ui-json>\n"
    f'[{{"version": "v0.9", "createSurface": {{"surfaceId": "s1", "catalogId": "{_CATALOG}"}}}}]\n'
    "</a2ui-json>"
)


def _run_input(content: str) -> RunAgentInput:
    """A minimal AG-UI run with a single trailing user message."""
    return RunAgentInput.model_validate(
        {
            "thread_id": "t1",
            "run_id": "r1",
            "state": {},
            "messages": [{"id": "m1", "role": "user", "content": content}],
            "tools": [],
            "context": [],
            "forwarded_props": {},
        },
    )


def _click_input(name: str, context: dict[str, Any], *, messages: list[dict[str, Any]] | None = None) -> RunAgentInput:
    """An AG-UI run carrying a CopilotKit button click in ``forwardedProps.a2uiAction``."""
    return RunAgentInput.model_validate(
        {
            "thread_id": "t1",
            "run_id": "r2",
            "state": {},
            "messages": messages or [],
            "tools": [],
            "context": [],
            "forwarded_props": {
                "a2uiAction": {
                    "userAction": {"name": name, "surfaceId": "s1", "sourceComponentId": "btn", "context": context},
                },
            },
        },
    )


def _client(app: Any) -> httpx.AsyncClient:
    """An async HTTP client bound to ``app`` over an in-process ASGI transport."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://a2ui.test")


async def _dispatch_events(server: A2UIServer, incoming: RunAgentInput) -> list[dict[str, Any]]:
    """Drive a turn over HTTP and decode the SSE-encoded AG-UI events into dicts."""
    events: list[dict[str, Any]] = []
    async with _client(server) as client:
        resp = await client.post("/", json=incoming.model_dump(by_alias=True))
        assert resp.status_code == 200
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


@pytest.mark.asyncio
async def test_single_turn_emits_text_then_activity_snapshot() -> None:
    server = A2UIServer(Agent(name="ui", config=TestConfig(_A2UI_RESPONSE)), transport=AgUiTransport())

    events = await _dispatch_events(server, _run_input("show ui"))

    types = [e["type"] for e in events]
    assert types[0] == "RUN_STARTED"
    assert types[-1] == "RUN_FINISHED"
    assert "RUN_ERROR" not in types

    # Prose arrives stripped of the <a2ui-json> block (it comes from the final
    # message, not live model chunks), so the raw tag never leaks into the text.
    [text] = [e for e in events if e["type"] == "TEXT_MESSAGE_CHUNK"]
    assert text["delta"] == "Here is your UI."
    assert "<a2ui-json>" not in text["delta"]

    [activity] = [e for e in events if e["type"] == "ACTIVITY_SNAPSHOT"]
    assert activity["activityType"] == "a2ui-surface"
    assert activity["content"] == {
        "a2ui_operations": [IsPartialDict({"createSurface": {"surfaceId": "s1", "catalogId": _CATALOG}})],
    }


@pytest.mark.asyncio
async def test_plain_text_emits_no_activity_snapshot() -> None:
    server = A2UIServer(
        Agent(name="ui", config=TestConfig("Just text.")),
        transport=AgUiTransport(),
        validate_responses=False,
    )

    events = await _dispatch_events(server, _run_input("hi"))

    assert [e["type"] for e in events if e["type"] == "ACTIVITY_SNAPSHOT"] == []
    [text] = [e for e in events if e["type"] == "TEXT_MESSAGE_CHUNK"]
    assert text["delta"] == "Just text."


@pytest.mark.asyncio
async def test_server_serves_turn_over_http() -> None:
    server = A2UIServer(Agent(name="ui", config=TestConfig(_A2UI_RESPONSE)), transport=AgUiTransport())

    async with _client(server) as client:
        resp = await client.post("/", json=_run_input("show ui").model_dump(by_alias=True))

    assert resp.status_code == 200
    # The stream is SSE — the response must advertise the encoder's content type,
    # not Starlette's text/plain default.
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert '"activityType":"a2ui-surface"' in body
    assert '"a2ui_operations"' in body
    assert '"type":"RUN_FINISHED"' in body


@pytest.mark.asyncio
async def test_button_click_in_forwarded_props_executes_server_action() -> None:
    # CopilotKit relays a click as forwardedProps.a2uiAction and re-runs the
    # agent (no new chat message). The click is rewritten into the turn; the
    # mocked LLM then calls the @a2ui_action (injected for the turn) and answers
    # once it returns. The agent stays plain — the action lives on the server.
    clicked: list[str] = []

    @a2ui_action(description="Schedule all posts for the given time")
    def schedule_posts(time: str) -> str:
        clicked.append(time)
        return f"scheduled {time}"

    agent = Agent(
        name="ui",
        config=TestConfig(ToolCallEvent(name="schedule_posts", arguments='{"time": "2:00 PM"}'), "All set."),
    )
    server = A2UIServer(agent, actions=[schedule_posts], transport=AgUiTransport(), validate_responses=False)

    events = await _dispatch_events(server, _click_input("schedule_posts", {"time": "2:00 PM"}))

    assert clicked == ["2:00 PM"]
    [text] = [e for e in events if e["type"] == "TEXT_MESSAGE_CHUNK"]
    assert text["delta"] == "All set."


@pytest.mark.asyncio
async def test_click_is_rewritten_into_the_llm_turn() -> None:
    # The click must reach the LLM as the current turn — proving it was read
    # from forwardedProps (not from messages, which is empty here).
    @a2ui_action(description="Schedule all posts for the given time")
    def schedule_posts(time: str) -> str:
        return f"scheduled {time}"

    tracking = TrackingConfig(TestConfig("ok"))
    server = A2UIServer(
        Agent(name="ui", config=tracking),
        actions=[schedule_posts],
        transport=AgUiTransport(),
        validate_responses=False,
    )

    await _dispatch_events(server, _click_input("schedule_posts", {"time": "2:00 PM"}))

    sent = tracking.mock.call_args_list[0].args[0]
    assert isinstance(sent, ModelRequest)
    turn_text = " ".join(p.content for p in sent.parts if isinstance(p, TextInput))
    assert "schedule_posts" in turn_text
    assert "2:00 PM" in turn_text


@pytest.mark.asyncio
async def test_invalid_body_returns_400() -> None:
    server = A2UIServer(Agent(name="ui", config=TestConfig(_A2UI_RESPONSE)), transport=AgUiTransport())

    async with _client(server) as client:
        resp = await client.post("/", content=b"{not json")

    assert resp.status_code == 400
    assert "error" in resp.json()
