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
from typing import Annotated, Any

import httpx
import pytest
from dirty_equals import IsPartialDict

pytest.importorskip("ag_ui")
from ag_ui.core import RunAgentInput

from ag2 import Agent, Depends
from ag2.a2ui import A2UIServer, a2ui_action
from ag2.a2ui.transports import AgUiTransport
from ag2.events import ModelRequest, TextInput
from ag2.testing import TestConfig, TrackingConfig

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
    # CopilotKit relays a click as forwardedProps.a2uiAction. The click runs the
    # registered @a2ui_action on the server (the agent is NOT invoked); its
    # returned surface update is emitted as an ACTIVITY_SNAPSHOT.
    clicked: list[str] = []

    @a2ui_action(description="Add the item to the cart")
    def add_to_basket(good_id: str) -> dict:
        clicked.append(good_id)
        return {"updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 1}}

    agent = Agent(name="ui", config=TestConfig("AGENT SHOULD NOT RUN"))
    server = A2UIServer(agent, actions=[add_to_basket], transport=AgUiTransport(), validate_responses=False)

    events = await _dispatch_events(server, _click_input("add_to_basket", {"good_id": "G1"}))

    assert clicked == ["G1"]
    # Agent skipped (pure server-action turn): no prose, one surface update.
    assert [e for e in events if e["type"] == "TEXT_MESSAGE_CHUNK"] == []
    [activity] = [e for e in events if e["type"] == "ACTIVITY_SNAPSHOT"]
    assert activity["activityType"] == "a2ui-surface"
    assert activity["content"] == {
        "a2ui_operations": [IsPartialDict({"updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 1}})],
    }


@pytest.mark.asyncio
async def test_button_click_resolves_injected_dependency() -> None:
    # DI is not transport-specific: the AG-UI transport shares the same turn
    # core as REST, so a clicked server action resolves its Depends(...) against
    # the agent's dependency_provider (here swapped via an override — the
    # test-DB pattern) before running, with the agent never invoked.
    class CartStore:
        def __init__(self) -> None:
            self.items: list[str] = []

    def get_store() -> CartStore:
        raise RuntimeError("the real store is unavailable under test")

    stub = CartStore()

    @a2ui_action(description="Add the item to the cart")
    def add_to_basket(good_id: str, store: Annotated[CartStore, Depends(get_store)]) -> dict:
        store.items.append(good_id)
        return {"updateDataModel": {"surfaceId": "cart", "path": "/count", "value": len(store.items)}}

    agent = Agent(name="ui", config=TestConfig("AGENT SHOULD NOT RUN"))
    agent.dependency_provider.override(get_store, lambda: stub)
    server = A2UIServer(agent, actions=[add_to_basket], transport=AgUiTransport(), validate_responses=False)

    events = await _dispatch_events(server, _click_input("add_to_basket", {"good_id": "G7"}))

    assert stub.items == ["G7"]
    [activity] = [e for e in events if e["type"] == "ACTIVITY_SNAPSHOT"]
    assert activity["content"] == {
        "a2ui_operations": [IsPartialDict({"updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 1}})],
    }


@pytest.mark.asyncio
async def test_unregistered_click_is_rewritten_into_the_llm_turn() -> None:
    # A click on a button with no registered action must reach the LLM as the
    # current turn — proving it was read from forwardedProps (messages is empty).
    tracking = TrackingConfig(TestConfig("ok"))
    server = A2UIServer(
        Agent(name="ui", config=tracking),
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
