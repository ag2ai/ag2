# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Annotated, Any

import httpx
import pytest
from pydantic import BaseModel

from ag2 import Agent, Context, Depends, Inject, Variable
from ag2.a2ui import A2UIServer, a2ui_action
from ag2.a2ui.transports import RestTransport
from ag2.testing import TestConfig


class Database:
    """A stand-in for an injected backend resource (e.g. a DB connection)."""

    def __init__(self) -> None:
        self.added: list[str] = []

    def add(self, good_id: str) -> int:
        self.added.append(good_id)
        return len(self.added)


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://a2ui.test")


def _click_body(name: str, context: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [],
        "a2ui": [
            {
                "version": "v0.9",
                "action": {
                    "name": name,
                    "surfaceId": "s1",
                    "sourceComponentId": "btn",
                    "timestamp": "2026-06-15T00:00:00Z",
                    "context": context,
                },
            }
        ],
    }


async def _post_click(app: Any, name: str, context: dict[str, Any]) -> httpx.Response:
    async with _client(app) as client:
        return await client.post("/a2ui", json=_click_body(name, context))


def _server(agent: Agent, action: Any) -> A2UIServer:
    return A2UIServer(
        agent,
        actions=[action],
        transport=RestTransport(encoding="jsonl"),
        validate_responses=False,
    )


@pytest.mark.asyncio
async def test_depends_injects_into_server_action() -> None:
    db = Database()

    def get_db() -> Database:
        return db

    @a2ui_action(description="Add the item to the cart")
    async def add_to_cart(good_id: str, database: Annotated[Database, Depends(get_db)]) -> dict:
        count = database.add(good_id)
        return {"updateDataModel": {"surfaceId": "cart", "path": "/count", "value": count}}

    app = _server(Agent(name="ui", config=TestConfig("AGENT SHOULD NOT RUN")), add_to_cart)

    resp = await _post_click(app, "add_to_cart", {"good_id": "G1"})

    assert resp.status_code == 200
    assert db.added == ["G1"]
    lines = [json.loads(line) for line in resp.text.splitlines() if line]
    assert lines == [{"version": "v0.9", "updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 1}}]


@pytest.mark.asyncio
async def test_dependency_override_swaps_implementation() -> None:
    def get_db() -> Database:
        raise RuntimeError("the real database is unavailable under test")

    stub = Database()

    def get_stub_db() -> Database:
        return stub

    @a2ui_action
    async def add_to_cart(good_id: str, database: Annotated[Database, Depends(get_db)]) -> dict:
        database.add(good_id)
        return {"ok": True}

    agent = Agent(name="ui", config=TestConfig("AGENT SHOULD NOT RUN"))
    agent.dependency_provider.override(get_db, get_stub_db)
    app = _server(agent, add_to_cart)

    resp = await _post_click(app, "add_to_cart", {"good_id": "G2"})

    assert resp.status_code == 200
    assert stub.added == ["G2"]


@pytest.mark.asyncio
async def test_inject_resolves_agent_dependency() -> None:
    store = Database()

    @a2ui_action
    async def add_to_cart(good_id: str, database: Annotated[Database, Inject("db")]) -> dict:
        database.add(good_id)
        return {"ok": True}

    agent = Agent(name="ui", config=TestConfig("AGENT SHOULD NOT RUN"), dependencies={"db": store})
    app = _server(agent, add_to_cart)

    resp = await _post_click(app, "add_to_cart", {"good_id": "G3"})

    assert resp.status_code == 200
    assert store.added == ["G3"]


@pytest.mark.asyncio
async def test_serializer_coerces_context_values() -> None:
    captured: list[int] = []

    @a2ui_action
    async def set_qty(qty: int) -> dict:
        captured.append(qty)
        return {"ok": True}

    app = _server(Agent(name="ui", config=TestConfig("AGENT SHOULD NOT RUN")), set_qty)

    # The client sends the number as a JSON string; the PydanticSerializer
    # coerces it to int before the handler runs.
    resp = await _post_click(app, "set_qty", {"qty": "3"})

    assert resp.status_code == 200
    assert captured == [3]
    assert type(captured[0]) is int


@pytest.mark.asyncio
async def test_sync_handler_runs_with_injected_dependency() -> None:
    # A plain (sync) @a2ui_action is run in a worker thread; DI still resolves.
    db = Database()

    def get_db() -> Database:
        return db

    @a2ui_action
    def add_to_cart(good_id: str, database: Annotated[Database, Depends(get_db)]) -> dict:
        count = database.add(good_id)
        return {"updateDataModel": {"surfaceId": "cart", "path": "/count", "value": count}}

    app = _server(Agent(name="ui", config=TestConfig("AGENT SHOULD NOT RUN")), add_to_cart)

    resp = await _post_click(app, "add_to_cart", {"good_id": "G5"})

    assert resp.status_code == 200
    assert db.added == ["G5"]
    lines = [json.loads(line) for line in resp.text.splitlines() if line]
    assert lines == [{"version": "v0.9", "updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 1}}]


@pytest.mark.asyncio
async def test_variable_injects_into_server_action() -> None:
    # A context Variable resolves into a server action exactly like a tool's:
    # the action's context carries the agent's variables merged with the turn's.
    captured: list[int] = []

    @a2ui_action
    async def show_count(count: Annotated[int, Variable("count")]) -> dict:
        captured.append(count)
        return {"ok": True}

    agent = Agent(name="ui", config=TestConfig("AGENT SHOULD NOT RUN"), variables={"count": 5})
    app = _server(agent, show_count)

    resp = await _post_click(app, "show_count", {})

    assert resp.status_code == 200
    assert captured == [5]


@pytest.mark.asyncio
async def test_context_injected_into_server_action() -> None:
    # `ctx: Context` injects the live ConversationContext, carrying the merged
    # variables and the agent's dependencies — the same surface a tool sees.
    seen: list[dict[str, Any]] = []

    @a2ui_action
    async def inspect_ctx(ctx: Context) -> dict:
        seen.append({"vars": dict(ctx.variables), "deps": dict(ctx.dependencies)})
        return {"ok": True}

    store = Database()
    agent = Agent(
        name="ui",
        config=TestConfig("AGENT SHOULD NOT RUN"),
        variables={"count": 7},
        dependencies={"db": store},
    )
    app = _server(agent, inspect_ctx)

    resp = await _post_click(app, "inspect_ctx", {})

    assert resp.status_code == 200
    assert seen == [{"vars": {"count": 7}, "deps": {"db": store}}]


@pytest.mark.asyncio
async def test_handler_result_is_serialized_for_wire() -> None:
    # A handler may return a message whose value is a non-JSON-native object
    # (here a datetime). It must be coerced through the serializer before the
    # wire, mirroring how a tool result is encoded — not crash json.dumps.
    when = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)

    @a2ui_action
    async def stamp() -> dict:
        return {"updateDataModel": {"surfaceId": "s1", "path": "/at", "value": when}}

    app = _server(Agent(name="ui", config=TestConfig("AGENT SHOULD NOT RUN")), stamp)

    resp = await _post_click(app, "stamp", {})

    assert resp.status_code == 200
    lines = [json.loads(line) for line in resp.text.splitlines() if line]
    assert lines == [
        {
            "version": "v0.9",
            "updateDataModel": {"surfaceId": "s1", "path": "/at", "value": "2026-06-23T12:00:00Z"},
        }
    ]


@pytest.mark.asyncio
async def test_action_response_value_is_serialized() -> None:
    # When the client asks for a correlated reply (v1.0 wantResponse), a handler
    # returning a Pydantic model must be coerced to JSON for the actionResponse
    # value — not passed raw to json.dumps.
    class Receipt(BaseModel):
        order_id: str
        total: float

    @a2ui_action
    async def checkout() -> Receipt:
        return Receipt(order_id="O1", total=9.5)

    agent = Agent(name="ui", config=TestConfig("AGENT SHOULD NOT RUN"))
    app = A2UIServer(
        agent,
        actions=[checkout],
        transport=RestTransport(encoding="jsonl"),
        validate_responses=False,
        protocol_version="v1.0",
    )

    body = {
        "messages": [],
        "a2ui": [
            {
                "version": "v1.0",
                "action": {
                    "name": "checkout",
                    "surfaceId": "s1",
                    "sourceComponentId": "btn",
                    "timestamp": "2026-06-15T00:00:00Z",
                    "context": {},
                    "wantResponse": True,
                    "actionId": "a1",
                },
            }
        ],
    }
    async with _client(app) as client:
        resp = await client.post("/a2ui", json=body)

    assert resp.status_code == 200
    lines = [json.loads(line) for line in resp.text.splitlines() if line]
    assert lines == [
        {
            "version": "v1.0",
            "actionId": "a1",
            "actionResponse": {"value": {"order_id": "O1", "total": 9.5}},
        }
    ]


@pytest.mark.asyncio
async def test_generator_dependency_is_set_up_and_torn_down() -> None:
    events: list[str] = []

    def get_session() -> Iterator[Database]:
        events.append("open")
        session = Database()
        try:
            yield session
        finally:
            events.append("close")

    @a2ui_action
    async def add_to_cart(good_id: str, session: Annotated[Database, Depends(get_session)]) -> dict:
        session.add(good_id)
        events.append(f"used:{session.added}")
        return {"ok": True}

    app = _server(Agent(name="ui", config=TestConfig("AGENT SHOULD NOT RUN")), add_to_cart)

    resp = await _post_click(app, "add_to_cart", {"good_id": "G4"})

    assert resp.status_code == 200
    assert events == ["open", "used:['G4']", "close"]
