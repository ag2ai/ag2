# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Phase 3a HTTP surface — the minimum 7-endpoint slice.

Design §9.1 lists 17 HTTP endpoints total. Phase 3a ships the 7 that
an HTTP-only client needs to join a hub, discover actors, create
sessions, and read WAL state; the rest (activity, force-close, admin,
metrics, knowledge-read-through) slip to Phase 3b.

Two test strategies:

1. **ASGI transport.** Fast, in-process, no real sockets. Uses
   ``httpx.ASGITransport`` to dispatch requests into the Starlette
   app directly. Covers every happy path and most error paths.
2. **Real uvicorn integration.** A single end-to-end test that
   spins ``HttpServer.serve()`` on a random port and hits it with
   a real ``httpx.AsyncClient``. Proves the transport wiring end
   to end and guards against "works in ASGITransport, breaks over
   a real socket" regressions.

Every route's happy + at least one failure path is covered.
"""

from __future__ import annotations

import asyncio

import pytest

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    Hub,
    HubClient,
    HttpServer,
    LocalLink,
    Rule,
    SessionType,
    build_app,
)


httpx = pytest.importorskip("httpx")
pytest.importorskip("starlette")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Echo:
    def __init__(self, name: str) -> None:
        self.name = name

    async def ask(self, content: str, **_kwargs):
        class _R:
            def __init__(self, body: str) -> None:
                self.body = body

        return _R(f"echo:{content}")


async def _asgi_client(hub: Hub) -> httpx.AsyncClient:
    """Build an in-process HTTP client bound to the hub's Starlette app."""

    transport = httpx.ASGITransport(app=build_app(hub))
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


async def _register_two_actors(
    hub: Hub,
) -> tuple["ActorClient", "ActorClient", HubClient, LocalLink]:
    """Register Alice and Bob via the in-process Link (for cross-checks)."""

    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)
    alice = await hc.register(
        _Echo("alice"), identity=ActorIdentity(name="alice")
    )
    bob = await hc.register(
        _Echo("bob"), identity=ActorIdentity(name="bob")
    )
    return alice, bob, hc, link


# ---------------------------------------------------------------------------
# POST /v1/actors — register
# ---------------------------------------------------------------------------


class TestRegisterActor:
    @pytest.mark.asyncio
    async def test_register_creates_identity_and_returns_actor_id(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            response = await client.post(
                "/v1/actors",
                json={
                    "identity": {
                        "name": "alice",
                        "capabilities": ["research"],
                        "summary": "Researcher",
                    }
                },
            )
            assert response.status_code == 201
            body = response.json()
            assert body["identity"]["name"] == "alice"
            assert body["identity"]["capabilities"] == ["research"]
            assert body["actor_id"]
            assert body["actor_id"] == body["identity"]["actor_id"]

            # The hub should have the identity in its in-memory index.
            assert body["actor_id"] in hub._identities

    @pytest.mark.asyncio
    async def test_register_accepts_explicit_rule(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        rule = Rule()
        rule.limits.max_concurrent_sessions = 7
        async with await _asgi_client(hub) as client:
            response = await client.post(
                "/v1/actors",
                json={
                    "identity": {"name": "bob"},
                    "rule": rule.to_dict(),
                },
            )
            assert response.status_code == 201
            actor_id = response.json()["actor_id"]
            assert hub._rules[actor_id].limits.max_concurrent_sessions == 7

    @pytest.mark.asyncio
    async def test_register_duplicate_name_returns_409(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            first = await client.post(
                "/v1/actors", json={"identity": {"name": "alice"}}
            )
            assert first.status_code == 201
            second = await client.post(
                "/v1/actors", json={"identity": {"name": "alice"}}
            )
            assert second.status_code == 409
            assert second.json()["error"] == "DuplicateRegistrationError"

    @pytest.mark.asyncio
    async def test_register_missing_identity_returns_400(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            response = await client.post("/v1/actors", json={})
            assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_register_invalid_json_returns_400(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            response = await client.post(
                "/v1/actors",
                content="not json",
                headers={"content-type": "application/json"},
            )
            assert response.status_code == 400


# ---------------------------------------------------------------------------
# GET /v1/actors — discover
# ---------------------------------------------------------------------------


class TestFindActors:
    @pytest.mark.asyncio
    async def test_find_lists_all_actors(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            await client.post("/v1/actors", json={"identity": {"name": "alice"}})
            await client.post("/v1/actors", json={"identity": {"name": "bob"}})
            response = await client.get("/v1/actors")
            assert response.status_code == 200
            actors = response.json()["actors"]
            names = sorted(a["name"] for a in actors)
            assert names == ["alice", "bob"]

    @pytest.mark.asyncio
    async def test_find_filters_by_capability(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            await client.post(
                "/v1/actors",
                json={
                    "identity": {
                        "name": "researcher",
                        "capabilities": ["research", "summarization"],
                    }
                },
            )
            await client.post(
                "/v1/actors",
                json={
                    "identity": {
                        "name": "coder",
                        "capabilities": ["coding"],
                    }
                },
            )
            response = await client.get("/v1/actors?capability=research")
            matched = response.json()["actors"]
            assert len(matched) == 1
            assert matched[0]["name"] == "researcher"

    @pytest.mark.asyncio
    async def test_find_on_empty_hub_returns_empty_list(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            response = await client.get("/v1/actors")
            assert response.status_code == 200
            assert response.json() == {"actors": []}


# ---------------------------------------------------------------------------
# GET /v1/actors/{id} — describe
# ---------------------------------------------------------------------------


class TestDescribeActor:
    @pytest.mark.asyncio
    async def test_describe_by_name(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            await client.post(
                "/v1/actors",
                json={
                    "identity": {
                        "name": "alice",
                        "capabilities": ["research"],
                        "summary": "Researcher",
                    }
                },
            )
            response = await client.get("/v1/actors/alice")
            assert response.status_code == 200
            identity = response.json()["identity"]
            assert identity["name"] == "alice"
            assert identity["summary"] == "Researcher"

    @pytest.mark.asyncio
    async def test_describe_by_id(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            reg = await client.post(
                "/v1/actors", json={"identity": {"name": "alice"}}
            )
            actor_id = reg.json()["actor_id"]
            response = await client.get(f"/v1/actors/{actor_id}")
            assert response.status_code == 200
            assert response.json()["identity"]["actor_id"] == actor_id

    @pytest.mark.asyncio
    async def test_describe_unknown_returns_404(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            response = await client.get("/v1/actors/unknown-person")
            assert response.status_code == 404
            assert response.json()["error"] == "UnknownActorError"


# ---------------------------------------------------------------------------
# POST /v1/sessions — create
# ---------------------------------------------------------------------------


class TestCreateSession:
    @pytest.mark.asyncio
    async def test_create_consulting_via_http(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two_actors(hub)
        try:
            async with await _asgi_client(hub) as client:
                response = await client.post(
                    "/v1/sessions",
                    json={
                        "creator_id": alice.actor_id,
                        "session_type": "consulting",
                        "participants": ["bob"],
                    },
                )
                assert response.status_code == 201
                metadata = response.json()["metadata"]
                assert metadata["type"] == "consulting"
                assert metadata["state"] == "active"
                assert metadata["creator_id"] == alice.actor_id
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_create_session_unknown_creator_returns_404(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two_actors(hub)
        try:
            async with await _asgi_client(hub) as client:
                response = await client.post(
                    "/v1/sessions",
                    json={
                        "creator_id": "ghost",
                        "session_type": "consulting",
                        "participants": ["bob"],
                    },
                )
                assert response.status_code == 404
                assert response.json()["error"] == "UnknownActorError"
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_create_session_missing_fields_returns_400(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            response = await client.post(
                "/v1/sessions",
                json={"creator_id": "who", "session_type": "consulting"},
            )
            assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_create_session_unknown_type_returns_400(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two_actors(hub)
        try:
            async with await _asgi_client(hub) as client:
                response = await client.post(
                    "/v1/sessions",
                    json={
                        "creator_id": alice.actor_id,
                        "session_type": "tournament-of-champions",
                        "participants": ["bob"],
                    },
                )
                # The hub raises SessionTypeError → 400.
                assert response.status_code == 400
                assert response.json()["error"] == "SessionTypeError"
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# GET /v1/sessions/{id} + close + wal
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    @pytest.mark.asyncio
    async def test_describe_returns_session_metadata(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two_actors(hub)
        try:
            session = await alice.open(SessionType.CONVERSATION, target="bob")
            async with await _asgi_client(hub) as client:
                response = await client.get(f"/v1/sessions/{session.session_id}")
                assert response.status_code == 200
                metadata = response.json()["metadata"]
                assert metadata["session_id"] == session.session_id
                assert metadata["type"] == "conversation"
                assert metadata["state"] == "active"
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_describe_unknown_session_returns_404(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            response = await client.get("/v1/sessions/ghost")
            assert response.status_code == 404
            assert response.json()["error"] == "UnknownSessionError"

    @pytest.mark.asyncio
    async def test_close_session_transitions_state(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two_actors(hub)
        try:
            session = await alice.open(SessionType.CONVERSATION, target="bob")
            async with await _asgi_client(hub) as client:
                response = await client.post(
                    f"/v1/sessions/{session.session_id}/close",
                    json={"reason": "done", "requested_by": alice.actor_id},
                )
                assert response.status_code == 200
                assert response.json()["closed"] is True

                # State should be closed now.
                describe = await client.get(f"/v1/sessions/{session.session_id}")
                assert describe.json()["metadata"]["state"] == "closed"
                assert describe.json()["metadata"]["close_reason"] == "done"
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_close_session_rejects_non_participant(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two_actors(hub)
        try:
            mallory = await hc.register(
                _Echo("mallory"), identity=ActorIdentity(name="mallory")
            )
            session = await alice.open(SessionType.CONVERSATION, target="bob")
            async with await _asgi_client(hub) as client:
                response = await client.post(
                    f"/v1/sessions/{session.session_id}/close",
                    json={"requested_by": mallory.actor_id},
                )
                assert response.status_code == 403
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_read_wal_returns_envelopes(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two_actors(hub)
        try:
            async def quiet(*_args, **_kwargs):
                return None

            bob.on("conversation")(quiet)
            session = await alice.open(SessionType.CONVERSATION, target="bob")
            await session.send("hello from http test")
            await asyncio.sleep(0.02)

            async with await _asgi_client(hub) as client:
                response = await client.get(
                    f"/v1/sessions/{session.session_id}/wal"
                )
                assert response.status_code == 200
                envelopes = response.json()["envelopes"]
                # Should include at least the invite + ack + opened +
                # the user text envelope we sent.
                types = [e["event"]["type"] for e in envelopes]
                assert "ag2.msg.text" in types
                assert "ag2.session.invite" in types
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_read_wal_since_filters_early_bytes(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two_actors(hub)
        try:
            async def quiet(*_args, **_kwargs):
                return None

            bob.on("conversation")(quiet)
            session = await alice.open(SessionType.CONVERSATION, target="bob")
            first_id = await session.send("first")
            # Find the WAL offset AFTER the first send by reading full WAL
            # and scanning the raw bytes.
            full = await hub.read_wal(session.session_id)
            assert len(full) >= 1
            raw = await hub._store.read(
                f"/hub/sessions/{session.session_id}/wal.jsonl"
            )
            offset = len(raw.encode("utf-8"))
            await session.send("second")
            await asyncio.sleep(0.02)

            async with await _asgi_client(hub) as client:
                response = await client.get(
                    f"/v1/sessions/{session.session_id}/wal",
                    params={"since": offset},
                )
                envelopes = response.json()["envelopes"]
                contents = [
                    e["event"]["data"].get("content")
                    for e in envelopes
                    if e["event"]["type"] == "ag2.msg.text"
                ]
                # Only "second" should survive the filter.
                assert "first" not in contents
                assert "second" in contents
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_read_wal_invalid_since_returns_400(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two_actors(hub)
        try:
            async with await _asgi_client(hub) as client:
                response = await client.get(
                    "/v1/sessions/doesnt-matter/wal",
                    params={"since": "not-a-number"},
                )
                assert response.status_code == 400
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# Real uvicorn integration — one end-to-end smoke test
# ---------------------------------------------------------------------------


class TestRealUvicornIntegration:
    @pytest.mark.asyncio
    async def test_register_and_describe_over_real_http(self) -> None:
        """Spin a real uvicorn server and hit it with a real httpx client."""

        pytest.importorskip("uvicorn")

        hub = Hub(MemoryKnowledgeStore())
        server = HttpServer(hub, host="127.0.0.1", port=0)
        await server.serve()
        try:
            async with httpx.AsyncClient(base_url=server.url) as client:
                reg = await client.post(
                    "/v1/actors",
                    json={
                        "identity": {
                            "name": "alice",
                            "capabilities": ["research"],
                        }
                    },
                )
                assert reg.status_code == 201
                actor_id = reg.json()["actor_id"]

                describe = await client.get(f"/v1/actors/{actor_id}")
                assert describe.status_code == 200
                assert describe.json()["identity"]["name"] == "alice"

                listing = await client.get("/v1/actors")
                assert listing.status_code == 200
                assert len(listing.json()["actors"]) == 1
        finally:
            await server.close()
