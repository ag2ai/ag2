# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Phase 3b HTTP routes added on top of the Phase 3a slice.

Six new routes land in 3b:

* ``GET /v1/sessions?state=&participant=&type=&limit=``
* ``POST /v1/sessions/{id}/force-close``
* ``GET /v1/actors/{id}/activity``
* ``GET /v1/actors/{id}/knowledge/{path:path}`` — KnowledgeAccess gated
* ``PUT /v1/actors/{id}/rule``
* ``GET /v1/admin/health``
* ``GET /v1/admin/metrics``

Task endpoints land as explicit 404 stubs. Every happy path + at least
one failure path is exercised.
"""

from __future__ import annotations

import pytest

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    AccessBlock,
    ActorIdentity,
    Hub,
    HubClient,
    KnowledgeAccess,
    LimitsBlock,
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
    transport = httpx.ASGITransport(app=build_app(hub))
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


async def _register_two(hub: Hub):
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
# PUT /v1/actors/{id}/rule
# ---------------------------------------------------------------------------


class TestUpdateRule:
    @pytest.mark.asyncio
    async def test_put_replaces_rule(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore())
        try:
            stamped = await hub.register(ActorIdentity(name="alice"))
            new_rule = Rule(limits=LimitsBlock(max_concurrent_sessions=7))

            async with await _asgi_client(hub) as client:
                r = await client.put(
                    f"/v1/actors/{stamped.actor_id}/rule",
                    json={"rule": new_rule.to_dict()},
                )
                assert r.status_code == 200
                body = r.json()
                assert body["actor_id"] == stamped.actor_id
                assert body["rule"]["limits"]["max_concurrent_sessions"] == 7

            # Hub in-memory cache updated too.
            refreshed = await hub.get_rule(stamped.actor_id)
            assert refreshed.limits.max_concurrent_sessions == 7
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_put_unknown_actor_returns_404(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            r = await client.put(
                "/v1/actors/ghost-id/rule",
                json={"rule": Rule().to_dict()},
            )
            assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_put_malformed_body_returns_400(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        stamped = await hub.register(ActorIdentity(name="alice"))
        async with await _asgi_client(hub) as client:
            r = await client.put(
                f"/v1/actors/{stamped.actor_id}/rule",
                json={},  # missing 'rule'
            )
            assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_put_invalid_rule_returns_400(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        stamped = await hub.register(ActorIdentity(name="alice"))
        async with await _asgi_client(hub) as client:
            r = await client.put(
                f"/v1/actors/{stamped.actor_id}/rule",
                json={"rule": {"limits": {"inbox": {"overflow": "explode"}}}},
            )
            assert r.status_code == 400


# ---------------------------------------------------------------------------
# GET /v1/sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_all_sessions(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two(hub)
        try:
            s1 = await alice.open(SessionType.CONVERSATION, target="bob")
            s2 = await alice.open(SessionType.CONSULTING, target="bob")
            async with await _asgi_client(hub) as client:
                r = await client.get("/v1/sessions")
                assert r.status_code == 200
                body = r.json()
                assert body["count"] >= 2
                ids = {s["session_id"] for s in body["sessions"]}
                assert s1.session_id in ids
                # s2 (consulting) closes itself after one ask but we
                # didn't call ask, so it may still be active.
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_list_filters_by_type(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two(hub)
        try:
            await alice.open(SessionType.CONVERSATION, target="bob")
            await alice.open(SessionType.NOTIFICATION, target="bob")
            async with await _asgi_client(hub) as client:
                r = await client.get("/v1/sessions", params={"type": "conversation"})
                body = r.json()
                for s in body["sessions"]:
                    assert s["type"] == "conversation"
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_list_filters_by_state(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two(hub)
        try:
            session = await alice.open(SessionType.CONVERSATION, target="bob")
            await hub.close_session(session.session_id)

            async with await _asgi_client(hub) as client:
                r_active = await client.get(
                    "/v1/sessions", params={"state": "active"}
                )
                r_closed = await client.get(
                    "/v1/sessions", params={"state": "closed"}
                )
                assert r_active.status_code == 200
                assert r_closed.status_code == 200
                assert all(
                    s["state"] == "active" for s in r_active.json()["sessions"]
                )
                assert any(
                    s["session_id"] == session.session_id
                    for s in r_closed.json()["sessions"]
                )
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_list_filters_by_participant(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two(hub)
        try:
            carol = await hc.register(
                _Echo("carol"), identity=ActorIdentity(name="carol")
            )
            await alice.open(SessionType.CONVERSATION, target="bob")
            async with await _asgi_client(hub) as client:
                # Only Bob's sessions — alice↔bob matches.
                r = await client.get(
                    "/v1/sessions", params={"participant": "bob"}
                )
                assert r.status_code == 200
                assert r.json()["count"] >= 1
                # Carol isn't in any session.
                r2 = await client.get(
                    "/v1/sessions", params={"participant": "carol"}
                )
                assert r2.status_code == 200
                assert r2.json()["count"] == 0
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_list_limit_cap(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two(hub)
        try:
            for _ in range(5):
                await alice.open(SessionType.CONVERSATION, target="bob")
            async with await _asgi_client(hub) as client:
                r = await client.get("/v1/sessions", params={"limit": "2"})
                body = r.json()
                assert len(body["sessions"]) == 2
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_list_unknown_participant_returns_404(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            r = await client.get(
                "/v1/sessions", params={"participant": "ghost"}
            )
            assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /v1/sessions/{id}/force-close
# ---------------------------------------------------------------------------


class TestForceClose:
    @pytest.mark.asyncio
    async def test_force_close_active_session(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two(hub)
        try:
            session = await alice.open(SessionType.CONVERSATION, target="bob")
            async with await _asgi_client(hub) as client:
                r = await client.post(
                    f"/v1/sessions/{session.session_id}/force-close",
                    json={"reason": "admin_test"},
                )
                assert r.status_code == 200
                assert r.json()["closed"] is True

            meta = hub.peek_session(session.session_id)
            assert meta.state.value == "closed"
            assert meta.close_reason == "admin_test"
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_force_close_unknown_session_returns_404(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            r = await client.post(
                "/v1/sessions/ghost-session-id/force-close",
                json={},
            )
            assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_force_close_default_reason(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two(hub)
        try:
            session = await alice.open(SessionType.CONVERSATION, target="bob")
            async with await _asgi_client(hub) as client:
                # No body at all.
                r = await client.post(
                    f"/v1/sessions/{session.session_id}/force-close"
                )
                assert r.status_code == 200
            assert hub.peek_session(session.session_id).close_reason == "admin_force_close"
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# GET /v1/actors/{id}/activity
# ---------------------------------------------------------------------------


class TestActorActivity:
    @pytest.mark.asyncio
    async def test_activity_lists_sessions_and_tasks(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two(hub)
        try:
            await alice.open(SessionType.CONVERSATION, target="bob")
            async with await _asgi_client(hub) as client:
                r = await client.get(f"/v1/actors/{alice.actor_id}/activity")
                assert r.status_code == 200
                body = r.json()
                assert body["actor_id"] == alice.actor_id
                assert len(body["sessions"]) >= 1
                assert "tasks" in body
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_activity_by_name(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice, bob, hc, link = await _register_two(hub)
        try:
            await alice.open(SessionType.CONVERSATION, target="bob")
            async with await _asgi_client(hub) as client:
                r = await client.get("/v1/actors/alice/activity")
                assert r.status_code == 200
                assert r.json()["actor_id"] == alice.actor_id
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_activity_unknown_actor_404(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            r = await client.get("/v1/actors/ghost/activity")
            assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /v1/actors/{id}/knowledge/{path:path}
# ---------------------------------------------------------------------------


class TestKnowledgeRead:
    @pytest.mark.asyncio
    async def test_allowed_reader_and_path(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        rule = Rule(
            access=AccessBlock(
                knowledge=KnowledgeAccess(
                    expose=["/public/**"],
                    readers=["ag2:*:*"],
                )
            )
        )
        stamped = await hub.register(
            ActorIdentity(name="owner"), rule=rule
        )
        # Pre-populate the owner's knowledge store slice.
        await hub._store.write(
            f"/actors/{stamped.actor_id}/knowledge/public/hello.txt",
            "hello world",
        )
        async with await _asgi_client(hub) as client:
            r = await client.get(
                f"/v1/actors/{stamped.actor_id}/knowledge/public/hello.txt",
                headers={"X-Ag2-Reader": "ag2:writer:1"},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["path"] == "/public/hello.txt"
            assert body["content"] == "hello world"

    @pytest.mark.asyncio
    async def test_missing_reader_header_returns_403(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        rule = Rule(
            access=AccessBlock(
                knowledge=KnowledgeAccess(expose=["/**"], readers=["*"])
            )
        )
        stamped = await hub.register(ActorIdentity(name="owner"), rule=rule)
        async with await _asgi_client(hub) as client:
            r = await client.get(
                f"/v1/actors/{stamped.actor_id}/knowledge/public/hello.txt",
            )
            assert r.status_code == 403

    @pytest.mark.asyncio
    async def test_denied_reader_pattern_returns_403(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        rule = Rule(
            access=AccessBlock(
                knowledge=KnowledgeAccess(
                    expose=["/public/**"],
                    readers=["ag2:*:*"],
                )
            )
        )
        stamped = await hub.register(ActorIdentity(name="owner"), rule=rule)
        async with await _asgi_client(hub) as client:
            r = await client.get(
                f"/v1/actors/{stamped.actor_id}/knowledge/public/hello.txt",
                headers={"X-Ag2-Reader": "acme:other:1"},
            )
            assert r.status_code == 403

    @pytest.mark.asyncio
    async def test_denied_path_returns_403(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        rule = Rule(
            access=AccessBlock(
                knowledge=KnowledgeAccess(
                    expose=["/public/**"],
                    readers=["*"],
                )
            )
        )
        stamped = await hub.register(ActorIdentity(name="owner"), rule=rule)
        async with await _asgi_client(hub) as client:
            r = await client.get(
                f"/v1/actors/{stamped.actor_id}/knowledge/private/secret.txt",
                headers={"X-Ag2-Reader": "ag2:writer:1"},
            )
            assert r.status_code == 403

    @pytest.mark.asyncio
    async def test_missing_file_returns_404(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        rule = Rule(
            access=AccessBlock(
                knowledge=KnowledgeAccess(expose=["/**"], readers=["*"])
            )
        )
        stamped = await hub.register(ActorIdentity(name="owner"), rule=rule)
        async with await _asgi_client(hub) as client:
            r = await client.get(
                f"/v1/actors/{stamped.actor_id}/knowledge/missing.txt",
                headers={"X-Ag2-Reader": "ag2:writer:1"},
            )
            assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_unknown_actor_returns_404(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            r = await client.get(
                "/v1/actors/ghost-id/knowledge/any.txt",
                headers={"X-Ag2-Reader": "ag2:r:1"},
            )
            assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_empty_knowledge_block_denies_by_default(self) -> None:
        """The default rule exposes nothing."""

        hub = Hub(MemoryKnowledgeStore())
        stamped = await hub.register(ActorIdentity(name="owner"))
        await hub._store.write(
            f"/actors/{stamped.actor_id}/knowledge/any/file.txt", "data"
        )
        async with await _asgi_client(hub) as client:
            r = await client.get(
                f"/v1/actors/{stamped.actor_id}/knowledge/any/file.txt",
                headers={"X-Ag2-Reader": "ag2:any:1"},
            )
            assert r.status_code == 403

    @pytest.mark.asyncio
    async def test_nested_path_preserved(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        rule = Rule(
            access=AccessBlock(
                knowledge=KnowledgeAccess(expose=["/**"], readers=["*"])
            )
        )
        stamped = await hub.register(ActorIdentity(name="owner"), rule=rule)
        await hub._store.write(
            f"/actors/{stamped.actor_id}/knowledge/a/b/c/deep.txt", "deep"
        )
        async with await _asgi_client(hub) as client:
            r = await client.get(
                f"/v1/actors/{stamped.actor_id}/knowledge/a/b/c/deep.txt",
                headers={"X-Ag2-Reader": "ag2:r:1"},
            )
            assert r.status_code == 200
            assert r.json()["content"] == "deep"


# ---------------------------------------------------------------------------
# GET /v1/admin/health
# ---------------------------------------------------------------------------


class TestAdminHealth:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            r = await client.get("/v1/admin/health")
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "ok"
            assert body["hub_id"] == hub.config.hub_id


# ---------------------------------------------------------------------------
# GET /v1/admin/metrics
# ---------------------------------------------------------------------------


class TestAdminMetrics:
    @pytest.mark.asyncio
    async def test_metrics_shape(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            r = await client.get("/v1/admin/metrics")
            assert r.status_code == 200
            body = r.json()
            assert set(body.keys()) == {
                "actors",
                "sessions",
                "tasks",
                "inbox",
                "uptime_s",
            }

    @pytest.mark.asyncio
    async def test_metrics_counters_reflect_registration(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        await hub.register(ActorIdentity(name="alice"))
        await hub.register(ActorIdentity(name="bob"))

        async with await _asgi_client(hub) as client:
            r = await client.get("/v1/admin/metrics")
            body = r.json()
            assert body["actors"]["registered"] == 2


# ---------------------------------------------------------------------------
# /v1/tasks/* — explicit 404 stubs (Phase 6)
# ---------------------------------------------------------------------------


class TestTaskStubs:
    @pytest.mark.asyncio
    async def test_get_tasks_list_returns_404(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            r = await client.get("/v1/tasks")
            assert r.status_code == 404
            assert r.json()["error"] == "NotImplemented"

    @pytest.mark.asyncio
    async def test_get_task_by_id_returns_404(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            r = await client.get("/v1/tasks/01932-xxx")
            assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_task_returns_404(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        async with await _asgi_client(hub) as client:
            r = await client.post("/v1/tasks/01932-xxx/cancel")
            assert r.status_code == 404
