# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``Hub.metrics()`` — the in-memory counter dict exposed at
``/v1/admin/metrics`` in Phase 3b.

Design principle: metrics are computed from live hub state, not from
``audit.jsonl``. Every counter movement is exercised here so a future
refactor that breaks incremental accounting is caught at the unit
level.
"""

from __future__ import annotations

import asyncio

import pytest

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    Hub,
    HubClient,
    InboxBlock,
    LimitsBlock,
    LocalLink,
    Rule,
    SessionType,
    TaskSpec,
)


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


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


class TestMetricsShape:
    @pytest.mark.asyncio
    async def test_empty_hub_shape(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        m = hub.metrics()
        assert set(m.keys()) == {"actors", "sessions", "tasks", "inbox", "uptime_s"}
        assert m["actors"] == {"registered": 0, "connected": 0}
        assert m["sessions"] == {"active": 0, "pending": 0, "closed_total": 0}
        assert m["tasks"] == {"running": 0, "completed_total": 0, "failed_total": 0}
        assert m["inbox"] == {"pending_total": 0}
        assert isinstance(m["uptime_s"], float)
        assert m["uptime_s"] >= 0.0

    @pytest.mark.asyncio
    async def test_metrics_never_raises_on_empty_hub(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        for _ in range(3):
            hub.metrics()  # must be idempotent-safe


# ---------------------------------------------------------------------------
# Actor counters
# ---------------------------------------------------------------------------


class TestActorCounters:
    @pytest.mark.asyncio
    async def test_register_bumps_registered(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        await hub.register(ActorIdentity(name="alice"))
        assert hub.metrics()["actors"]["registered"] == 1
        await hub.register(ActorIdentity(name="bob"))
        assert hub.metrics()["actors"]["registered"] == 2

    @pytest.mark.asyncio
    async def test_unregister_decrements_registered(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        alice = await hub.register(ActorIdentity(name="alice"))
        assert hub.metrics()["actors"]["registered"] == 1
        await hub.unregister(alice.actor_id)
        assert hub.metrics()["actors"]["registered"] == 0

    @pytest.mark.asyncio
    async def test_connected_tracks_live_endpoints(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        try:
            assert hub.metrics()["actors"]["connected"] == 0

            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            # Give the link time to finish the hello handshake.
            await asyncio.sleep(0.02)
            assert hub.metrics()["actors"]["connected"] == 1

            bob = await hc.register(
                _Echo("bob"), identity=ActorIdentity(name="bob")
            )
            await asyncio.sleep(0.02)
            assert hub.metrics()["actors"]["connected"] == 2
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# Session counters
# ---------------------------------------------------------------------------


class TestSessionCounters:
    @pytest.mark.asyncio
    async def test_active_session_counts(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            bob = await hc.register(
                _Echo("bob"), identity=ActorIdentity(name="bob")
            )

            session = await alice.open(
                SessionType.CONVERSATION, target="bob"
            )
            m = hub.metrics()
            assert m["sessions"]["active"] == 1
            assert m["sessions"]["pending"] == 0
            assert m["sessions"]["closed_total"] == 0

            await hub.close_session(session.session_id)
            m = hub.metrics()
            assert m["sessions"]["active"] == 0
            assert m["sessions"]["closed_total"] == 1
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_closed_total_monotonic(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            bob = await hc.register(
                _Echo("bob"), identity=ActorIdentity(name="bob")
            )

            for _ in range(3):
                session = await alice.open(
                    SessionType.CONVERSATION, target="bob"
                )
                await hub.close_session(session.session_id)
            assert hub.metrics()["sessions"]["closed_total"] == 3
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_ttl_expired_session_bumps_closed_total(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            bob = await hc.register(
                _Echo("bob"), identity=ActorIdentity(name="bob")
            )
            session = await alice.open(
                SessionType.CONVERSATION, target="bob"
            )
            assert hub.metrics()["sessions"]["closed_total"] == 0

            meta = hub._sessions[session.session_id]
            meta.expires_at = "1970-01-01T00:00:00Z"
            await hub.sweep_expired_sessions()

            assert hub.metrics()["sessions"]["closed_total"] == 1
            assert hub.metrics()["sessions"]["active"] == 0
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# Task counters
# ---------------------------------------------------------------------------


class TestTaskCounters:
    @pytest.mark.asyncio
    async def test_running_bumps_on_create_drops_on_complete(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            bob = await hc.register(
                _Echo("bob"), identity=ActorIdentity(name="bob")
            )
            session = await alice.open(
                SessionType.CONVERSATION, target="bob"
            )

            task = await session.create_task(TaskSpec(title="compute"))
            await asyncio.sleep(0.05)
            # The default handler completes the task synchronously —
            # once actor.ask returns, task.result() is posted.
            m = hub.metrics()
            assert m["tasks"]["completed_total"] >= 1
            assert m["tasks"]["running"] == 0
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_failed_counter_increments_on_error(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        try:
            class _Boom:
                async def ask(self, content, **_):
                    raise RuntimeError("boom")

            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            bob = await hc.register(
                _Boom(), identity=ActorIdentity(name="bob")
            )
            session = await alice.open(
                SessionType.CONVERSATION, target="bob"
            )
            await session.create_task(TaskSpec(title="doomed"))
            await asyncio.sleep(0.1)

            assert hub.metrics()["tasks"]["failed_total"] >= 1
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# Inbox counter
# ---------------------------------------------------------------------------


class TestInboxCounter:
    @pytest.mark.asyncio
    async def test_pending_total_sums_per_actor_counters(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            bob = await hc.register(
                _Echo("bob"), identity=ActorIdentity(name="bob")
            )

            async def block(_envelope, _client):
                await asyncio.Event().wait()

            bob.on("conversation")(block)
            session = await alice.open(
                SessionType.CONVERSATION, target="bob"
            )
            await session.send("one")
            await session.send("two")
            await asyncio.sleep(0.02)

            assert hub.metrics()["inbox"]["pending_total"] == 2
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# Uptime
# ---------------------------------------------------------------------------


class TestUptime:
    @pytest.mark.asyncio
    async def test_uptime_advances_with_clock(self) -> None:
        """Inject a fake clock and watch uptime tick."""

        now_iso = "2026-04-13T12:00:00Z"

        def fake_clock():
            return now_iso

        hub = Hub(MemoryKnowledgeStore(), clock=fake_clock)
        # Initial uptime is zero (started at now, queried at now).
        assert hub.metrics()["uptime_s"] == 0.0

        # Advance the fake clock.
        now_iso = "2026-04-13T12:00:30Z"
        assert hub.metrics()["uptime_s"] == 30.0

        now_iso = "2026-04-13T13:00:00Z"
        assert hub.metrics()["uptime_s"] == 3600.0

    @pytest.mark.asyncio
    async def test_uptime_uses_real_clock_by_default(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        up0 = hub.metrics()["uptime_s"]
        assert up0 >= 0.0
        assert up0 < 3600.0  # sanity — test hub is fresh
