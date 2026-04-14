# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``Hub.archive_closed_sessions`` — the Phase 3b background
sweeper that moves closed/expired session WALs to cold storage and
leaves a compact summary for read-only inspection.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from autogen.beta.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    Hub,
    HubClient,
    LocalLink,
    SessionType,
)
from autogen.beta.network.hub import layout


class _Echo:
    def __init__(self, name: str) -> None:
        self.name = name

    async def ask(self, content: str, **_kwargs):
        class _R:
            def __init__(self, body: str) -> None:
                self.body = body

        return _R(f"echo:{content}")


async def _run_session(hub: Hub) -> str:
    """Spin up two actors, run a short conversation, close the session."""

    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)
    alice = await hc.register(
        _Echo("alice"), identity=ActorIdentity(name="alice")
    )
    bob = await hc.register(
        _Echo("bob"), identity=ActorIdentity(name="bob")
    )
    session = await alice.open(SessionType.CONVERSATION, target="bob")
    await session.send("hello")
    await session.send("second")
    await hub.close_session(session.session_id)
    sid = session.session_id
    await hc.close()
    await link.close()
    return sid


# ---------------------------------------------------------------------------
# Happy path — closed session archived on demand
# ---------------------------------------------------------------------------


class TestArchiveClosedSessions:
    @pytest.mark.asyncio
    async def test_closed_session_moves_to_archive(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        session_id = await _run_session(hub)

        # Immediate sweep with age_threshold=0 picks up the session.
        archived = await hub.archive_closed_sessions(age_threshold_s=0.0)
        assert session_id in archived

        # Summary file exists at the archive location.
        summary_raw = await hub._store.read(
            f"{layout.archive_session_dir(session_id)}/summary.json"
        )
        assert summary_raw is not None
        summary = json.loads(summary_raw)
        assert summary["session_id"] == session_id
        assert summary["envelope_count"] > 0
        assert summary["state"] == "closed"
        assert "alice" in [p for p in summary["participants"]] or len(summary["participants"]) == 2
        assert summary["archived_at"]

        # Archive WAL copy exists.
        archive_wal = await hub._store.read(
            f"{layout.archive_session_dir(session_id)}/wal.jsonl"
        )
        assert archive_wal is not None
        assert len(archive_wal.strip().split("\n")) == summary["envelope_count"]

        # Live WAL is gone.
        live_wal = await hub._store.read(layout.session_wal(session_id))
        assert live_wal is None

        # Metadata still present, now stamped.
        meta = hub.peek_session(session_id)
        assert meta.archived_at is not None

    @pytest.mark.asyncio
    async def test_active_session_untouched(self) -> None:
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
            await session.send("hi")

            archived = await hub.archive_closed_sessions(age_threshold_s=0.0)
            assert session.session_id not in archived
            assert hub.peek_session(session.session_id).archived_at is None
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_sweeper_is_idempotent(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        session_id = await _run_session(hub)

        first = await hub.archive_closed_sessions(age_threshold_s=0.0)
        second = await hub.archive_closed_sessions(age_threshold_s=0.0)
        assert session_id in first
        assert session_id not in second  # already archived

    @pytest.mark.asyncio
    async def test_age_threshold_skips_fresh_sessions(self) -> None:
        """With a positive age threshold, fresh closes are not archived."""

        hub = Hub(MemoryKnowledgeStore())
        session_id = await _run_session(hub)
        archived = await hub.archive_closed_sessions(age_threshold_s=3600.0)
        assert session_id not in archived
        assert hub.peek_session(session_id).archived_at is None

    @pytest.mark.asyncio
    async def test_age_threshold_with_deterministic_now(self) -> None:
        """Inject a fake ``now`` to drive the sweeper without real time."""

        now_holder = {"value": "2026-04-13T12:00:00Z"}

        def fake_clock():
            return now_holder["value"]

        hub = Hub(MemoryKnowledgeStore(), clock=fake_clock)
        session_id = await _run_session(hub)
        # Advance the clock by 2 hours and sweep with a 1-hour threshold.
        now_holder["value"] = "2026-04-13T14:00:00Z"
        archived = await hub.archive_closed_sessions(
            age_threshold_s=3600.0, now="2026-04-13T14:00:00Z"
        )
        assert session_id in archived


# ---------------------------------------------------------------------------
# Expired sessions are also archived
# ---------------------------------------------------------------------------


class TestExpiredSessionArchival:
    @pytest.mark.asyncio
    async def test_expired_session_eligible(self) -> None:
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
            await session.send("hello")

            # Force-expire via the TTL sweeper.
            meta = hub._sessions[session.session_id]
            meta.expires_at = "1970-01-01T00:00:00Z"
            await hub.sweep_expired_sessions()
            assert meta.state.value == "expired"

            archived = await hub.archive_closed_sessions(age_threshold_s=0.0)
            assert session.session_id in archived
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# Scheduler callback
# ---------------------------------------------------------------------------


class TestArchiveSweepCallback:
    @pytest.mark.asyncio
    async def test_callback_is_async_and_callable(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        cb = hub.archive_sweep_callback(age_threshold_s=0.0)
        await cb(None, None)  # no-op on an empty hub

    @pytest.mark.asyncio
    async def test_callback_archives_closed_sessions(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        session_id = await _run_session(hub)
        cb = hub.archive_sweep_callback(age_threshold_s=0.0)
        await cb(None, None)
        assert hub.peek_session(session_id).archived_at is not None


# ---------------------------------------------------------------------------
# Disk round-trip + hydrate
# ---------------------------------------------------------------------------


class TestArchiveDiskRoundTrip:
    @pytest.mark.asyncio
    async def test_disk_archive_survives_hydrate(self, tmp_path) -> None:
        root = tmp_path / "hub"
        store = DiskKnowledgeStore(str(root))

        hub = Hub(store)
        session_id = await _run_session(hub)
        archived = await hub.archive_closed_sessions(age_threshold_s=0.0)
        assert session_id in archived

        # Reopen from disk; the session metadata should rehydrate with
        # archived_at intact.
        store2 = DiskKnowledgeStore(str(root))
        hub2 = await Hub.open(store2)
        try:
            meta = hub2.peek_session(session_id)
            assert meta is not None
            assert meta.archived_at is not None
            # The archive files are reachable at the archive path.
            raw_summary = await store2.read(
                f"{layout.archive_session_dir(session_id)}/summary.json"
            )
            assert raw_summary is not None
        finally:
            await hub2.close()


# ---------------------------------------------------------------------------
# Summary content
# ---------------------------------------------------------------------------


class TestArchiveSummaryContent:
    @pytest.mark.asyncio
    async def test_summary_preserves_basic_stats(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        session_id = await _run_session(hub)
        await hub.archive_closed_sessions(age_threshold_s=0.0)

        raw = await hub._store.read(
            f"{layout.archive_session_dir(session_id)}/summary.json"
        )
        summary = json.loads(raw)
        assert summary["type"] == "conversation"
        assert summary["envelope_count"] >= 2
        assert len(summary["participants"]) == 2
        assert summary["first_envelope_at"]
        assert summary["last_envelope_at"]
