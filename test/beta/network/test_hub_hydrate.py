# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for :meth:`Hub.hydrate` — cold-restart rebuild.

The hub holds its registry and live sessions in in-memory dicts for
hot-path access. On process restart those dicts start empty; hydrate
walks the backing store and rebuilds them. Without hydrate the first
restart with live sessions silently loses membership state — these
tests are the regression gate for that.

Invariants we assert:

* Every ``ActorIdentity`` + ``Rule`` on disk reappears in
  :attr:`Hub._identities` / :attr:`Hub._rules` / :attr:`Hub._name_to_id`.
* Every session (ACTIVE, CLOSED, EXPIRED) reappears in
  :attr:`Hub._sessions`.
* ``_active_sessions`` reserves slots only for ACTIVE sessions.
* Half-written PENDING sessions (no ``_PendingInvite`` future alive)
  are transitioned to EXPIRED so a lingering ack can't corrupt state.
* Post-hydrate ``post_envelope`` on a live session still works — so
  Phase 3 cross-process deployments can fail a process over with live
  sessions intact.
* SKILL.md sidecar is reattached to the in-memory identity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autogen.beta.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Rule,
    SessionState,
    SessionType,
)
from autogen.beta.network.hub import layout


# ---------------------------------------------------------------------------
# Minimal actor so we can register real clients
# ---------------------------------------------------------------------------


class _Echo:
    def __init__(self, name: str) -> None:
        self.name = name

    async def ask(self, content: str):
        class _R:
            def __init__(self, body: str) -> None:
                self.body = body

        return _R(f"echo:{content}")


async def _seed_store(tmp_path: Path) -> str:
    """Spin up a hub, register actors, run an exchange, return root path."""

    root = tmp_path / "hub-seed"
    store = DiskKnowledgeStore(str(root))
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)

    alice = await hc.register(
        _Echo("alice"),
        identity=ActorIdentity(
            name="alice",
            capabilities=["asking"],
            skill_md="## Alice\nAsks questions.",
        ),
    )
    bob = await hc.register(
        _Echo("bob"),
        identity=ActorIdentity(
            name="bob",
            capabilities=["answering"],
            skill_md="## Bob\nAnswers them.",
        ),
    )

    session = await alice.open(SessionType.CONVERSATION, target="bob")
    await session.send("how are you?")
    await hc.close()
    await link.close()
    # Keep in-memory references alive so GC doesn't close endpoints in
    # the middle of the test.
    return str(root)


# ---------------------------------------------------------------------------
# Core hydrate invariants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hydrate_rebuilds_identities_and_rules(tmp_path: Path) -> None:
    root = await _seed_store(tmp_path)

    hub2 = await Hub.open(DiskKnowledgeStore(root))

    names = sorted(hub2._name_to_id)
    assert names == ["alice", "bob"]
    alice_id = hub2._name_to_id["alice"]
    bob_id = hub2._name_to_id["bob"]
    assert hub2._identities[alice_id].name == "alice"
    assert hub2._identities[alice_id].capabilities == ["asking"]
    # SKILL.md sidecar should be re-attached.
    assert hub2._identities[alice_id].skill_md == "## Alice\nAsks questions."
    assert hub2._identities[bob_id].capabilities == ["answering"]
    assert isinstance(hub2._rules[alice_id], Rule)


@pytest.mark.asyncio
async def test_hydrate_rebuilds_sessions_and_active_slots(tmp_path: Path) -> None:
    root = await _seed_store(tmp_path)

    hub2 = await Hub.open(DiskKnowledgeStore(root))

    # One session on disk — conversation stays ACTIVE until explicit close.
    session_ids = list(hub2._sessions)
    assert len(session_ids) == 1
    meta = hub2._sessions[session_ids[0]]
    assert meta.type == "conversation"
    assert meta.state is SessionState.ACTIVE

    # Both participants should have the session counted in their
    # active-slots set.
    alice_id = hub2._name_to_id["alice"]
    bob_id = hub2._name_to_id["bob"]
    assert meta.session_id in hub2._active_sessions[alice_id]
    assert meta.session_id in hub2._active_sessions[bob_id]


@pytest.mark.asyncio
async def test_hydrate_skips_inactive_slots_for_closed_sessions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "seed"
    store = DiskKnowledgeStore(str(root))
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)
    alice = await hc.register(_Echo("alice"), identity=ActorIdentity(name="alice"))
    bob = await hc.register(_Echo("bob"), identity=ActorIdentity(name="bob"))
    session = await alice.open(SessionType.CONSULTING, target="bob")
    _ = await session.ask("q", timeout=2.0)
    # Consulting auto-closes after reply.
    await hc.close()
    await link.close()

    hub2 = await Hub.open(DiskKnowledgeStore(str(root)))

    alice_id = hub2._name_to_id["alice"]
    bob_id = hub2._name_to_id["bob"]
    # The session is loaded but its state is CLOSED, so it should not
    # occupy participant slots.
    assert len(hub2._sessions) == 1
    meta = next(iter(hub2._sessions.values()))
    assert meta.state is SessionState.CLOSED
    assert meta.session_id not in hub2._active_sessions[alice_id]
    assert meta.session_id not in hub2._active_sessions[bob_id]


@pytest.mark.asyncio
async def test_hydrate_expires_orphaned_pending_sessions(tmp_path: Path) -> None:
    """A PENDING session on disk has no live ``_PendingInvite`` future,
    so any incoming ack would have nowhere to resolve. Hydrate must
    transition such sessions to EXPIRED and mark the reason."""

    root = tmp_path / "seed"
    store = DiskKnowledgeStore(str(root))
    hub = Hub(store)
    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"))

    # Hand-craft a PENDING session metadata file on disk — simulating
    # a crash between "write metadata" and "receive invite ack".
    from autogen.beta.network.session_types import (
        Participant,
        ParticipantRole,
        SessionMetadata,
    )

    meta = SessionMetadata(
        session_id="01-stale",
        type="conversation",
        creator_id=alice.actor_id or "",
        participants=[
            Participant(
                actor_id=alice.actor_id or "",
                role=ParticipantRole.INITIATOR,
                joined_at="t0",
                order=0,
            ),
            Participant(
                actor_id=bob.actor_id or "",
                role=ParticipantRole.RESPONDENT,
                joined_at="t0",
                order=1,
            ),
        ],
        state=SessionState.PENDING,
        created_at="t0",
        expires_at="t1",
    )
    await store.write(layout.session_metadata(meta.session_id), meta.to_json())

    hub2 = await Hub.open(DiskKnowledgeStore(str(root)))

    restored = hub2._sessions["01-stale"]
    assert restored.state is SessionState.EXPIRED
    assert restored.close_reason == "hydrate_orphaned_pending"
    # The alice/bob slots should NOT be reserved by this dead session.
    alice_id = hub2._name_to_id["alice"]
    bob_id = hub2._name_to_id["bob"]
    assert "01-stale" not in hub2._active_sessions.get(alice_id, set())
    assert "01-stale" not in hub2._active_sessions.get(bob_id, set())


@pytest.mark.asyncio
async def test_hydrate_is_idempotent(tmp_path: Path) -> None:
    root = await _seed_store(tmp_path)
    hub2 = await Hub.open(DiskKnowledgeStore(root))
    names_before = sorted(hub2._name_to_id)
    session_count_before = len(hub2._sessions)
    await hub2.hydrate()  # second hydrate is a no-op
    assert sorted(hub2._name_to_id) == names_before
    assert len(hub2._sessions) == session_count_before


@pytest.mark.asyncio
async def test_hydrate_missing_rule_file_skips_actor_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    root = tmp_path / "seed"
    store = DiskKnowledgeStore(str(root))
    hub = Hub(store)
    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"))

    # Corrupt Bob's rule file by deleting it.
    await store.delete(layout.actor_rule(bob.actor_id or ""))

    with caplog.at_level(logging.WARNING, logger="autogen.beta.network.hub"):
        hub2 = await Hub.open(DiskKnowledgeStore(str(root)))

    # Alice survives.
    assert "alice" in hub2._name_to_id
    # Bob is dropped because his rule is missing.
    assert "bob" not in hub2._name_to_id
    assert any("missing identity/rule" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_hydrate_live_session_still_accepts_envelopes(tmp_path: Path) -> None:
    """After hydrate, the rebuilt session must accept new envelopes.

    This catches regressions where hydrate populates the metadata but
    forgets to re-key the adapters, participants, or active_sessions
    slots in a way that ``post_envelope`` would need."""

    root = await _seed_store(tmp_path)
    hub2 = await Hub.open(DiskKnowledgeStore(root))

    alice_id = hub2._name_to_id["alice"]
    bob_id = hub2._name_to_id["bob"]
    session_id = next(iter(hub2._sessions))

    env = Envelope.text(
        session_id=session_id,
        sender_id=bob_id,
        content="I am fine thanks",
        recipient_id=alice_id,
    )
    envelope_id, wal_offset = await hub2.post_envelope(env)
    assert envelope_id
    assert wal_offset > 0

    # The appended envelope shows up in the WAL.
    prior = await hub2.read_wal(session_id)
    assert any(e.event_data.get("content") == "I am fine thanks" for e in prior)


@pytest.mark.asyncio
async def test_hydrate_does_nothing_on_fresh_store() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store)
    assert hub._identities == {}
    assert hub._sessions == {}
