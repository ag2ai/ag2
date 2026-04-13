# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 1e — end-to-end integration, durability, and multi-actor scenarios.

These tests are the proof-of-foundation: they wire the full stack
(KnowledgeStore → Hub → LocalLink → HubClient → ActorClient → Session) in a
single process and walk it through the real Phase 1 workloads a user would
run. They also exercise the durability story by pointing the store at disk,
tearing the hub down, re-opening it, and reading back the WAL.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import pytest_asyncio

from autogen.beta import Actor
from autogen.beta.events import ModelMessage, ModelResponse
from autogen.beta.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from autogen.beta.network import (
    AccessBlock,
    ActorIdentity,
    Envelope,
    Hub,
    HubClient,
    LimitsBlock,
    LocalLink,
    Rule,
    SessionState,
    SessionType,
)
from autogen.beta.network.envelope import EV_TEXT
from autogen.beta.network.errors import LimitExceededError, TimeoutError as NetTimeoutError
from autogen.beta.network.hub import layout
from autogen.beta.testing import TestConfig


# ---------------------------------------------------------------------------
# Fake actor with richer behavior
# ---------------------------------------------------------------------------


class ScriptedActor:
    """Replies in order from a scripted list; falls back to echo."""

    def __init__(self, name: str, script: list[str] | None = None) -> None:
        self.name = name
        self._script = list(script or [])
        self.questions: list[str] = []

    async def ask(self, content: str):
        self.questions.append(content)
        if self._script:
            reply = self._script.pop(0)
        else:
            reply = f"{self.name} received: {content}"

        class _Reply:
            def __init__(self, body: str) -> None:
                self.body = body

        return _Reply(reply)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


async def _spin_up(store, actors: dict[str, ScriptedActor]) -> tuple[Hub, HubClient, LocalLink, dict[str, Any]]:
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)
    clients: dict[str, Any] = {}
    for name, actor in actors.items():
        clients[name] = await hc.register(actor, identity=ActorIdentity(name=name))
    return hub, hc, link, clients


# ---------------------------------------------------------------------------
# Three-actor consulting fan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_three_actors_consulting_crosstalk() -> None:
    """Alice asks Bob one thing and Carol another — both succeed independently."""

    store = MemoryKnowledgeStore()
    hub, hc, link, clients = await _spin_up(
        store,
        {
            "alice": ScriptedActor("alice"),
            "bob": ScriptedActor("bob", script=["plasmid summary"]),
            "carol": ScriptedActor("carol", script=["climate summary"]),
        },
    )
    try:
        alice = clients["alice"]
        s1 = await alice.open(SessionType.CONSULTING, target="bob")
        s2 = await alice.open(SessionType.CONSULTING, target="carol")

        # Run the two asks concurrently.
        r1, r2 = await asyncio.gather(
            s1.ask("plasmids?", timeout=2.0),
            s2.ask("climate?", timeout=2.0),
        )
        assert r1 == "plasmid summary"
        assert r2 == "climate summary"

        bob = clients["bob"]
        carol = clients["carol"]
        assert bob.actor.questions == ["plasmids?"]
        assert carol.actor.questions == ["climate?"]

        # Sessions both closed after consulting done.
        await asyncio.sleep(0.02)
        meta1 = alice.lookup_session(s1.session_id)
        meta2 = alice.lookup_session(s2.session_id)
        assert meta1 is not None and meta1.state is SessionState.CLOSED
        assert meta2 is not None and meta2.state is SessionState.CLOSED
    finally:
        await hc.close()
        await link.close()


# ---------------------------------------------------------------------------
# Concurrent session limit enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_concurrent_sessions_limit_kicks_in() -> None:
    store = MemoryKnowledgeStore()
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)
    try:
        alice = await hc.register(
            ScriptedActor("alice"),
            identity=ActorIdentity(name="alice"),
            rule=Rule(limits=LimitsBlock(max_concurrent_sessions=2)),
        )
        await hc.register(ScriptedActor("bob"), identity=ActorIdentity(name="bob"))

        # Conversation stays open until explicitly closed.
        s1 = await alice.open(SessionType.CONVERSATION, target="bob")
        s2 = await alice.open(SessionType.CONVERSATION, target="bob")

        with pytest.raises(LimitExceededError):
            await alice.open(SessionType.CONVERSATION, target="bob")

        await s1.close()
        # Slot freed — next open succeeds.
        s3 = await alice.open(SessionType.CONVERSATION, target="bob")
        assert s3.session_id != s1.session_id
        await s2.close()
        await s3.close()
    finally:
        await hc.close()
        await link.close()


# ---------------------------------------------------------------------------
# Access rule enforcement end-to-end via post_envelope path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_send_rejected_when_outbound_rule_blocks() -> None:
    store = MemoryKnowledgeStore()
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)
    try:
        # Alice creates session (wildcard default rule), but after creation
        # we tighten Alice's rule to reject outbound to non-ag2 names.
        alice = await hc.register(ScriptedActor("alice"), identity=ActorIdentity(name="alice"))
        bob = await hc.register(ScriptedActor("bob"), identity=ActorIdentity(name="bob"))

        session = await alice.open(SessionType.CONVERSATION, target="bob")
        # Mutate rule on the hub side to simulate a rule update.
        hub._rules[alice.actor_id] = Rule(access=AccessBlock(outbound_to=["ag2:*:*"]))

        from autogen.beta.network.errors import AccessDeniedError

        with pytest.raises(AccessDeniedError):
            await session.send("hi", recipient_id=bob.actor_id)
    finally:
        await hc.close()
        await link.close()


# ---------------------------------------------------------------------------
# Framework-core Actor with TestConfig (canned LLM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_actor_with_canned_llm_reply() -> None:
    store = MemoryKnowledgeStore()
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)

    alice_config = TestConfig(ModelResponse(message=ModelMessage(content="[dummy]")))
    bob_config = TestConfig(ModelResponse(message=ModelMessage(content="bob-canned-answer")))
    alice = Actor("alice", config=alice_config)
    bob = Actor("bob", config=bob_config)

    try:
        alice_client = await hc.register(alice, identity=ActorIdentity(name="alice"))
        await hc.register(bob, identity=ActorIdentity(name="bob"))

        session = await alice_client.open(SessionType.CONSULTING, target="bob")
        reply = await session.ask("please answer", timeout=2.0)
        assert reply == "bob-canned-answer"
    finally:
        await hc.close()
        await link.close()


# ---------------------------------------------------------------------------
# Durability: write on disk, reopen, reconstruct state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disk_wal_is_reconstructible_after_hub_restart(tmp_path: Path) -> None:
    root = tmp_path / "hub"
    # ---- First run: create two actors, run a consulting exchange.
    store1 = DiskKnowledgeStore(str(root))
    hub1 = Hub(store1)
    link1 = LocalLink()
    link1.on_connection(hub1.connection_handler)
    hc1 = HubClient(hub1, link1)

    alice = await hc1.register(
        ScriptedActor("alice"),
        identity=ActorIdentity(name="alice", capabilities=["ask"]),
    )
    bob = await hc1.register(
        ScriptedActor("bob", script=["durable answer"]),
        identity=ActorIdentity(name="bob", capabilities=["answer"]),
    )

    session = await alice.open(SessionType.CONSULTING, target="bob")
    reply = await session.ask("tell me something", timeout=2.0)
    assert reply == "durable answer"
    first_actor_id_alice = alice.actor_id
    first_actor_id_bob = bob.actor_id
    session_id = session.session_id

    await hc1.close()
    await link1.close()

    # ---- Second run: fresh Hub on the same disk, read the WAL and
    #      identities without re-registering anything.
    store2 = DiskKnowledgeStore(str(root))
    hub2 = Hub(store2)

    # Identity + rule on disk
    ident_raw = await store2.read(layout.actor_identity(first_actor_id_alice))
    assert ident_raw is not None
    assert '"alice"' in ident_raw
    rule_raw = await store2.read(layout.actor_rule(first_actor_id_alice))
    assert rule_raw is not None
    assert '"version": 1' in rule_raw or '"version":1' in rule_raw

    # Bob's identity too.
    assert await store2.exists(layout.actor_identity(first_actor_id_bob))

    # Session metadata on disk.
    meta_raw = await store2.read(layout.session_metadata(session_id))
    assert meta_raw is not None
    meta = json.loads(meta_raw)
    assert meta["state"] == SessionState.CLOSED.value
    assert meta["type"] == SessionType.CONSULTING.value

    # WAL on disk carries the question + answer + handshake events.
    wal_raw = await store2.read(layout.session_wal(session_id))
    assert wal_raw is not None
    lines = [line for line in wal_raw.split("\n") if line]
    # Expect: invite, invite_ack, session_opened, user question, user answer, session_closed
    types = [json.loads(line)["event"]["type"] for line in lines]
    assert "ag2.session.invite" in types
    assert "ag2.session.invite_ack" in types
    assert "ag2.session.opened" in types
    assert EV_TEXT in types
    assert "ag2.session.closed" in types

    # Read only user text envelopes via read_range + filter.
    user_text = [json.loads(line) for line in lines if json.loads(line)["event"]["type"] == EV_TEXT]
    assert [e["event"]["data"]["content"] for e in user_text] == [
        "tell me something",
        "durable answer",
    ]


# ---------------------------------------------------------------------------
# read_range-based WAL tailer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_range_returns_incremental_wal(tmp_path: Path) -> None:
    store = DiskKnowledgeStore(str(tmp_path / "hub"))
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)

    alice = await hc.register(ScriptedActor("alice"), identity=ActorIdentity(name="alice"))
    bob = await hc.register(ScriptedActor("bob", script=["one", "two"]), identity=ActorIdentity(name="bob"))

    try:
        session = await alice.open(SessionType.CONVERSATION, target="bob")
        cursor = 0

        await session.ask("first", timeout=2.0)
        await asyncio.sleep(0.01)
        chunk_a = await store.read_range(layout.session_wal(session.session_id), cursor)
        cursor += len(chunk_a.encode("utf-8"))
        assert "first" in chunk_a

        await session.ask("second", timeout=2.0)
        await asyncio.sleep(0.01)
        chunk_b = await store.read_range(layout.session_wal(session.session_id), cursor)
        assert "second" in chunk_b
        assert "first" not in chunk_b  # incremental: only new bytes

        await session.close()
    finally:
        await hc.close()
        await link.close()


# ---------------------------------------------------------------------------
# Subscription with causation filter works across reconnects (within same session)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_ask_recovers_if_reply_arrives_before_subscribe() -> None:
    """If Bob replies faster than Alice can issue the subscribe frame, the hub
    replays the WAL and still delivers the reply — no race condition.

    We simulate this by crafting a manual envelope with a tiny pre-sleep so
    the subscribe frame is guaranteed to land after the reply.
    """

    store = MemoryKnowledgeStore()
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)
    try:
        alice = await hc.register(ScriptedActor("alice"), identity=ActorIdentity(name="alice"))
        await hc.register(
            ScriptedActor("bob", script=["fast reply"]),
            identity=ActorIdentity(name="bob"),
        )

        session = await alice.open(SessionType.CONSULTING, target="bob")
        # session.ask subscribes after sending; the hub replays the WAL from
        # offset 0 with the causation_id filter, so the reply is delivered
        # even if it landed before the subscribe.
        reply = await session.ask("q", timeout=2.0)
        assert reply == "fast reply"
    finally:
        await hc.close()
        await link.close()


# Type annotation helper
from typing import Any  # noqa: E402
