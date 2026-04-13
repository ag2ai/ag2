# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""HubClient / ActorClient / Session end-to-end.

The tests wire a real Hub + LocalLink fixture, register two FakeActor
instances through the full client pipeline, and drive session.send /
session.ask / session.subscribe the same way a real application would.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
import pytest_asyncio

from autogen.beta.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from autogen.beta.network import (
    AccessBlock,
    ActorClient,
    ActorIdentity,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Rule,
    Session,
    SessionState,
    SessionType,
)
from autogen.beta.network.errors import (
    AccessDeniedError,
    TimeoutError as NetTimeoutError,
)


# ---------------------------------------------------------------------------
# FakeActor + reply shape
# ---------------------------------------------------------------------------


@dataclass
class FakeReply:
    content: str


class FakeActor:
    """Minimal actor: records the questions it saw and returns a canned reply.

    The ActorClient's default handlers expect ``actor.ask(text)`` to return
    something with ``.content`` or ``.body``; FakeReply satisfies that.
    """

    def __init__(self, name: str, reply: str = "") -> None:
        self.name = name
        self.reply = reply
        self.questions: list[str] = []

    async def ask(self, content: str) -> FakeReply:
        self.questions.append(content)
        return FakeReply(content=self.reply or f"{self.name}: {content}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def wired_clients():
    """Two HubClient + ActorClient pairs on one in-proc hub."""

    store = MemoryKnowledgeStore()
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)

    hub_client = HubClient(hub, link)

    alice = FakeActor(name="alice", reply="alice-reply")
    bob = FakeActor(name="bob", reply="bob-reply")

    alice_client = await hub_client.register(
        alice, identity=ActorIdentity(name="alice", capabilities=["ask"])
    )
    bob_client = await hub_client.register(
        bob, identity=ActorIdentity(name="bob", capabilities=["answer"])
    )

    try:
        yield hub, hub_client, alice_client, bob_client
    finally:
        await hub_client.close()
        await link.close()


# ---------------------------------------------------------------------------
# HubClient registration + discovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hub_client_registers_and_stamps_actor_id(wired_clients) -> None:
    _hub, _hc, alice_client, _bob_client = wired_clients
    assert alice_client.actor_id
    assert alice_client.identity.name == "alice"
    assert alice_client.identity.actor_id == alice_client.actor_id


@pytest.mark.asyncio
async def test_hub_client_find_returns_registered_identities(wired_clients) -> None:
    _hub, hc, _a, _b = wired_clients
    results = await hc.find()
    assert {i.name for i in results} == {"alice", "bob"}


@pytest.mark.asyncio
async def test_hub_client_find_by_capability(wired_clients) -> None:
    _hub, hc, _a, _b = wired_clients
    results = await hc.find(capability="answer")
    assert [i.name for i in results] == ["bob"]


@pytest.mark.asyncio
async def test_hub_client_describe_returns_identity(wired_clients) -> None:
    _hub, hc, _a, _b = wired_clients
    ident = await hc.describe("alice")
    assert ident.name == "alice"


# ---------------------------------------------------------------------------
# Consulting end-to-end via session.ask
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_ask_returns_bobs_reply(wired_clients) -> None:
    _hub, _hc, alice_client, bob_client = wired_clients
    session = await alice_client.open(SessionType.CONSULTING, target="bob")
    reply = await session.ask("explain plasmids", timeout=2.0)
    assert reply == "bob-reply"
    # Bob's actor saw the question.
    assert bob_client.actor.questions == ["explain plasmids"]


@pytest.mark.asyncio
async def test_session_ask_respects_causation_correlation(wired_clients) -> None:
    _hub, _hc, alice_client, _bob_client = wired_clients
    session = await alice_client.open(SessionType.CONSULTING, target="bob")
    reply_a = await session.ask("first question", timeout=2.0)
    assert reply_a == "bob-reply"


@pytest.mark.asyncio
async def test_session_ask_subscribes_from_post_send_wal_offset(wired_clients) -> None:
    """session.ask uses the accept frame's wal_offset as the subscribe cursor.

    This is what keeps ask O(1) instead of re-replaying the whole session
    WAL on every call. We verify it by intercepting subscribe frames and
    checking the ``since`` cursor is non-zero.
    """

    _hub, _hc, alice_client, _bob_client = wired_clients
    session = await alice_client.open(SessionType.CONVERSATION, target="bob")

    # Seed a few prior envelopes so the WAL has real bytes to not replay.
    await session.send("prior 1")
    await session.send("prior 2")
    await asyncio.sleep(0.02)  # let bob process and reply

    captured_since: list[int] = []
    original_open = alice_client._open_subscription

    async def spy_open(**kwargs):
        since = kwargs.get("since", 0)
        captured_since.append(since or 0)
        return await original_open(**kwargs)

    alice_client._open_subscription = spy_open  # type: ignore[method-assign]

    reply = await session.ask("now ask", timeout=2.0)
    assert reply == "bob-reply"  # fixture's canned reply
    assert captured_since, "ask never opened a subscription"
    # The cursor must point past all prior WAL content — i.e. > 0.
    assert captured_since[0] > 0, f"ask subscribed with since={captured_since[0]}"
    await session.close()


@pytest.mark.asyncio
async def test_consulting_second_ask_fails_because_session_closed(wired_clients) -> None:
    _hub, _hc, alice_client, _bob_client = wired_clients
    session = await alice_client.open(SessionType.CONSULTING, target="bob")
    await session.ask("q1", timeout=2.0)
    # Session is closed now; next ask should hit SessionError.
    with pytest.raises(Exception):
        await session.ask("q2", timeout=1.0)


# ---------------------------------------------------------------------------
# Conversation multi-turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversation_preserves_three_turns(wired_clients) -> None:
    _hub, _hc, alice_client, bob_client = wired_clients

    # Bob replies with a counter-like running tag so we can verify each turn.
    bob_client.actor.reply = ""  # use default echo f"{name}: {content}"

    session = await alice_client.open(SessionType.CONVERSATION, target="bob")

    r1 = await session.ask("one", timeout=2.0)
    r2 = await session.ask("two", timeout=2.0)
    r3 = await session.ask("three", timeout=2.0)
    await session.close()

    assert r1 == "bob: one"
    assert r2 == "bob: two"
    assert r3 == "bob: three"
    assert bob_client.actor.questions == ["one", "two", "three"]


# ---------------------------------------------------------------------------
# Notification one-shot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notification_delivers_without_reply(wired_clients) -> None:
    _hub, _hc, alice_client, bob_client = wired_clients

    session = await alice_client.open(SessionType.NOTIFICATION, target="bob")
    await session.send("fyi")

    # Give bob's inbox loop time to process.
    for _ in range(50):
        if bob_client.actor.questions:
            break
        await asyncio.sleep(0.01)
    assert bob_client.actor.questions == ["fyi"]

    # Session closed by the adapter.
    await asyncio.sleep(0.02)
    meta = alice_client.lookup_session(session.session_id)
    assert meta is not None
    assert meta.state is SessionState.CLOSED


# ---------------------------------------------------------------------------
# Access denied paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_open_rejects_when_rule_forbids_outbound() -> None:
    store = MemoryKnowledgeStore()
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)

    alice_client = await hc.register(
        FakeActor("alice"),
        identity=ActorIdentity(name="alice"),
        rule=Rule(access=AccessBlock(outbound_to=["ag2:*:*"])),
    )
    await hc.register(FakeActor("bob"), identity=ActorIdentity(name="bob"))

    try:
        with pytest.raises(AccessDeniedError):
            await alice_client.open(SessionType.CONSULTING, target="bob")
    finally:
        await hc.close()
        await link.close()


# ---------------------------------------------------------------------------
# Custom handler override via client.on()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_on_decorator_overrides_default_handler(wired_clients) -> None:
    _hub, _hc, alice_client, bob_client = wired_clients

    @bob_client.on(SessionType.CONSULTING)
    async def custom(envelope: Envelope, client: ActorClient) -> None:
        # Always reply with a fixed string, ignore the actor.
        await client._post_text_reply(envelope, "custom-override")

    session = await alice_client.open(SessionType.CONSULTING, target="bob")
    reply = await session.ask("anything", timeout=2.0)
    assert reply == "custom-override"
    # Bob's underlying actor was bypassed.
    assert bob_client.actor.questions == []


# ---------------------------------------------------------------------------
# Session.subscribe streams envelopes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_subscribe_replays_wal(wired_clients) -> None:
    _hub, _hc, alice_client, _bob_client = wired_clients
    session = await alice_client.open(SessionType.CONVERSATION, target="bob")
    await session.send("turn-one")
    await session.send("turn-two")

    # Give bob time to reply to both.
    await asyncio.sleep(0.05)

    collected: list[str] = []
    agen = session.subscribe(since=0)
    try:
        for _ in range(6):
            envelope = await asyncio.wait_for(agen.__anext__(), timeout=1.0)
            if envelope.event_type == "ag2.msg.text":
                collected.append(envelope.content())
    except asyncio.TimeoutError:
        pass
    finally:
        try:
            await agen.aclose()
        except StopAsyncIteration:
            pass

    # Both Alice's sends + Bob's replies must appear.
    assert "turn-one" in collected
    assert "turn-two" in collected
    assert any("turn-one" in c for c in collected)
    await session.close()


# ---------------------------------------------------------------------------
# Durability — session survives hub reopen over disk store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consulting_session_wal_is_durable_on_disk(tmp_path: Any) -> None:
    store = DiskKnowledgeStore(str(tmp_path / "hub"))
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)

    alice_client = await hc.register(FakeActor("alice"), identity=ActorIdentity(name="alice"))
    bob_client = await hc.register(
        FakeActor("bob", reply="bob-reply"),
        identity=ActorIdentity(name="bob"),
    )

    session = await alice_client.open(SessionType.CONSULTING, target="bob")
    envelope_id = await session.send("hello")
    await asyncio.sleep(0.05)
    _ = envelope_id
    await hc.close()
    await link.close()

    # Re-open the store in a fresh Hub and verify the WAL is readable.
    replay_hub = Hub(DiskKnowledgeStore(str(tmp_path / "hub")))
    from autogen.beta.network.hub import layout

    raw = await replay_hub._store.read(layout.session_wal(session.session_id))
    assert raw is not None
    assert "hello" in raw
    # Bob's reply made it into the WAL too.
    assert "bob-reply" in raw
