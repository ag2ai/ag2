# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Phase 2 ChunkFrame streaming.

Chunks flow bidirectionally through the Link: a sender emits
``ChunkFrame`` as tokens arrive from the model, the hub routes them to
the recipient's endpoint, and the recipient aggregates them via
``Session.iter_chunks``. The final envelope (complete text) still goes
through the regular SendFrame → ``post_envelope`` path, so the WAL stays
clean — chunks are transient frame-level relays in Phase 2.

Scope:

* Frame-level round-trip through :class:`LocalLink` (encoder + decoder
  stay consistent for the new fields).
* End-to-end: Alice sends, Bob streams chunks while producing the
  reply, Alice's ``iter_chunks`` yields each chunk.
* Fan-out: a broadcast session fans chunks to every non-sender
  participant.
* Validation: chunks from a non-participant are rejected with an
  :class:`ErrorFrame`.
"""

from __future__ import annotations

import asyncio

import pytest

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    ActorClient,
    ActorIdentity,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    SessionType,
)
from autogen.beta.network.transport.frames import (
    ChunkFrame,
    decode_frame,
    encode_frame,
)


# ---------------------------------------------------------------------------
# Frame-level round-trip
# ---------------------------------------------------------------------------


def test_chunk_frame_round_trips_new_fields() -> None:
    original = ChunkFrame(
        envelope_id="env-1",
        chunk_index=3,
        content="partial",
        session_id="sess-1",
        sender_id="bob",
        recipient_id="alice",
        final=True,
    )
    restored = decode_frame(encode_frame(original))
    assert isinstance(restored, ChunkFrame)
    assert restored.envelope_id == "env-1"
    assert restored.chunk_index == 3
    assert restored.content == "partial"
    assert restored.session_id == "sess-1"
    assert restored.sender_id == "bob"
    assert restored.recipient_id == "alice"
    assert restored.final is True


def test_chunk_frame_round_trips_default_fields() -> None:
    original = ChunkFrame(envelope_id="e", chunk_index=0, content="hi")
    restored = decode_frame(encode_frame(original))
    assert isinstance(restored, ChunkFrame)
    assert restored.session_id is None
    assert restored.sender_id is None
    assert restored.recipient_id is None
    assert restored.final is False


# ---------------------------------------------------------------------------
# End-to-end hub streaming
# ---------------------------------------------------------------------------


class _NoopActor:
    def __init__(self, name: str) -> None:
        self.name = name

    async def ask(self, content: str, **kwargs):
        class _R:
            def __init__(self, body: str) -> None:
                self.body = body

        return _R(f"done:{content}")


async def _spin(n_bobs: int = 1) -> tuple[Hub, HubClient, LocalLink, ActorClient, list[ActorClient]]:
    hub = Hub(MemoryKnowledgeStore())
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)
    alice = await hc.register(_NoopActor("alice"), identity=ActorIdentity(name="alice"))
    bobs: list[ActorClient] = []
    for i in range(n_bobs):
        name = f"bob{i}"
        bobs.append(
            await hc.register(_NoopActor(name), identity=ActorIdentity(name=name))
        )
    return hub, hc, link, alice, bobs


async def _quiet_handler(envelope: Envelope, client: ActorClient) -> None:
    """Never replies. Used when a test drives chunks manually."""

    return None


@pytest.mark.asyncio
async def test_chunks_are_delivered_to_recipient_in_order() -> None:
    hub, hc, link, alice, bobs = await _spin(n_bobs=1)
    (bob,) = bobs
    bob.on("conversation")(_quiet_handler)
    alice.on("conversation")(_quiet_handler)
    try:
        alice_session = await alice.open(SessionType.CONVERSATION, target="bob0")

        # Bob's ActorClient is in the same session; construct a Session
        # handle for his side pointing at the same metadata so he can
        # emit chunks addressed back to Alice.
        from autogen.beta.network.client.session import Session

        bob_session = Session(client=bob, metadata=alice_session.metadata)

        # Register the chunk queue first so we don't race.
        consumed: list[str] = []

        async def consume() -> None:
            async for chunk in alice_session.iter_chunks("reply-1"):
                consumed.append(chunk)

        consume_task = asyncio.create_task(consume())
        await asyncio.sleep(0)  # let the consumer register its queue

        for i, token in enumerate(["hel", "lo ", "world"]):
            await bob_session.send_chunk(
                envelope_id="reply-1",
                chunk_index=i,
                content=token,
                recipient_id=alice.actor_id,
                final=(i == 2),
            )
        await asyncio.wait_for(consume_task, timeout=1.0)
        assert consumed == ["hel", "lo ", "world"]
    finally:
        await hc.close()
        await link.close()


@pytest.mark.asyncio
async def test_chunk_broadcast_fans_out_to_every_non_sender_participant() -> None:
    hub, hc, link, alice, bobs = await _spin(n_bobs=2)
    bob, carol = bobs
    bob.on("broadcast")(_quiet_handler)
    carol.on("broadcast")(_quiet_handler)
    alice.on("broadcast")(_quiet_handler)
    try:
        alice_session = await alice.open(
            SessionType.BROADCAST, target=["bob0", "bob1"]
        )

        from autogen.beta.network.client.session import Session

        bob_session = Session(client=bob, metadata=alice_session.metadata)
        carol_session = Session(client=carol, metadata=alice_session.metadata)

        bob_consumed: list[str] = []
        carol_consumed: list[str] = []

        async def consume(session: Session, sink: list[str]) -> None:
            async for chunk in session.iter_chunks("stream-1"):
                sink.append(chunk)

        bob_task = asyncio.create_task(consume(bob_session, bob_consumed))
        carol_task = asyncio.create_task(consume(carol_session, carol_consumed))
        await asyncio.sleep(0)  # let consumers register queues

        # Alice streams to everybody (recipient_id=None → fan-out).
        for i, chunk in enumerate(["one", "two", "three"]):
            await alice_session.send_chunk(
                envelope_id="stream-1",
                chunk_index=i,
                content=chunk,
                recipient_id=None,
                final=(i == 2),
            )
        await asyncio.wait_for(bob_task, timeout=1.0)
        await asyncio.wait_for(carol_task, timeout=1.0)
        assert bob_consumed == ["one", "two", "three"]
        assert carol_consumed == ["one", "two", "three"]
    finally:
        await hc.close()
        await link.close()


@pytest.mark.asyncio
async def test_chunk_sender_must_be_session_participant() -> None:
    hub, hc, link, alice, bobs = await _spin(n_bobs=1)
    (bob,) = bobs
    # Register a third actor NOT in the session and try to inject a chunk.
    mallory = await hc.register(
        _NoopActor("mallory"), identity=ActorIdentity(name="mallory")
    )
    alice.on("conversation")(_quiet_handler)
    bob.on("conversation")(_quiet_handler)
    try:
        meta = await alice.open(SessionType.CONVERSATION, target="bob0")

        from autogen.beta.network.client.session import Session

        mallory_session = Session(client=mallory, metadata=meta)
        # Mallory's chunk must be rejected — she's not in the session.
        await mallory_session.send_chunk(
            envelope_id="sneak",
            chunk_index=0,
            content="injected",
            recipient_id=alice.actor_id,
            final=True,
        )

        # Alice's chunk queue must stay empty.
        queue = alice._chunk_queue_for("sneak")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.get(), timeout=0.1)
    finally:
        await hc.close()
        await link.close()


@pytest.mark.asyncio
async def test_chunks_landing_before_queue_registration_are_buffered() -> None:
    """If the sender is fast enough to emit before the recipient opens
    its chunk queue, the ActorClient must still keep the frames around
    in a queue created on first arrival. Chunks are dropped only when
    the recipient calls ``_discard_chunk_queue`` via
    ``iter_chunks.__aexit__``.
    """

    hub, hc, link, alice, bobs = await _spin(n_bobs=1)
    (bob,) = bobs
    alice.on("conversation")(_quiet_handler)
    bob.on("conversation")(_quiet_handler)
    try:
        meta = await alice.open(SessionType.CONVERSATION, target="bob0")

        from autogen.beta.network.client.session import Session

        bob_session = Session(client=bob, metadata=meta)

        # Bob streams before Alice has called iter_chunks.
        for i, token in enumerate(["a", "b", "c"]):
            await bob_session.send_chunk(
                envelope_id="pre-reg",
                chunk_index=i,
                content=token,
                recipient_id=alice.actor_id,
                final=(i == 2),
            )
        await asyncio.sleep(0.02)

        # Alice now opens the iterator and drains what's buffered.
        alice_session = Session(client=alice, metadata=meta)
        consumed: list[str] = []
        async for chunk in alice_session.iter_chunks("pre-reg"):
            consumed.append(chunk)
        assert consumed == ["a", "b", "c"]
    finally:
        await hc.close()
        await link.close()
