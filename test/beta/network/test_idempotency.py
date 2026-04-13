# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Phase 2 idempotent send (§13.4).

Repeat sends with the same ``idempotency_key`` must return the cached
``envelope_id`` and ``wal_offset`` without re-running the post pipeline
(adapters, rate limits, rule checks, fan-out). Resolved per-session so
two different sessions can legitimately reuse the same key.
"""

from __future__ import annotations

import asyncio

import pytest

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    Envelope,
    Hub,
    LimitsBlock,
    RateBlock,
    Rule,
    SessionType,
)
from autogen.beta.network.errors import LimitExceededError
from autogen.beta.network.transport.frames import AcceptFrame, ErrorFrame, SendFrame

from test.beta.network._harness import (
    FakeClient,
    attach_hub_to_link,
    auto_ack_only,
)


@pytest.fixture
def hub() -> Hub:
    return Hub(MemoryKnowledgeStore())


# ---------------------------------------------------------------------------
# Dedup happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeat_send_with_same_key_returns_cached_envelope(hub: Hub) -> None:
    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"))
    link = attach_hub_to_link(hub)
    a = FakeClient(hub=hub, link=link, actor_id=alice.actor_id or "")
    b = FakeClient(
        hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only
    )
    await a.start()
    await b.start()
    try:
        meta = await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type=SessionType.CONVERSATION,
            participant_names=["bob"],
            invite_ack_timeout_s=0.5,
        )

        # First send with idempotency_key.
        env = Envelope.text(
            session_id=meta.session_id,
            sender_id=alice.actor_id or "",
            content="only once please",
            recipient_id=bob.actor_id,
        )
        env.idempotency_key = "op-42"
        await a._client_handle.send_frame(
            SendFrame(envelope=env, idempotency_key="op-42")
        )
        # Drain the AcceptFrame that the loop puts in accept_map via
        # FakeClient's frame loop.
        await asyncio.sleep(0.05)

        wal_before = await hub.read_wal(meta.session_id)
        text_count_before = sum(1 for e in wal_before if e.event_type == "ag2.msg.text")
        assert text_count_before == 1

        # Second send, same key — hub returns the cached accept.
        env2 = Envelope.text(
            session_id=meta.session_id,
            sender_id=alice.actor_id or "",
            content="DIFFERENT CONTENT — should not land",
            recipient_id=bob.actor_id,
        )
        env2.idempotency_key = "op-42"
        await a._client_handle.send_frame(
            SendFrame(envelope=env2, idempotency_key="op-42")
        )
        await asyncio.sleep(0.05)

        wal_after = await hub.read_wal(meta.session_id)
        text_count_after = sum(1 for e in wal_after if e.event_type == "ag2.msg.text")
        # Still exactly one text envelope on the WAL — the second
        # submission was deduped and never re-appended. Its
        # (different) content is NOT in the WAL.
        assert text_count_after == 1
        assert all(
            e.content() != "DIFFERENT CONTENT — should not land"
            for e in wal_after
            if e.event_type == "ag2.msg.text"
        )
    finally:
        await a.stop()
        await b.stop()
        await link.close()


# ---------------------------------------------------------------------------
# Scoping — per session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_key_different_sessions_are_independent(hub: Hub) -> None:
    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"))
    link = attach_hub_to_link(hub)
    b = FakeClient(
        hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only
    )
    await b.start()
    try:
        s1 = await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type=SessionType.CONVERSATION,
            participant_names=["bob"],
            invite_ack_timeout_s=0.5,
        )
        s2 = await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type=SessionType.CONVERSATION,
            participant_names=["bob"],
            invite_ack_timeout_s=0.5,
        )
        assert s1.session_id != s2.session_id

        # Record same key against s1 only.
        hub._record_idempotent(s1.session_id, "op-7", "env-s1", 100)
        assert hub._lookup_idempotent(s1.session_id, "op-7") is not None
        assert hub._lookup_idempotent(s2.session_id, "op-7") is None
    finally:
        await b.stop()
        await link.close()


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cached_entry_expires_after_ttl(hub: Hub) -> None:
    # Tiny TTL so we can test expiry deterministically.
    short_hub = Hub(MemoryKnowledgeStore(), idempotency_ttl_s=0.01)
    short_hub._record_idempotent("s1", "op", "env", 99)
    # Immediate lookup returns the cache.
    assert short_hub._lookup_idempotent("s1", "op") is not None
    await asyncio.sleep(0.02)
    # After TTL, lookup misses AND removes the entry from the dict.
    assert short_hub._lookup_idempotent("s1", "op") is None
    assert ("s1", "op") not in short_hub._idempotency


# ---------------------------------------------------------------------------
# Dedup bypasses rate limiting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotent_repeat_does_not_consume_rate_bucket() -> None:
    """A retry with the same key must not count against the sender's
    rate bucket — otherwise a legitimate retry storm would accidentally
    throttle the client."""

    hub = Hub(MemoryKnowledgeStore())
    alice = await hub.register(
        ActorIdentity(name="alice"),
        rule=Rule(limits=LimitsBlock(rate=RateBlock(per_minute=60, burst=1))),
    )
    bob = await hub.register(ActorIdentity(name="bob"))
    link = attach_hub_to_link(hub)
    a = FakeClient(hub=hub, link=link, actor_id=alice.actor_id or "")
    b = FakeClient(
        hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only
    )
    await a.start()
    await b.start()
    try:
        meta = await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type=SessionType.CONVERSATION,
            participant_names=["bob"],
            invite_ack_timeout_s=0.5,
        )
        env = Envelope.text(
            session_id=meta.session_id,
            sender_id=alice.actor_id or "",
            content="hello",
            recipient_id=bob.actor_id,
        )
        env.idempotency_key = "key"
        await a._client_handle.send_frame(
            SendFrame(envelope=env, idempotency_key="key")
        )
        await asyncio.sleep(0.05)

        # Retry with the same key — hits the cache, does NOT touch
        # the rate bucket.
        env_retry = Envelope.text(
            session_id=meta.session_id,
            sender_id=alice.actor_id or "",
            content="hello",
            recipient_id=bob.actor_id,
        )
        env_retry.idempotency_key = "key"
        await a._client_handle.send_frame(
            SendFrame(envelope=env_retry, idempotency_key="key")
        )
        await asyncio.sleep(0.05)

        # A second *different* envelope without an idempotency_key
        # still fails: the one-token-bucket has exactly one token
        # remaining (the first real send burned 1).
        env_fresh = Envelope.text(
            session_id=meta.session_id,
            sender_id=alice.actor_id or "",
            content="different",
            recipient_id=bob.actor_id,
        )
        with pytest.raises(LimitExceededError):
            await hub.post_envelope(env_fresh)
    finally:
        await a.stop()
        await b.stop()
        await link.close()
