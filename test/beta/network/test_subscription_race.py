# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the subscription-replay race fixed in Phase 2.

The Phase 1 implementation registered a subscription *before* reading
the WAL for replay, so an envelope landing between the register step
and the read step would be delivered twice (once via fan-out, once via
replay). Conversely, reading before registering would leave that same
envelope undelivered. Phase 2 closes the window by holding
``Hub._wal_lock`` around both (WAL append + sub snapshot) and
(WAL read + sub register), so every envelope is delivered exactly once.

These tests assert the exactly-once invariant by blasting concurrent
sends at a session that also has a live subscription.
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
from autogen.beta.network.envelope import EV_TEXT


# ---------------------------------------------------------------------------
# Minimal test actors
# ---------------------------------------------------------------------------


class _EchoActor:
    def __init__(self, name: str) -> None:
        self.name = name

    async def ask(self, content: str):
        class _R:
            def __init__(self, body: str) -> None:
                self.body = body

        return _R(f"echo:{content}")


class _QuietActor:
    """Never replies — lets us drive the session externally."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def ask(self, content: str):  # pragma: no cover — shouldn't fire
        class _R:
            def __init__(self, body: str) -> None:
                self.body = body

        return _R("")


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


async def _noop_handler(envelope: Envelope, client: ActorClient) -> None:
    """Discards incoming envelopes without posting a reply.

    We use this on Alice in tests that drive the WAL from the outside so
    the default ``handle_conversation`` handler doesn't auto-echo back
    into the session and change the byte offsets we're measuring.
    """

    return None


async def _spin_up(
    n_bobs: int,
) -> tuple[Hub, HubClient, LocalLink, ActorClient, list[ActorClient]]:
    store = MemoryKnowledgeStore()
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)
    alice = await hc.register(_QuietActor("alice"), identity=ActorIdentity(name="alice"))
    # Override Alice's default conversation handler so she doesn't echo.
    alice.on("conversation")(_noop_handler)
    bobs: list[ActorClient] = []
    for i in range(n_bobs):
        name = f"bob{i}"
        bobs.append(
            await hc.register(_EchoActor(name), identity=ActorIdentity(name=name))
        )
        bobs[-1].on("conversation")(_noop_handler)
    return hub, hc, link, alice, bobs


# ---------------------------------------------------------------------------
# Exactly-once under racing publishers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_during_concurrent_sends_delivers_exactly_once() -> None:
    """Open a subscription while envelopes are flying in.

    Uses a ``conversation`` session (bidirectional, no auto-close) so we
    can pump many envelopes before subscribing. The invariant to verify:
    every envelope posted between ``subscribe`` and a quiet period is
    delivered to the subscriber exactly once, regardless of whether it
    landed during or after the WAL snapshot.
    """

    hub, hc, link, alice, bobs = await _spin_up(n_bobs=1)
    (bob,) = bobs
    try:
        session = await alice.open(SessionType.CONVERSATION, target="bob0")

        # Fire 20 envelopes from Bob's side concurrently with Alice
        # opening a subscription. Some envelopes will land before the
        # snapshot, some during, some after — exactly-once must hold.
        async def producer() -> list[str]:
            ids: list[str] = []
            for i in range(20):
                env = Envelope.text(
                    session_id=session.session_id,
                    sender_id=bob.actor_id,
                    content=f"msg{i}",
                    recipient_id=alice.actor_id,
                )
                eid, _ = await bob._send_envelope(env)
                ids.append(eid)
                # Tiny yield so producer and subscriber actually interleave.
                await asyncio.sleep(0)
            return ids

        async def subscriber(received: list[Envelope]) -> None:
            queue = await alice._open_subscription(session_id=session.session_id)
            try:
                # Collect envelopes until we've been quiet for 50 ms.
                quiet_start: float | None = None
                loop = asyncio.get_event_loop()
                while True:
                    try:
                        env = await asyncio.wait_for(queue.get(), timeout=0.05)
                    except asyncio.TimeoutError:
                        if quiet_start is None:
                            quiet_start = loop.time()
                        if loop.time() - quiet_start > 0.05:
                            return
                        continue
                    quiet_start = None
                    received.append(env)
            finally:
                await alice._close_subscription(queue)

        received: list[Envelope] = []
        prod_task = asyncio.create_task(producer())
        sub_task = asyncio.create_task(subscriber(received))
        produced_ids = await prod_task
        await sub_task

        # Collect just the user text envelopes from the subscriber.
        text_envs = [e for e in received if e.event_type == EV_TEXT]
        received_ids = [e.envelope_id for e in text_envs]

        # 1. Every produced envelope is received.
        assert set(produced_ids).issubset(set(received_ids)), (
            f"missed envelopes: {set(produced_ids) - set(received_ids)}"
        )
        # 2. No envelope is received twice.
        assert len(received_ids) == len(set(received_ids)), (
            f"duplicate deliveries: {[x for x in received_ids if received_ids.count(x) > 1]}"
        )
        # 3. The final delivered count equals the produced count
        # (plus possibly some handshake envelopes filtered out above).
        assert len([e for e in text_envs if e.sender_id == bob.actor_id]) == 20
    finally:
        await hc.close()
        await link.close()


@pytest.mark.asyncio
async def test_subscribe_with_since_cursor_replays_only_after_cursor() -> None:
    """A subscription with ``since=N`` must not re-deliver pre-N envelopes."""

    hub, hc, link, alice, bobs = await _spin_up(n_bobs=1)
    (bob,) = bobs
    try:
        session = await alice.open(SessionType.CONVERSATION, target="bob0")

        # First 3 envelopes before cursor.
        for i in range(3):
            env = Envelope.text(
                session_id=session.session_id,
                sender_id=bob.actor_id,
                content=f"pre{i}",
                recipient_id=alice.actor_id,
            )
            _, cursor = await bob._send_envelope(env)

        # Second 3 envelopes after cursor.
        post_ids: list[str] = []
        for i in range(3):
            env = Envelope.text(
                session_id=session.session_id,
                sender_id=bob.actor_id,
                content=f"post{i}",
                recipient_id=alice.actor_id,
            )
            eid, _ = await bob._send_envelope(env)
            post_ids.append(eid)

        queue = await alice._open_subscription(
            session_id=session.session_id, since=cursor
        )
        try:
            # Drain everything the hub feeds us.
            received: list[Envelope] = []
            for _ in range(3):
                env = await asyncio.wait_for(queue.get(), timeout=0.5)
                received.append(env)
            # No more — subscriber is quiet.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(queue.get(), timeout=0.05)
        finally:
            await alice._close_subscription(queue)

        text_ids = [e.envelope_id for e in received if e.event_type == EV_TEXT]
        assert set(text_ids) == set(post_ids)
    finally:
        await hc.close()
        await link.close()


@pytest.mark.asyncio
async def test_session_ask_still_works_under_the_fixed_subscription_path() -> None:
    """Smoke: the high-level ``Session.ask`` flow still returns the reply.

    This is the Phase 1 integration test re-asserted against the Phase 2
    locked WAL path so we catch regressions to the common case.
    """

    hub, hc, link, alice, bobs = await _spin_up(n_bobs=1)
    try:
        session = await alice.open(SessionType.CONSULTING, target="bob0")
        reply = await session.ask("hi", timeout=2.0)
        assert reply == "echo:hi"
    finally:
        await hc.close()
        await link.close()
