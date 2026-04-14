# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Phase 3b ``drop_oldest`` / ``drop_newest`` inbox overflow policies.

Phase 3a shipped ``reject`` + ``spool`` as real overflow modes, and stored
``drop_oldest`` / ``drop_newest`` verbatim in the rule while falling back
to reject semantics at enforcement time. Phase 3b wires the real
behavior:

* ``drop_oldest`` — when pending/ is at ``max_pending``, the hub evicts
  the oldest file (UUID7 envelope ids sort lexicographically = time
  order) before delivering the new envelope. Pending counter stays at
  the cap.
* ``drop_newest`` — the incoming envelope is silently discarded
  (no WAL mutation impact — the WAL records the post, only the
  recipient's inbox drops), the pending counter stays at the cap, and
  the handler never sees a notify.

Both modes emit audit-log entries so operators can observe drops — the
audit writer is task #4, but the call site is wired today.
"""

from __future__ import annotations

import asyncio

import pytest

from autogen.beta.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    Hub,
    HubClient,
    InboxBlock,
    LimitsBlock,
    LocalLink,
    Rule,
    SessionType,
)
from autogen.beta.network.hub import layout


# ---------------------------------------------------------------------------
# Helpers mirroring test_inbox_structured
# ---------------------------------------------------------------------------


class _Echo:
    def __init__(self, name: str) -> None:
        self.name = name

    async def ask(self, content: str, **_kwargs):
        class _R:
            def __init__(self, body: str) -> None:
                self.body = body

        return _R(f"echo:{content}")


async def _spin_two_actors(
    *,
    bob_rule: Rule | None = None,
    alice_rule: Rule | None = None,
    store=None,
):
    hub = Hub(store or MemoryKnowledgeStore())
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)

    alice = await hc.register(
        _Echo("alice"),
        identity=ActorIdentity(name="alice"),
        rule=alice_rule,
    )
    bob = await hc.register(
        _Echo("bob"),
        identity=ActorIdentity(name="bob"),
        rule=bob_rule,
    )
    return hub, hc, link, alice, bob


async def _block_forever(_envelope, _client):
    await asyncio.Event().wait()


# ---------------------------------------------------------------------------
# drop_oldest — evicts the oldest pending envelope, keeps counter at cap
# ---------------------------------------------------------------------------


class TestDropOldest:
    @pytest.mark.asyncio
    async def test_at_capacity_evicts_oldest(self) -> None:
        bob_rule = Rule(
            limits=LimitsBlock(
                inbox=InboxBlock(max_pending=2, overflow="drop_oldest")
            )
        )
        hub, hc, link, alice, bob = await _spin_two_actors(bob_rule=bob_rule)
        try:
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            first_id = await session.send("one")
            await asyncio.sleep(0.01)
            second_id = await session.send("two")
            await asyncio.sleep(0.01)
            assert hub._inbox_pending[bob.actor_id] == 2

            third_id = await session.send("three")
            await asyncio.sleep(0.01)

            # Counter still capped at max_pending.
            assert hub._inbox_pending[bob.actor_id] == 2

            # The oldest (first) file is gone.
            assert await hub._store.read(
                layout.actor_inbox_pending(bob.actor_id, first_id)
            ) is None
            # The two newer envelopes remain in pending/.
            assert await hub._store.read(
                layout.actor_inbox_pending(bob.actor_id, second_id)
            ) is not None
            assert await hub._store.read(
                layout.actor_inbox_pending(bob.actor_id, third_id)
            ) is not None
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_below_capacity_no_eviction(self) -> None:
        """drop_oldest only kicks in when pending is AT the cap."""

        bob_rule = Rule(
            limits=LimitsBlock(
                inbox=InboxBlock(max_pending=5, overflow="drop_oldest")
            )
        )
        hub, hc, link, alice, bob = await _spin_two_actors(bob_rule=bob_rule)
        try:
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            for i in range(3):
                await session.send(f"msg-{i}")
                await asyncio.sleep(0.005)

            assert hub._inbox_pending[bob.actor_id] == 3
            # All three files still present.
            pending = await hub._store.list(
                layout.actor_inbox_pending_dir(bob.actor_id) + "/"
            )
            assert len(pending) == 3
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_multiple_evictions_cycle(self) -> None:
        """Under sustained pressure, each new send evicts the oldest."""

        bob_rule = Rule(
            limits=LimitsBlock(
                inbox=InboxBlock(max_pending=2, overflow="drop_oldest")
            )
        )
        hub, hc, link, alice, bob = await _spin_two_actors(bob_rule=bob_rule)
        try:
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            ids = []
            for i in range(5):
                ids.append(await session.send(f"msg-{i}"))
                await asyncio.sleep(0.01)

            assert hub._inbox_pending[bob.actor_id] == 2
            # Only the last two remain.
            assert await hub._store.read(
                layout.actor_inbox_pending(bob.actor_id, ids[-1])
            ) is not None
            assert await hub._store.read(
                layout.actor_inbox_pending(bob.actor_id, ids[-2])
            ) is not None
            # The earlier three are gone.
            for evicted in ids[:3]:
                assert await hub._store.read(
                    layout.actor_inbox_pending(bob.actor_id, evicted)
                ) is None
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_new_envelope_delivered_after_eviction(self) -> None:
        """After eviction, the incoming envelope still reaches the handler."""

        from autogen.beta.network.envelope import Envelope

        bob_rule = Rule(
            limits=LimitsBlock(
                inbox=InboxBlock(max_pending=1, overflow="drop_oldest")
            )
        )
        hub, hc, link, alice, bob = await _spin_two_actors(bob_rule=bob_rule)
        try:
            seen: list[str] = []

            async def record(envelope, client):
                seen.append(envelope.content())
                await asyncio.Event().wait()

            bob.on("conversation")(record)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            await session.send("one")
            await asyncio.sleep(0.03)
            assert "one" in seen

            await session.send("two")
            await asyncio.sleep(0.03)
            # Bob's first handler is still blocked, but the second
            # envelope still landed in pending/ and was notified.
            # What we care about: "two" is in pending/, not dropped.
            pending_files = await hub._store.list(
                layout.actor_inbox_pending_dir(bob.actor_id) + "/"
            )
            assert len(pending_files) == 1
            # The surviving file corresponds to "two".
            entry_raw = await hub._store.read(
                f"{layout.actor_inbox_pending_dir(bob.actor_id)}/{pending_files[0]}"
            )
            entry = Envelope.from_json(entry_raw)
            assert entry.content() == "two"
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_disk_store_cycle(self, tmp_path) -> None:
        """drop_oldest works on DiskKnowledgeStore (real filesystem)."""

        store = DiskKnowledgeStore(str(tmp_path / "hub"))
        bob_rule = Rule(
            limits=LimitsBlock(
                inbox=InboxBlock(max_pending=2, overflow="drop_oldest")
            )
        )
        hub, hc, link, alice, bob = await _spin_two_actors(
            bob_rule=bob_rule, store=store
        )
        try:
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            sent_ids = []
            for i in range(4):
                sent_ids.append(await session.send(f"disk-{i}"))
                await asyncio.sleep(0.01)

            assert hub._inbox_pending[bob.actor_id] == 2
            # Oldest two evicted.
            assert await store.read(
                layout.actor_inbox_pending(bob.actor_id, sent_ids[0])
            ) is None
            assert await store.read(
                layout.actor_inbox_pending(bob.actor_id, sent_ids[1])
            ) is None
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_hydrate_after_eviction_round_trip(self, tmp_path) -> None:
        """After eviction, hydrate rebuilds the correct counter."""

        root = tmp_path / "hub"
        store = DiskKnowledgeStore(str(root))
        bob_rule = Rule(
            limits=LimitsBlock(
                inbox=InboxBlock(max_pending=2, overflow="drop_oldest")
            )
        )
        hub, hc, link, alice, bob = await _spin_two_actors(
            bob_rule=bob_rule, store=store
        )
        bob_id = bob.actor_id

        try:
            bob.on("conversation")(_block_forever)
            session = await alice.open(SessionType.CONVERSATION, target="bob")
            for i in range(5):
                await session.send(f"pre-hydrate-{i}")
                await asyncio.sleep(0.01)
            assert hub._inbox_pending[bob_id] == 2
        finally:
            await hc.close()
            await link.close()

        # Reopen against the same store; hydrate must find 2 pending.
        store2 = DiskKnowledgeStore(str(root))
        hub2 = await Hub.open(store2)
        assert hub2._inbox_pending.get(bob_id, 0) == 2


# ---------------------------------------------------------------------------
# drop_newest — incoming envelope silently discarded
# ---------------------------------------------------------------------------


class TestDropNewest:
    @pytest.mark.asyncio
    async def test_at_capacity_drops_new(self) -> None:
        bob_rule = Rule(
            limits=LimitsBlock(
                inbox=InboxBlock(max_pending=2, overflow="drop_newest")
            )
        )
        hub, hc, link, alice, bob = await _spin_two_actors(bob_rule=bob_rule)
        try:
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            first_id = await session.send("one")
            await asyncio.sleep(0.01)
            second_id = await session.send("two")
            await asyncio.sleep(0.01)
            assert hub._inbox_pending[bob.actor_id] == 2

            third_id = await session.send("three")
            await asyncio.sleep(0.01)

            # Counter stays at cap.
            assert hub._inbox_pending[bob.actor_id] == 2
            # The new envelope is NOT in pending/.
            assert await hub._store.read(
                layout.actor_inbox_pending(bob.actor_id, third_id)
            ) is None
            # The older two are still present.
            assert await hub._store.read(
                layout.actor_inbox_pending(bob.actor_id, first_id)
            ) is not None
            assert await hub._store.read(
                layout.actor_inbox_pending(bob.actor_id, second_id)
            ) is not None
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_no_raise_on_drop(self) -> None:
        """drop_newest is silent — ``send`` does not raise ``InboxFullError``."""

        bob_rule = Rule(
            limits=LimitsBlock(
                inbox=InboxBlock(max_pending=1, overflow="drop_newest")
            )
        )
        hub, hc, link, alice, bob = await _spin_two_actors(bob_rule=bob_rule)
        try:
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            await session.send("one")
            await asyncio.sleep(0.01)

            # This must not raise — drop_newest is silent.
            dropped_id = await session.send("two")
            await asyncio.sleep(0.01)

            # But nothing landed in the inbox for the drop.
            assert await hub._store.read(
                layout.actor_inbox_pending(bob.actor_id, dropped_id)
            ) is None
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_wal_still_records_the_post(self) -> None:
        """drop_newest drops only at the inbox — the WAL still records."""

        bob_rule = Rule(
            limits=LimitsBlock(
                inbox=InboxBlock(max_pending=1, overflow="drop_newest")
            )
        )
        hub, hc, link, alice, bob = await _spin_two_actors(bob_rule=bob_rule)
        try:
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            await session.send("one")
            await asyncio.sleep(0.01)
            await session.send("two-dropped")
            await asyncio.sleep(0.01)

            envelopes = await hub._read_user_envelopes(session.session_id)
            contents = [e.content() for e in envelopes]
            # Both posts appear in the WAL — the drop is at the inbox,
            # not at post_envelope.
            assert "one" in contents
            assert "two-dropped" in contents
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_notify_not_pushed_for_dropped_envelope(self) -> None:
        """The handler never sees a dropped envelope."""

        bob_rule = Rule(
            limits=LimitsBlock(
                inbox=InboxBlock(max_pending=1, overflow="drop_newest")
            )
        )
        hub, hc, link, alice, bob = await _spin_two_actors(bob_rule=bob_rule)
        try:
            seen: list[str] = []

            async def record(envelope, client):
                seen.append(envelope.content())
                await asyncio.Event().wait()

            bob.on("conversation")(record)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            await session.send("one")
            await asyncio.sleep(0.03)
            # One is pinned in the handler; two will be dropped.
            await session.send("two-dropped")
            await asyncio.sleep(0.03)

            assert seen == ["one"]
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_frees_slot_after_ack_accepts_new(self) -> None:
        """After ack, drop_newest sends are accepted again."""

        from autogen.beta.network.transport.frames import ReceiptFrame

        bob_rule = Rule(
            limits=LimitsBlock(
                inbox=InboxBlock(max_pending=1, overflow="drop_newest")
            )
        )
        hub, hc, link, alice, bob = await _spin_two_actors(bob_rule=bob_rule)
        try:
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            first_id = await session.send("one")
            await asyncio.sleep(0.01)
            await session.send("two-dropped")
            await asyncio.sleep(0.01)
            assert hub._inbox_pending[bob.actor_id] == 1

            # Clear the pending slot manually (simulates the handler
            # eventually running and acking).
            endpoint = hub._endpoints[bob.actor_id]
            await hub._handle_receipt(
                endpoint,
                ReceiptFrame(envelope_id=first_id, status="ack"),
                bob.actor_id,
            )
            assert hub._inbox_pending[bob.actor_id] == 0

            third_id = await session.send("three")
            await asyncio.sleep(0.01)
            assert await hub._store.read(
                layout.actor_inbox_pending(bob.actor_id, third_id)
            ) is not None
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# Fan-out interaction: drop policies are per-recipient
# ---------------------------------------------------------------------------


class TestDropPoliciesFanout:
    @pytest.mark.asyncio
    async def test_per_recipient_policy(self) -> None:
        """Broadcast where Bob uses drop_oldest, Carol uses reject.

        Bob's inbox is at cap and will evict; Carol's is not yet at cap
        and accepts normally. The atomic-preflight rule from Phase 3a
        only blocks broadcasts on ``reject``-mode recipients that are
        already full — which Carol isn't here — so the whole broadcast
        proceeds and Bob's oldest gets evicted.
        """

        hub = Hub(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            bob = await hc.register(
                _Echo("bob"),
                identity=ActorIdentity(name="bob"),
                rule=Rule(
                    limits=LimitsBlock(
                        inbox=InboxBlock(max_pending=1, overflow="drop_oldest")
                    )
                ),
            )
            carol = await hc.register(
                _Echo("carol"),
                identity=ActorIdentity(name="carol"),
                rule=Rule(
                    limits=LimitsBlock(
                        inbox=InboxBlock(max_pending=5, overflow="reject")
                    )
                ),
            )
            bob.on("broadcast")(_block_forever)
            carol.on("broadcast")(_block_forever)
            alice.on("broadcast")(_block_forever)

            session = await alice.open(
                SessionType.BROADCAST, target=["bob", "carol"]
            )
            await session.send("first")
            await asyncio.sleep(0.02)
            assert hub._inbox_pending[bob.actor_id] == 1
            assert hub._inbox_pending[carol.actor_id] == 1

            # Bob is at his cap; drop_oldest evicts his "first" and
            # delivers the new envelope. Carol just stacks normally.
            await session.send("second")
            await asyncio.sleep(0.02)

            assert hub._inbox_pending[bob.actor_id] == 1
            assert hub._inbox_pending[carol.actor_id] == 2
        finally:
            await hc.close()
            await link.close()
