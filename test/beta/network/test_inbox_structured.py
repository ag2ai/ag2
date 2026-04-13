# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Phase 3a structured inbox layout.

Phase 2 appended every delivered envelope to a flat
``hub/actors/{id}/inbox.jsonl`` log and silently ignored receipt
frames. Phase 3a moves to the §7.1 layout:

* ``hub/actors/{id}/inbox/pending/{envelope_id}.json`` — newly
  delivered, not yet acked.
* ``hub/actors/{id}/inbox/received/{envelope_id}.json`` — the actor
  has durably accepted the envelope (ack receipt).
* ``hub/actors/{id}/inbox/nacks.jsonl`` — structured nack log (one
  line per rejected envelope).
* ``hub/actors/{id}/inbox/overflow/{envelope_id}.json`` — envelopes
  spooled when the pending directory is at ``max_pending``.

This test module exercises the full matrix:

* Deliver → file lands in ``pending/``.
* Ack → file moves ``pending/ → received/``, pending counter
  decrements.
* Nack → file removed from ``pending/``, nack entry appended.
* ``max_pending=reject`` → ``post_envelope`` raises
  ``InboxFullError`` synchronously, no WAL mutation.
* ``max_pending=spool`` → delivery writes to ``overflow/`` without
  bumping the pending counter.
* Ack for an envelope that was already processed (or spooled) is a
  no-op.
* ``Hub.hydrate()`` rebuilds the pending counter from the store.
* ``InboxBlock`` round-trips through rule JSON with every overflow
  mode.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from autogen.beta.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    Envelope,
    Hub,
    HubClient,
    InboxBlock,
    InboxFullError,
    LimitsBlock,
    LocalLink,
    Rule,
    SessionType,
)
from autogen.beta.network.hub import layout


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


async def _spin_two_actors(
    *,
    bob_rule: Rule | None = None,
    alice_rule: Rule | None = None,
) -> tuple[Hub, HubClient, LocalLink, "_Client", "_Client"]:
    hub = Hub(MemoryKnowledgeStore())
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


class _Client:  # just a type placeholder so the annotation above reads
    pass


async def _block_forever(_envelope, _client):
    """Notify handler that never returns.

    Used in pending-count tests: the default ActorClient flow sends a
    ``ReceiptFrame(ack)`` *after* the handler returns, which triggers
    ``Hub._handle_receipt`` and drops the pending counter. A handler
    that blocks indefinitely keeps the envelope in ``pending/`` for
    the duration of the test. On test teardown, ``ActorClient.disconnect``
    cancels the blocked task (Phase 3a Step 2 fixed the Phase 2 debt
    where disconnect awaited tasks without cancelling first).
    """

    await asyncio.Event().wait()


# ---------------------------------------------------------------------------
# Rule serialization — InboxBlock round-trip
# ---------------------------------------------------------------------------


class TestInboxBlock:
    def test_default_is_unlimited_reject(self) -> None:
        block = InboxBlock()
        assert block.max_pending == 0
        assert block.overflow == "reject"

    def test_round_trip_reject(self) -> None:
        block = InboxBlock(max_pending=10, overflow="reject")
        restored = InboxBlock.from_dict(block.to_dict())
        assert restored == block

    def test_round_trip_spool(self) -> None:
        block = InboxBlock(max_pending=5, overflow="spool")
        restored = InboxBlock.from_dict(block.to_dict())
        assert restored == block

    def test_round_trip_drop_policies_accepted(self) -> None:
        """Phase 3b forms must round-trip in Phase 3a rule JSON."""

        oldest = InboxBlock.from_dict({"max_pending": 10, "overflow": "drop_oldest"})
        newest = InboxBlock.from_dict({"max_pending": 10, "overflow": "drop_newest"})
        assert oldest.overflow == "drop_oldest"
        assert newest.overflow == "drop_newest"

    def test_rejects_unknown_overflow_policy(self) -> None:
        with pytest.raises(ValueError, match="inbox.overflow must be one of"):
            InboxBlock.from_dict({"max_pending": 5, "overflow": "explode"})

    def test_rule_round_trip_carries_inbox(self) -> None:
        rule = Rule(
            limits=LimitsBlock(inbox=InboxBlock(max_pending=3, overflow="spool"))
        )
        restored = Rule.from_json(rule.to_json())
        assert restored.limits.inbox.max_pending == 3
        assert restored.limits.inbox.overflow == "spool"


# ---------------------------------------------------------------------------
# Happy path — pending / received files appear in the right places
# ---------------------------------------------------------------------------


class TestStructuredInboxDelivery:
    @pytest.mark.asyncio
    async def test_deliver_lands_in_pending(self) -> None:
        hub, hc, link, alice, bob = await _spin_two_actors()
        try:
            # Bob's handler blocks forever so the envelope stays in
            # pending/ for the duration of the assertion — otherwise
            # the default ack path would race us and move it to
            # received/ before we can observe it.
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            envelope_id = await session.send("hello")

            # Give the hub a tick to write pending/
            await asyncio.sleep(0.01)

            pending = await hub._store.read(
                layout.actor_inbox_pending(bob.actor_id, envelope_id)
            )
            assert pending is not None
            env = Envelope.from_json(pending)
            assert env.envelope_id == envelope_id
            assert env.recipient_id == bob.actor_id
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_ack_moves_pending_to_received(self) -> None:
        hub, hc, link, alice, bob = await _spin_two_actors()
        try:
            reply = await (
                await alice.open(SessionType.CONSULTING, target="bob")
            ).ask("hi", timeout=2.0)
            assert reply.startswith("echo:")

            # After the consulting round-trip, Alice's send was acked
            # by Bob's default handler. That ack drives the hub to
            # move Alice's pending/{id}.json to received/{id}.json.
            await asyncio.sleep(0.05)

            pending_entries = await hub._store.list(
                layout.actor_inbox_pending_dir(bob.actor_id) + "/"
            )
            received_entries = await hub._store.list(
                layout.actor_inbox_received_dir(bob.actor_id) + "/"
            )
            # Bob has received the consulting question, acked it, and
            # the file should now live in received/.
            assert pending_entries == []
            assert len(received_entries) >= 1
            assert all(e.endswith(".json") for e in received_entries)
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_pending_counter_tracks_delivery_and_ack(self) -> None:
        hub, hc, link, alice, bob = await _spin_two_actors()
        try:
            # Block Bob's handler so the auto-ack doesn't race us.
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            await session.send("one")
            await session.send("two")
            await session.send("three")
            await asyncio.sleep(0.02)

            assert hub._inbox_pending[bob.actor_id] == 3
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_pending_counter_decrements_on_ack(self) -> None:
        hub, hc, link, alice, bob = await _spin_two_actors()
        try:
            await (
                await alice.open(SessionType.CONSULTING, target="bob")
            ).ask("hi", timeout=2.0)
            await asyncio.sleep(0.05)
            # Both sides of the consulting exchange have been
            # acked by their default handlers, so pending should be 0
            # on both actors.
            assert hub._inbox_pending[bob.actor_id] == 0
            assert hub._inbox_pending[alice.actor_id] == 0
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# Ack / nack idempotency and edge cases
# ---------------------------------------------------------------------------


class TestReceiptHandling:
    @pytest.mark.asyncio
    async def test_double_ack_is_idempotent(self) -> None:
        """A receipt for an envelope that was already moved is a no-op."""

        from autogen.beta.network.transport.frames import ReceiptFrame

        hub, hc, link, alice, bob = await _spin_two_actors()
        try:
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            envelope_id = await session.send("hi")
            await asyncio.sleep(0.01)

            # Manually post an ack twice — the second one should not
            # drop the counter below zero or raise.
            endpoint = hub._endpoints[bob.actor_id]
            await hub._handle_receipt(
                endpoint,
                ReceiptFrame(envelope_id=envelope_id, status="ack"),
                bob.actor_id,
            )
            await hub._handle_receipt(
                endpoint,
                ReceiptFrame(envelope_id=envelope_id, status="ack"),
                bob.actor_id,
            )
            assert hub._inbox_pending[bob.actor_id] == 0

            pending = await hub._store.read(
                layout.actor_inbox_pending(bob.actor_id, envelope_id)
            )
            received = await hub._store.read(
                layout.actor_inbox_received(bob.actor_id, envelope_id)
            )
            assert pending is None
            assert received is not None
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_nack_writes_log_and_clears_pending(self) -> None:
        from autogen.beta.network.transport.frames import ReceiptFrame

        hub, hc, link, alice, bob = await _spin_two_actors()
        try:
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            envelope_id = await session.send("hello")
            await asyncio.sleep(0.01)

            endpoint = hub._endpoints[bob.actor_id]
            await hub._handle_receipt(
                endpoint,
                ReceiptFrame(
                    envelope_id=envelope_id, status="nack", reason="rule_rejected"
                ),
                bob.actor_id,
            )

            # Pending file gone.
            pending = await hub._store.read(
                layout.actor_inbox_pending(bob.actor_id, envelope_id)
            )
            assert pending is None
            # Nack log has one line with the envelope id + reason.
            nacks_raw = await hub._store.read(
                layout.actor_inbox_nacks(bob.actor_id)
            )
            assert nacks_raw is not None
            lines = [line for line in nacks_raw.strip().split("\n") if line]
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["envelope_id"] == envelope_id
            assert entry["reason"] == "rule_rejected"
            assert hub._inbox_pending[bob.actor_id] == 0
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_ack_for_unknown_envelope_is_noop(self) -> None:
        from autogen.beta.network.transport.frames import ReceiptFrame

        hub, hc, link, alice, bob = await _spin_two_actors()
        try:
            # Nothing to ack — the pending counter stays at 0 and no
            # exception is raised.
            endpoint = hub._endpoints[bob.actor_id]
            await hub._handle_receipt(
                endpoint,
                ReceiptFrame(envelope_id="ghost-envelope-id", status="ack"),
                bob.actor_id,
            )
            assert hub._inbox_pending[bob.actor_id] == 0
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# max_pending — reject policy (preflight blocks post_envelope)
# ---------------------------------------------------------------------------


class TestMaxPendingReject:
    @pytest.mark.asyncio
    async def test_reject_raises_before_wal_append(self) -> None:
        bob_rule = Rule(
            limits=LimitsBlock(inbox=InboxBlock(max_pending=2, overflow="reject"))
        )
        hub, hc, link, alice, bob = await _spin_two_actors(bob_rule=bob_rule)
        try:
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            await session.send("one")
            await session.send("two")
            await asyncio.sleep(0.02)
            assert hub._inbox_pending[bob.actor_id] == 2

            # Third send must raise InboxFullError.
            with pytest.raises(InboxFullError, match="full"):
                await session.send("three")

            # WAL must not contain the third envelope.
            prior = await hub._read_user_envelopes(session.session_id)
            assert len(prior) == 2
            contents = [p.content() for p in prior]
            assert "three" not in contents
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_reject_consumes_rate_bucket_documents_ordering(self) -> None:
        """A rejected post_envelope consumes a rate-bucket token.

        Phase 3a intentionally runs the rate check *before* the inbox
        preflight: rate-limit fairness (one actor can't blast a full
        inbox on a high-cost recipient) matters more than "refund
        rejections." This test documents that ordering — if we ever
        want rejections to be free, we'd have to move the preflight
        ahead of rate consumption.
        """

        from autogen.beta.network.rule import RateBlock

        bob_rule = Rule(
            limits=LimitsBlock(
                inbox=InboxBlock(max_pending=1, overflow="reject"),
            )
        )
        alice_rule = Rule(limits=LimitsBlock(rate=RateBlock(per_minute=60, burst=5)))
        hub, hc, link, alice, bob = await _spin_two_actors(
            bob_rule=bob_rule, alice_rule=alice_rule
        )
        try:
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            tokens_before = hub._rate_limiter._buckets.get(alice.actor_id)
            await session.send("one")
            await asyncio.sleep(0.02)
            with pytest.raises(InboxFullError):
                await session.send("two")

            bucket = hub._rate_limiter._buckets.get(alice.actor_id)
            assert bucket is not None
            # Alice's 5-capacity bucket has taken at least two draws
            # (the accepted "one" and the rejected "two") plus any
            # earlier handshake envelopes. Assert "tokens decreased"
            # rather than an exact count to stay robust to ordering
            # of invite acks.
            assert bucket.tokens < bucket.capacity
            assert tokens_before is None or bucket.tokens <= tokens_before.tokens
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_reject_frees_slot_after_ack(self) -> None:
        """After the actor acks, a previously-full inbox accepts sends again."""

        from autogen.beta.network.transport.frames import ReceiptFrame

        bob_rule = Rule(
            limits=LimitsBlock(inbox=InboxBlock(max_pending=1, overflow="reject"))
        )
        hub, hc, link, alice, bob = await _spin_two_actors(bob_rule=bob_rule)
        try:
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            first_id = await session.send("one")
            await asyncio.sleep(0.02)
            with pytest.raises(InboxFullError):
                await session.send("two")

            # Manually ack the first envelope — simulates Bob's
            # (blocked) handler eventually completing and the
            # auto-ack path firing.
            endpoint = hub._endpoints[bob.actor_id]
            await hub._handle_receipt(
                endpoint,
                ReceiptFrame(envelope_id=first_id, status="ack"),
                bob.actor_id,
            )
            assert hub._inbox_pending[bob.actor_id] == 0

            # Now the retry succeeds.
            second_id = await session.send("two")
            assert second_id != first_id
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# max_pending — spool policy (overflow directory)
# ---------------------------------------------------------------------------


class TestMaxPendingSpool:
    @pytest.mark.asyncio
    async def test_spool_writes_to_overflow_without_bumping_counter(self) -> None:
        bob_rule = Rule(
            limits=LimitsBlock(inbox=InboxBlock(max_pending=1, overflow="spool"))
        )
        hub, hc, link, alice, bob = await _spin_two_actors(bob_rule=bob_rule)
        try:
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            first_id = await session.send("one")  # pending
            await asyncio.sleep(0.02)
            second_id = await session.send("two")  # spool
            await asyncio.sleep(0.01)
            third_id = await session.send("three")  # spool

            # Pending counter stays at max_pending = 1.
            assert hub._inbox_pending[bob.actor_id] == 1

            # The overflow files exist.
            overflow_entries = await hub._store.list(
                layout.actor_inbox_overflow_dir(bob.actor_id) + "/"
            )
            assert any(second_id in e for e in overflow_entries)
            assert any(third_id in e for e in overflow_entries)

            # The pending directory holds exactly one file — the first one.
            pending_entries = await hub._store.list(
                layout.actor_inbox_pending_dir(bob.actor_id) + "/"
            )
            assert len(pending_entries) == 1
            assert first_id in pending_entries[0]
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_spool_does_not_push_notify_frame(self) -> None:
        """Spooled envelopes wait for explicit drain — no NotifyFrame push."""

        bob_rule = Rule(
            limits=LimitsBlock(inbox=InboxBlock(max_pending=1, overflow="spool"))
        )
        hub, hc, link, alice, bob = await _spin_two_actors(bob_rule=bob_rule)
        try:
            notify_seen: list[str] = []

            # Record then block — we need the first envelope to fill
            # the pending slot (so the second spools), but we also
            # need the handler to observe the notify before blocking.
            async def record_then_block(envelope, client):
                notify_seen.append(envelope.envelope_id)
                await asyncio.Event().wait()

            bob.on("conversation")(record_then_block)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            first_id = await session.send("one")
            await asyncio.sleep(0.03)  # let the first notify land
            spooled_id = await session.send("two")
            await asyncio.sleep(0.03)

            # Bob's handler should have seen the first envelope but NOT
            # the spooled one — spool bypasses NotifyFrame delivery.
            assert first_id in notify_seen
            assert spooled_id not in notify_seen
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# Fan-out preflight — a single full recipient rejects the whole broadcast
# ---------------------------------------------------------------------------


class TestFanoutPreflight:
    @pytest.mark.asyncio
    async def test_one_full_recipient_rejects_entire_broadcast(self) -> None:
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
                        inbox=InboxBlock(max_pending=1, overflow="reject")
                    )
                ),
            )
            carol = await hc.register(
                _Echo("carol"), identity=ActorIdentity(name="carol")
            )
            # Block both recipients so the first broadcast fills their
            # pending slots and the second one hits the reject path.
            bob.on("broadcast")(_block_forever)
            carol.on("broadcast")(_block_forever)
            alice.on("broadcast")(_block_forever)

            session = await alice.open(
                SessionType.BROADCAST, target=["bob", "carol"]
            )
            await session.send("first")
            await asyncio.sleep(0.02)
            # Bob is now at his inbox ceiling. The next broadcast must
            # be rejected before it reaches Carol so the post stays
            # atomic (either all recipients deliver or none do).
            with pytest.raises(InboxFullError, match="bob"):
                await session.send("second")

            # Carol's pending count stays at 1 — she never saw the
            # second broadcast. Bob's stays at 1 too.
            assert hub._inbox_pending[carol.actor_id] == 1
            assert hub._inbox_pending[bob.actor_id] == 1
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# Hydrate — pending counter rebuilt from the store after restart
# ---------------------------------------------------------------------------


class TestInboxHydrate:
    @pytest.mark.asyncio
    async def test_hydrate_rebuilds_pending_counter(self, tmp_path) -> None:
        root = tmp_path / "hub"
        store = DiskKnowledgeStore(str(root))
        hub = Hub(store)
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)

        alice = await hc.register(
            _Echo("alice"), identity=ActorIdentity(name="alice")
        )
        bob = await hc.register(
            _Echo("bob"), identity=ActorIdentity(name="bob")
        )

        # Block Bob so sent envelopes stay in pending/ and survive the
        # hydrate round-trip. (If Bob acked, the files would move to
        # received/ and the pending counter would be 0 on both hubs —
        # which is correct but makes the test meaningless.)
        bob.on("conversation")(_block_forever)

        session = await alice.open(SessionType.CONVERSATION, target="bob")
        await session.send("one")
        await session.send("two")
        await asyncio.sleep(0.02)
        assert hub._inbox_pending[bob.actor_id] == 2

        bob_id = bob.actor_id
        await hc.close()
        await link.close()

        # Cold-restart: new hub, same store.
        hub2 = await Hub.open(DiskKnowledgeStore(str(root)))
        assert hub2._inbox_pending[bob_id] == 2
