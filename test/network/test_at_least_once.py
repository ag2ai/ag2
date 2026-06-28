# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""At-least-once delivery: per-recipient inbox cursor advance via
``ReceiptFrame``, causation-id dedup index + ``find_envelope_by_causation``,
and replay of unacked envelopes on reconnect with
``HelloFrame.since_envelope_id``.

Two test patterns coexist here:

* HubClient-level: register through ``HubClient.register``; the default
  in-process notify path auto-acks each delivered envelope so the
  cursor naturally tracks the WAL head. Used for end-to-end coverage
  of cursor / dedup / replay over the normal flow.
* Raw-link: register through ``Hub.register`` and bind a bare
  ``LocalLinkClient``. No auto-ack runs, so tests can drive
  ``ReceiptFrame`` traffic by hand. Used for the low-level cursor
  invariants (monotonicity, malformed receipts, NACK behaviour).
"""

import asyncio
import json
from unittest import mock

import pytest

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_INVITE,
    EV_CHANNEL_INVITE_ACK,
    EV_TEXT,
    Envelope,
    HelloFrame,
    Hub,
    HubClient,
    LocalLink,
    NotifyFrame,
    Passport,
    ReceiptFrame,
    RequestFrame,
    Resume,
    WelcomeFrame,
)
from ag2.network.ids import make_id

from ._helpers import ScriptedConfig, wait_for_text_count


def _post_frame(envelope: Envelope) -> RequestFrame:
    """Build a control-plane ``post_envelope`` request — the wire path a
    raw client uses to post an envelope now that ``SendFrame`` is retired."""
    return RequestFrame(request_id=make_id(), op="post_envelope", params={"envelope": envelope.to_dict()})


def _agent(name: str, *replies: str) -> Agent:
    return Agent(name=name, config=ScriptedConfig(*replies))


async def _new_hub() -> Hub:
    return await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )


async def _drain_frame(link, timeout: float = 1.0):
    """Read one frame off a ``LocalLinkClient`` with a timeout."""

    async def _first():
        async for frame in link.frames():
            return frame
        raise AssertionError("link closed before a frame arrived")

    return await asyncio.wait_for(_first(), timeout)


async def _raw_bob(hub: Hub, name: str = "bob"):
    """Register an agent directly via ``Hub.register`` and bind a bare
    ``LocalLinkClient`` to it. The returned link can be driven with
    arbitrary frames without the default auto-ack interfering."""
    passport = await hub.register_identity(Passport(name=name), Resume())
    link = LocalLink(hub)
    raw_client = link.client()
    hub.bind_endpoint(raw_client.endpoint_id, passport.agent_id)
    return passport, raw_client


async def _consume_invite_and_ack(raw_client, agent_id: str, *, timeout: float = 1.0) -> str:
    """Walk inbound frames until ``EV_CHANNEL_INVITE`` lands, post the
    matching ``EV_CHANNEL_INVITE_ACK`` back via a ``post_envelope`` request,
    return the channel id the invite was for. Helper for raw-bob test setups
    that bypass the HubClient/AgentClient default handler."""

    async def _scan() -> str:
        async for frame in raw_client.frames():
            if not isinstance(frame, NotifyFrame):
                continue
            env = frame.envelope
            if env.event_type != EV_CHANNEL_INVITE:
                continue
            ack = Envelope(
                channel_id=env.channel_id,
                sender_id=agent_id,
                audience=None,
                event_type=EV_CHANNEL_INVITE_ACK,
                event_data={"channel_id": env.channel_id},
                causation_id=env.envelope_id,
            )
            await raw_client.send_frame(_post_frame(ack))
            return env.channel_id
        raise AssertionError("link closed before any invite arrived")

    return await asyncio.wait_for(_scan(), timeout)


async def _await_cursor(hub, agent_id, channel_id, predicate, *, timeout=5.0):
    """Poll the per-channel inbox cursor until ``predicate(cursor)`` holds.

    Acks travel an async path (notify handler -> ReceiptFrame -> hub), so a
    fixed sleep races with delivery on a slow runner; poll the cursor instead
    of guessing a delay.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate(hub.inbox_cursor(agent_id, channel_id)):
            return
        await asyncio.sleep(0.01)


class TestReceiptCursorAdvance:
    @pytest.mark.asyncio
    async def test_auto_ack_advances_cursor_to_wal_head(self) -> None:
        """End-to-end: a HubClient-served recipient auto-acks every
        delivery, so the cursor follows the WAL head."""
        hub = await _new_hub()
        alice = await hub.register(_agent("alice"))
        bob = await hub.register(_agent("bob"))

        try:
            channel = await alice.open(type="discussion", target=["bob"])
            await channel.send("hello bob")
            await wait_for_text_count(hub, channel.channel_id, 1)

            wal = await hub.read_wal(channel.channel_id)
            text_env = next(e for e in wal if e.event_type == EV_TEXT)

            await _await_cursor(hub, bob.agent_id, channel.channel_id, lambda c: c == text_env.envelope_id)
            assert hub.inbox_cursor(bob.agent_id, channel.channel_id) == text_env.envelope_id
            cursor_blob = await hub._store.read(f"/agents/{bob.agent_id}/inbox.cursors.json")
            assert json.loads(cursor_blob)[channel.channel_id] == text_env.envelope_id
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_cursor_tracks_wal_head_when_clock_does_not_advance(self) -> None:
        """Regression: envelope ids must sort in WAL-append order even when
        the wall clock does not advance between mints — coarse-resolution
        timers (Windows ~15 ms) or many envelopes inside one tick. The hub
        mints through a strictly-monotonic source, so the cursor follows the
        WAL head instead of landing on whichever same-tick envelope drew the
        largest random suffix. Frozen here (the worst case of a coarse clock)
        so the assertion would fail ~always without strict monotonicity."""
        with mock.patch("ag2.network.ids.time") as fake_time:
            fake_time.time_ns.return_value = 1_700_000_000_000_000_000

            hub = await _new_hub()
            alice = await hub.register(_agent("alice"))
            bob = await hub.register(_agent("bob"))

            try:
                # conversation channel: no turn-taking, so one sender may
                # post many envelopes back-to-back (all within the frozen tick).
                channel = await alice.open(type="conversation", target=["bob"])
                for i in range(20):
                    await channel.send(f"msg {i}")
                await wait_for_text_count(hub, channel.channel_id, 20)

                wal = await hub.read_wal(channel.channel_id)
                env_ids = [e.envelope_id for e in wal]
                assert env_ids == sorted(env_ids)  # sort order == append order
                assert len(set(env_ids)) == len(env_ids)  # all unique

                last_text = [e for e in wal if e.event_type == EV_TEXT][-1]
                await _await_cursor(hub, bob.agent_id, channel.channel_id, lambda c: c == last_text.envelope_id)
                assert hub.inbox_cursor(bob.agent_id, channel.channel_id) == last_text.envelope_id
            finally:
                await hub.close()

    @pytest.mark.asyncio
    async def test_ack_is_monotonic_under_out_of_order_receipts(self) -> None:
        """Acks for older envelope ids must not rewind the cursor."""
        hub = await _new_hub()
        alice = await hub.register(_agent("alice"))

        bob_passport, bob_raw = await _raw_bob(hub)

        try:
            channel_open = asyncio.create_task(alice.open(type="discussion", target=["bob"]))
            channel_id = await _consume_invite_and_ack(bob_raw, bob_passport.agent_id)
            channel = await channel_open
            assert channel.channel_id == channel_id

            await channel.send("first text")
            # Wait for bob to receive a frame for the text envelope.
            async for frame in bob_raw.frames():
                if isinstance(frame, NotifyFrame) and frame.envelope.event_type == EV_TEXT:
                    break
            wal = await hub.read_wal(channel_id)
            text_envs = [e for e in wal if e.event_type == EV_TEXT]
            assert len(text_envs) == 1
            text_env = text_envs[0]
            opened_env = next(e for e in wal if e.event_type != EV_TEXT and e.sender_id == alice.agent_id)
            # text_env was stamped after opened_env so it's strictly later.
            assert text_env.envelope_id > opened_env.envelope_id

            # Ack the newer (text) first.
            await bob_raw.send_frame(
                ReceiptFrame(
                    envelope_id=text_env.envelope_id,
                    status="ack",
                    recipient_id=bob_passport.agent_id,
                    channel_id=channel_id,
                )
            )
            # Then ack an older envelope. Cursor must NOT rewind.
            await bob_raw.send_frame(
                ReceiptFrame(
                    envelope_id=opened_env.envelope_id,
                    status="ack",
                    recipient_id=bob_passport.agent_id,
                    channel_id=channel_id,
                )
            )
            await _await_cursor(hub, bob_passport.agent_id, channel_id, lambda c: c == text_env.envelope_id)
            assert hub.inbox_cursor(bob_passport.agent_id, channel_id) == text_env.envelope_id
        finally:
            await bob_raw.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_nack_leaves_cursor_unchanged(self) -> None:
        """A NACK does not advance the cursor — without an ack later,
        the envelope will be replayed on the next reconnect."""
        hub = await _new_hub()
        alice = await hub.register(_agent("alice"))

        bob_passport, bob_raw = await _raw_bob(hub)

        try:
            channel_open = asyncio.create_task(alice.open(type="discussion", target=["bob"]))
            await _consume_invite_and_ack(bob_raw, bob_passport.agent_id)
            channel = await channel_open

            await channel.send("hello")
            async for frame in bob_raw.frames():
                if isinstance(frame, NotifyFrame) and frame.envelope.event_type == EV_TEXT:
                    text_env = frame.envelope
                    break

            cursor_before = hub.inbox_cursor(bob_passport.agent_id, channel.channel_id)
            await bob_raw.send_frame(
                ReceiptFrame(
                    envelope_id=text_env.envelope_id,
                    status="nack",
                    recipient_id=bob_passport.agent_id,
                    channel_id=channel.channel_id,
                    reason="processing failed",
                )
            )
            await asyncio.sleep(0.05)
            cursor_after = hub.inbox_cursor(bob_passport.agent_id, channel.channel_id)
            assert cursor_after == cursor_before
        finally:
            await bob_raw.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_receipt_without_recipient_id_is_ignored(self) -> None:
        """A receipt missing ``recipient_id`` cannot be attributed and
        must not affect any cursor — defensive against legacy or
        malformed clients."""
        hub = await _new_hub()
        alice = await hub.register(_agent("alice"))

        bob_passport, bob_raw = await _raw_bob(hub)

        try:
            channel_open = asyncio.create_task(alice.open(type="discussion", target=["bob"]))
            await _consume_invite_and_ack(bob_raw, bob_passport.agent_id)
            channel = await channel_open

            await channel.send("hello")
            async for frame in bob_raw.frames():
                if isinstance(frame, NotifyFrame) and frame.envelope.event_type == EV_TEXT:
                    text_env = frame.envelope
                    break

            cursor_before = hub.inbox_cursor(bob_passport.agent_id, channel.channel_id)
            await bob_raw.send_frame(
                ReceiptFrame(envelope_id=text_env.envelope_id, status="ack")  # recipient_id="" default
            )
            await asyncio.sleep(0.05)
            cursor_after = hub.inbox_cursor(bob_passport.agent_id, channel.channel_id)
            assert cursor_after == cursor_before
        finally:
            await bob_raw.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_receipt_without_channel_id_is_ignored(self) -> None:
        """An ack the hub can't attribute to a channel cannot advance any
        per-channel cursor — receipts must name their channel."""
        hub = await _new_hub()
        alice = await hub.register(_agent("alice"))

        bob_passport, bob_raw = await _raw_bob(hub)

        try:
            channel_open = asyncio.create_task(alice.open(type="discussion", target=["bob"]))
            await _consume_invite_and_ack(bob_raw, bob_passport.agent_id)
            channel = await channel_open

            await channel.send("hello")
            async for frame in bob_raw.frames():
                if isinstance(frame, NotifyFrame) and frame.envelope.event_type == EV_TEXT:
                    text_env = frame.envelope
                    break

            cursor_before = hub.inbox_cursor(bob_passport.agent_id, channel.channel_id)
            await bob_raw.send_frame(
                # recipient_id set, channel_id omitted (default "").
                ReceiptFrame(
                    envelope_id=text_env.envelope_id,
                    status="ack",
                    recipient_id=bob_passport.agent_id,
                )
            )
            await asyncio.sleep(0.05)
            cursor_after = hub.inbox_cursor(bob_passport.agent_id, channel.channel_id)
            assert cursor_after == cursor_before == ""
        finally:
            await bob_raw.close()
            await hub.close()


class TestCausationIndex:
    @pytest.mark.asyncio
    async def test_find_returns_prior_envelope_for_same_causation_id(self) -> None:
        hub = await _new_hub()
        alice = await hub.register(_agent("alice"))
        await hub.register(_agent("bob"))

        try:
            channel = await alice.open(type="discussion", target=["bob"])
            await channel.send("hello", causation_id="logical-retry-1")
            await wait_for_text_count(hub, channel.channel_id, 1)

            found = await hub.find_envelope_by_causation(
                channel.channel_id,
                sender_id=alice.agent_id,
                causation_id="logical-retry-1",
            )
            assert found is not None
            assert found.event_data.get("text") == "hello"
            assert found.sender_id == alice.agent_id
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_find_returns_none_when_causation_id_unseen(self) -> None:
        hub = await _new_hub()
        alice = await hub.register(_agent("alice"))
        await hub.register(_agent("bob"))

        try:
            channel = await alice.open(type="discussion", target=["bob"])
            await channel.send("hello")
            await wait_for_text_count(hub, channel.channel_id, 1)

            found = await hub.find_envelope_by_causation(
                channel.channel_id,
                sender_id=alice.agent_id,
                causation_id="never-existed",
            )
            assert found is None
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_envelope_with_null_causation_is_not_indexed(self) -> None:
        """Envelopes posted without a causation_id (the default) must
        not consume an index slot under any falsy key."""
        hub = await _new_hub()
        alice = await hub.register(_agent("alice"))
        await hub.register(_agent("bob"))

        try:
            channel = await alice.open(type="discussion", target=["bob"])
            await channel.send("hello")  # No causation_id supplied.
            await wait_for_text_count(hub, channel.channel_id, 1)

            wal = await hub.read_wal(channel.channel_id)
            text_env = next(e for e in wal if e.event_type == EV_TEXT)
            assert text_env.causation_id is None

            # No "" / None / falsy key should be present for alice in this channel.
            assert (channel.channel_id, alice.agent_id, "") not in hub._causation_index
            for (cid, sid, _key), env_id in hub._causation_index.items():
                if cid == channel.channel_id and sid == alice.agent_id:
                    # The only matches must be from the alice-sender protocol
                    # envelopes (invite carries no causation; opened carries none).
                    assert env_id != text_env.envelope_id
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_terminal_channel_prunes_index_entries(self) -> None:
        hub = await _new_hub()
        alice = await hub.register(_agent("alice"))
        await hub.register(_agent("bob"))

        try:
            channel = await alice.open(type="discussion", target=["bob"])
            await channel.send("hello", causation_id="will-be-pruned")
            await wait_for_text_count(hub, channel.channel_id, 1)
            assert (channel.channel_id, alice.agent_id, "will-be-pruned") in hub._causation_index

            await hub.close_channel(channel.channel_id, reason="test")
            assert not any(k[0] == channel.channel_id for k in hub._causation_index)

            found = await hub.find_envelope_by_causation(
                channel.channel_id,
                sender_id=alice.agent_id,
                causation_id="will-be-pruned",
            )
            assert found is None
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_hydrate_rebuilds_index_from_wal(self) -> None:
        """Restarting the hub against the same store re-populates the
        causation index from each active channel's persisted WAL."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
        alice = await hub.register(_agent("alice"))
        await hub.register(_agent("bob"))

        # Use two separate channels so alice can post two causation-keyed
        # texts back-to-back without round-robin interference.
        chan_a = await alice.open(type="discussion", target=["bob"])
        chan_b = await alice.open(type="discussion", target=["bob"])
        await chan_a.send("first", causation_id="c-first")
        await chan_b.send("second", causation_id="c-second")
        await wait_for_text_count(hub, chan_a.channel_id, 1)
        await wait_for_text_count(hub, chan_b.channel_id, 1)

        chan_a_id = chan_a.channel_id
        chan_b_id = chan_b.channel_id
        alice_id = alice.agent_id

        await hub.close()

        # Re-open against the same store: fresh memory, identical disk.
        hub2 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
        try:
            assert (chan_a_id, alice_id, "c-first") in hub2._causation_index
            assert (chan_b_id, alice_id, "c-second") in hub2._causation_index

            found = await hub2.find_envelope_by_causation(chan_b_id, sender_id=alice_id, causation_id="c-second")
            assert found is not None
            assert found.event_data.get("text") == "second"
        finally:
            await hub2.close()


class TestReplayOnReconnect:
    @pytest.mark.asyncio
    async def test_replay_redelivers_unacked_envelopes_past_since(self) -> None:
        """While bob is disconnected, alice posts envelopes targeted at
        him via ``hub.post_envelope`` (using ``ag2.task.*`` event types
        which bypass adapter round-robin). Reconnecting bob with
        ``since_envelope_id`` set to his last seen id replays only the
        envelopes posted past that mark, in WAL order."""
        hub = await _new_hub()
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

        try:
            channel = await alice.open(type="discussion", target=["bob"])
            await channel.send("seen-before-disconnect")
            await wait_for_text_count(hub, channel.channel_id, 1)
            await asyncio.sleep(0.05)
            wal = await hub.read_wal(channel.channel_id)
            seen_env = next(e for e in wal if e.event_type == EV_TEXT)

            await bob_hc.close()

            # Post two task-progress envelopes addressed to bob while he
            # is offline. Task events bypass adapter ``validate_send``
            # so they don't need round-robin compliance — they land in
            # the WAL and dispatch to bob's (now-missing) endpoint.
            missed_1 = Envelope(
                channel_id=channel.channel_id,
                sender_id=alice.agent_id,
                audience=[bob.agent_id],
                event_type="ag2.task.progress",
                event_data={"step": "missed-1"},
            )
            await hub.post_envelope(missed_1)
            missed_2 = Envelope(
                channel_id=channel.channel_id,
                sender_id=alice.agent_id,
                audience=[bob.agent_id],
                event_type="ag2.task.progress",
                event_data={"step": "missed-2"},
            )
            await hub.post_envelope(missed_2)

            raw_link = LocalLink(hub)
            raw_client = raw_link.client()
            try:
                await raw_client.send_frame(HelloFrame(name="bob", since_envelope_id=seen_env.envelope_id))
                welcome = await _drain_frame(raw_client)
                assert isinstance(welcome, WelcomeFrame)

                replayed: list[Envelope] = []
                for _ in range(2):
                    frame = await _drain_frame(raw_client, timeout=1.0)
                    assert isinstance(frame, NotifyFrame)
                    assert frame.recipient_id == bob.agent_id
                    replayed.append(frame.envelope)

                steps = [e.event_data.get("step") for e in replayed]
                assert steps == ["missed-1", "missed-2"]
            finally:
                await raw_client.close()
        finally:
            await alice_hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_replay_skips_terminal_channels(self) -> None:
        """A channel that closed while the recipient was offline is
        pruned from replay scope — replay only walks active channels."""
        hub = await _new_hub()
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

        try:
            channel = await alice.open(type="discussion", target=["bob"])
            await channel.send("seen")
            await wait_for_text_count(hub, channel.channel_id, 1)
            await asyncio.sleep(0.05)
            wal = await hub.read_wal(channel.channel_id)
            seen_env = next(e for e in wal if e.event_type == EV_TEXT)

            await bob_hc.close()
            await hub.close_channel(channel.channel_id, reason="offline test")

            raw_link = LocalLink(hub)
            raw_client = raw_link.client()
            try:
                await raw_client.send_frame(HelloFrame(name="bob", since_envelope_id=seen_env.envelope_id))
                welcome = await _drain_frame(raw_client)
                assert isinstance(welcome, WelcomeFrame)

                with pytest.raises(asyncio.TimeoutError):
                    await _drain_frame(raw_client, timeout=0.25)
            finally:
                await raw_client.close()
        finally:
            await alice_hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_replay_skips_envelopes_at_or_below_cursor(self) -> None:
        """When the persisted cursor is already past an envelope, it
        is not replayed even when ``since_envelope_id`` is empty."""
        hub = await _new_hub()
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

        try:
            channel = await alice.open(type="discussion", target=["bob"])
            await channel.send("first")
            await wait_for_text_count(hub, channel.channel_id, 1)
            wal = await hub.read_wal(channel.channel_id)
            text_env = next(e for e in wal if e.event_type == EV_TEXT)
            # Wait until bob has acked through the WAL head (not merely acked
            # *something*), so the reconnect below has nothing left to replay.
            await _await_cursor(hub, bob.agent_id, channel.channel_id, lambda c: c == text_env.envelope_id)

            cursor = hub.inbox_cursor(bob.agent_id, channel.channel_id)
            assert cursor == text_env.envelope_id
            await bob_hc.close()

            raw_link = LocalLink(hub)
            raw_client = raw_link.client()
            try:
                await raw_client.send_frame(HelloFrame(name="bob", since_envelope_id=""))
                welcome = await _drain_frame(raw_client)
                assert isinstance(welcome, WelcomeFrame)

                # Cursor wins over the empty since_envelope_id, so no
                # replay frames arrive.
                with pytest.raises(asyncio.TimeoutError):
                    await _drain_frame(raw_client, timeout=0.25)
            finally:
                await raw_client.close()
        finally:
            await alice_hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_reconnect_without_since_does_not_replay(self) -> None:
        """A plain reconnect (``since_envelope_id=None``) gets a Welcome
        but no historic frames — replay is opt-in."""
        hub = await _new_hub()
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

        try:
            channel = await alice.open(type="discussion", target=["bob"])
            await channel.send("miscible")
            await wait_for_text_count(hub, channel.channel_id, 1)
            await bob_hc.close()

            raw_link = LocalLink(hub)
            raw_client = raw_link.client()
            try:
                await raw_client.send_frame(HelloFrame(name="bob"))  # no since
                welcome = await _drain_frame(raw_client)
                assert isinstance(welcome, WelcomeFrame)

                with pytest.raises(asyncio.TimeoutError):
                    await _drain_frame(raw_client, timeout=0.25)
            finally:
                await raw_client.close()
        finally:
            await alice_hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_ack_in_one_channel_does_not_suppress_replay_in_another(self) -> None:
        """The delivery cursor is per (recipient, channel). Acking a
        newer envelope in one channel must not skip an older unacked
        envelope in a different channel on reconnect — the failure mode
        a single global high-water cursor would cause."""
        hub = await _new_hub()
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

        try:
            chan_a = await alice.open(type="discussion", target=["bob"])
            chan_b = await alice.open(type="discussion", target=["bob"])
            await asyncio.sleep(0.05)  # bob auto-acks both opened envelopes
            await bob_hc.close()
            await asyncio.sleep(0.05)

            # Channel B's envelope is stamped first (lower id) and stays
            # unacked; channel A's is stamped later (higher id) and gets
            # acked. A single global cursor at channel A's id would
            # wrongly suppress channel B's replay.
            missed_b = Envelope(
                channel_id=chan_b.channel_id,
                sender_id=alice.agent_id,
                audience=[bob.agent_id],
                event_type="ag2.task.progress",
                event_data={"step": "missed-b"},
            )
            await hub.post_envelope(missed_b)
            missed_a = Envelope(
                channel_id=chan_a.channel_id,
                sender_id=alice.agent_id,
                audience=[bob.agent_id],
                event_type="ag2.task.progress",
                event_data={"step": "missed-a"},
            )
            missed_a_id = await hub.post_envelope(missed_a)

            # Reconnect once; ack ONLY channel A's envelope.
            raw_link = LocalLink(hub)
            raw1 = raw_link.client()
            try:
                await raw1.send_frame(HelloFrame(name="bob", since_envelope_id=""))
                welcome = await _drain_frame(raw1)
                assert isinstance(welcome, WelcomeFrame)
                await raw1.send_frame(
                    ReceiptFrame(
                        envelope_id=missed_a_id,
                        status="ack",
                        recipient_id=bob.agent_id,
                        channel_id=chan_a.channel_id,
                    )
                )
                await asyncio.sleep(0.05)
            finally:
                await raw1.close()

            # Reconnect again: channel B's envelope must still replay
            # (its cursor never advanced); channel A's must not.
            raw2 = raw_link.client()
            try:
                await raw2.send_frame(HelloFrame(name="bob", since_envelope_id=""))
                welcome = await _drain_frame(raw2)
                assert isinstance(welcome, WelcomeFrame)
                steps: list[str | None] = []
                while True:
                    try:
                        frame = await _drain_frame(raw2, timeout=0.5)
                    except asyncio.TimeoutError:
                        break
                    if isinstance(frame, NotifyFrame) and frame.envelope.event_type == "ag2.task.progress":
                        steps.append(frame.envelope.event_data.get("step"))
                assert "missed-b" in steps, f"channel B replay was suppressed; saw {steps}"
                assert "missed-a" not in steps, f"channel A re-replayed after ack; saw {steps}"
            finally:
                await raw2.close()
        finally:
            await alice_hc.close()
            await hub.close()


class TestCursorPersistence:
    @pytest.mark.asyncio
    async def test_hydrate_restores_cursor(self) -> None:
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
        alice = await hub.register(_agent("alice"))
        bob = await hub.register(_agent("bob"))
        channel = await alice.open(type="discussion", target=["bob"])
        await channel.send("hello")
        await wait_for_text_count(hub, channel.channel_id, 1)
        bob_id = bob.agent_id
        channel_id = channel.channel_id
        wal = await hub.read_wal(channel_id)
        text_env = next(e for e in wal if e.event_type == EV_TEXT)
        await _await_cursor(hub, bob_id, channel_id, lambda c: c == text_env.envelope_id)
        expected_cursor = hub.inbox_cursor(bob_id, channel_id)
        assert expected_cursor == text_env.envelope_id

        await hub.close()

        hub2 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
        try:
            assert hub2.inbox_cursor(bob_id, channel_id) == expected_cursor
        finally:
            await hub2.close()

    @pytest.mark.asyncio
    async def test_unregister_clears_cursor_and_disk_file(self) -> None:
        hub = await _new_hub()
        alice = await hub.register(_agent("alice"))
        bob = await hub.register(_agent("bob"))

        try:
            channel = await alice.open(type="discussion", target=["bob"])
            await channel.send("hello")
            await wait_for_text_count(hub, channel.channel_id, 1)
            bob_id = bob.agent_id
            await _await_cursor(hub, bob_id, channel.channel_id, lambda c: c != "")
            assert bob_id in hub._inbox_cursors

            await bob.unregister()
            assert bob_id not in hub._inbox_cursors
            assert await hub._store.read(f"/agents/{bob_id}/inbox.cursors.json") is None
        finally:
            await hub.close()


class TestRedeliveryIdempotency:
    @pytest.mark.asyncio
    async def test_redelivered_trigger_does_not_double_post(self) -> None:
        """At-least-once redelivery of a trigger the agent already
        answered must not re-run the LLM or post a second reply. The
        default handler short-circuits on the causation index. Uses a
        conversation channel (free-form turns) so only the dedup guard —
        not turn ordering — can prevent the second reply."""
        hub = await _new_hub()
        alice = await hub.register(_agent("alice"))
        bob = await hub.register(_agent("bob", "reply-A", "reply-B"))

        try:
            channel = await alice.open(type="conversation", target=["bob"])
            await channel.send("question", audience=[bob.agent_id])

            # alice's question + bob's first reply.
            wal = await wait_for_text_count(hub, channel.channel_id, 2)
            trigger = next(e for e in wal if e.event_type == EV_TEXT and e.sender_id == alice.agent_id)
            bob_first = [e for e in wal if e.event_type == EV_TEXT and e.sender_id == bob.agent_id]
            assert len(bob_first) == 1
            assert bob_first[0].event_data.get("text") == "reply-A"

            # Redeliver the exact same trigger envelope.
            await bob.receive(trigger)
            await asyncio.sleep(0.1)

            wal_after = await hub.read_wal(channel.channel_id)
            bob_after = [e for e in wal_after if e.event_type == EV_TEXT and e.sender_id == bob.agent_id]
            assert len(bob_after) == 1, f"redelivery double-posted: {[e.event_data for e in bob_after]}"
            assert bob_after[0].event_data.get("text") == "reply-A"
        finally:
            await hub.close()
