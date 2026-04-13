# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Phase 2 :class:`BroadcastAdapter`.

Broadcast is the one-to-many fan-out session: the initiator posts,
every recipient gets a NotifyFrame, recipients don't post back. These
tests cover:

* Create-time shape validation (initiator + ≥1 recipient, exactly one
  initiator).
* Handshake quorum (default all, custom ``required_acks``, partial
  reject that still reaches quorum, full reject that fails the
  handshake).
* Delivery direction (only initiator may send; non-initiator sends are
  rejected by :class:`SessionTypeError`).
* Fan-out semantics (every non-sender participant receives a notify
  frame with its own ``recipient_id`` stamped on the clone).
* Explicit close is required — broadcasts do not auto-close after the
  first message.
"""

from __future__ import annotations

import asyncio

import pytest

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    Envelope,
    Hub,
    Participant,
    ParticipantRole,
    SessionMetadata,
    SessionState,
    SessionType,
    SessionTypeError,
)
from autogen.beta.network.adapters import BroadcastAdapter
from autogen.beta.network.envelope import EV_SESSION_INVITE, EV_TEXT
from autogen.beta.network.errors import InviteRejectedError

from test.beta.network._harness import (
    FakeClient,
    attach_hub_to_link,
    auto_ack_only,
)


# ---------------------------------------------------------------------------
# validate_create
# ---------------------------------------------------------------------------


def _mk_meta(participants: list[Participant]) -> SessionMetadata:
    return SessionMetadata(
        session_id="01-s",
        type=SessionType.BROADCAST.value,
        creator_id=participants[0].actor_id,
        participants=participants,
    )


def test_broadcast_requires_initiator_plus_at_least_one_recipient() -> None:
    adapter = BroadcastAdapter()
    with pytest.raises(SessionTypeError):
        adapter.validate_create(
            _mk_meta([Participant(actor_id="a", role=ParticipantRole.INITIATOR)])
        )


def test_broadcast_requires_exactly_one_initiator() -> None:
    adapter = BroadcastAdapter()
    with pytest.raises(SessionTypeError):
        adapter.validate_create(
            _mk_meta(
                [
                    Participant(actor_id="a", role=ParticipantRole.INITIATOR),
                    Participant(actor_id="b", role=ParticipantRole.INITIATOR),
                    Participant(actor_id="c", role=ParticipantRole.PARTICIPANT),
                ]
            )
        )


def test_broadcast_ok_with_initiator_and_three_recipients() -> None:
    adapter = BroadcastAdapter()
    adapter.validate_create(
        _mk_meta(
            [
                Participant(actor_id="a", role=ParticipantRole.INITIATOR),
                Participant(actor_id="b", role=ParticipantRole.PARTICIPANT),
                Participant(actor_id="c", role=ParticipantRole.PARTICIPANT),
                Participant(actor_id="d", role=ParticipantRole.PARTICIPANT),
            ]
        )
    )


# ---------------------------------------------------------------------------
# validate_send — only initiator may send
# ---------------------------------------------------------------------------


def test_broadcast_rejects_text_from_non_initiator() -> None:
    adapter = BroadcastAdapter()
    meta = _mk_meta(
        [
            Participant(actor_id="a", role=ParticipantRole.INITIATOR),
            Participant(actor_id="b", role=ParticipantRole.PARTICIPANT),
        ]
    )
    env = Envelope.text(session_id="01-s", sender_id="b", content="hi")
    with pytest.raises(SessionTypeError, match="only the initiator may send"):
        adapter.validate_send(meta, env, prior_envelopes=[])


def test_broadcast_rejects_non_text_event_types() -> None:
    adapter = BroadcastAdapter()
    meta = _mk_meta(
        [
            Participant(actor_id="a", role=ParticipantRole.INITIATOR),
            Participant(actor_id="b", role=ParticipantRole.PARTICIPANT),
        ]
    )
    env = Envelope(
        session_id="01-s",
        sender_id="a",
        event_type="custom.event",
    )
    with pytest.raises(SessionTypeError, match="only carries"):
        adapter.validate_send(meta, env, prior_envelopes=[])


def test_broadcast_accepts_initiator_send_with_no_recipient() -> None:
    adapter = BroadcastAdapter()
    meta = _mk_meta(
        [
            Participant(actor_id="a", role=ParticipantRole.INITIATOR),
            Participant(actor_id="b", role=ParticipantRole.PARTICIPANT),
        ]
    )
    env = Envelope.text(session_id="01-s", sender_id="a", content="hi")
    adapter.validate_send(meta, env, prior_envelopes=[])


def test_broadcast_rejects_explicit_non_participant_recipient() -> None:
    adapter = BroadcastAdapter()
    meta = _mk_meta(
        [
            Participant(actor_id="a", role=ParticipantRole.INITIATOR),
            Participant(actor_id="b", role=ParticipantRole.PARTICIPANT),
        ]
    )
    env = Envelope.text(
        session_id="01-s", sender_id="a", content="hi", recipient_id="nonparticipant"
    )
    with pytest.raises(SessionTypeError, match="not a participant"):
        adapter.validate_send(meta, env, prior_envelopes=[])


# ---------------------------------------------------------------------------
# Integration with the hub — handshake and fan-out
# ---------------------------------------------------------------------------


@pytest.fixture
def hub() -> Hub:
    return Hub(MemoryKnowledgeStore())


@pytest.mark.asyncio
async def test_hub_broadcast_handshake_waits_for_all_recipients_by_default(
    hub: Hub,
) -> None:
    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"))
    carol = await hub.register(ActorIdentity(name="carol"))
    link = attach_hub_to_link(hub)
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only)
    c = FakeClient(hub=hub, link=link, actor_id=carol.actor_id or "", handler=auto_ack_only)
    await b.start()
    await c.start()
    try:
        meta = await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type=SessionType.BROADCAST,
            participant_names=["bob", "carol"],
            invite_ack_timeout_s=0.5,
        )
        assert meta.state is SessionState.ACTIVE
        assert meta.type == "broadcast"
        assert [p.actor_id for p in meta.participants] == [
            alice.actor_id,
            bob.actor_id,
            carol.actor_id,
        ]
    finally:
        await b.stop()
        await c.stop()
        await link.close()


@pytest.mark.asyncio
async def test_hub_broadcast_returns_early_on_required_acks_quorum(
    hub: Hub,
) -> None:
    """With ``required_acks=1`` the session goes ACTIVE after one ack."""

    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"))
    carol = await hub.register(ActorIdentity(name="carol"))
    link = attach_hub_to_link(hub)
    # Only Bob auto-acks; Carol stays silent.
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only)
    c = FakeClient(hub=hub, link=link, actor_id=carol.actor_id or "")
    await b.start()
    await c.start()
    try:
        meta = await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type=SessionType.BROADCAST,
            participant_names=["bob", "carol"],
            required_acks=1,
            invite_ack_timeout_s=0.5,
        )
        assert meta.state is SessionState.ACTIVE
    finally:
        await b.stop()
        await c.stop()
        await link.close()


@pytest.mark.asyncio
async def test_hub_broadcast_times_out_when_quorum_not_reached(hub: Hub) -> None:
    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"))
    carol = await hub.register(ActorIdentity(name="carol"))
    link = attach_hub_to_link(hub)
    # Only one recipient acks — quorum=all = 2, so the handshake times out.
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only)
    c = FakeClient(hub=hub, link=link, actor_id=carol.actor_id or "")
    await b.start()
    await c.start()
    try:
        with pytest.raises(InviteRejectedError, match="1/2 invites acked"):
            await hub.create_session(
                creator_id=alice.actor_id or "",
                session_type=SessionType.BROADCAST,
                participant_names=["bob", "carol"],
                invite_ack_timeout_s=0.25,
            )
    finally:
        await b.stop()
        await c.stop()
        await link.close()


@pytest.mark.asyncio
async def test_hub_broadcast_delivers_to_all_recipients(hub: Hub) -> None:
    """A broadcast envelope from the initiator should land in every
    recipient's notify queue, with each clone's ``recipient_id`` set to
    the individual recipient."""

    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"))
    carol = await hub.register(ActorIdentity(name="carol"))
    link = attach_hub_to_link(hub)
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only)
    c = FakeClient(hub=hub, link=link, actor_id=carol.actor_id or "", handler=auto_ack_only)
    a = FakeClient(hub=hub, link=link, actor_id=alice.actor_id or "")
    await a.start()
    await b.start()
    await c.start()
    try:
        meta = await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type=SessionType.BROADCAST,
            participant_names=["bob", "carol"],
            invite_ack_timeout_s=0.5,
        )
        assert meta.state is SessionState.ACTIVE

        # Drain the invite envelopes out of Bob & Carol's queues so they
        # don't shadow the text broadcast we're about to send.
        for _ in range(2):  # each saw one invite
            try:
                await asyncio.wait_for(b.notify_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                break
        for _ in range(2):
            try:
                await asyncio.wait_for(c.notify_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                break

        # Alice broadcasts (no recipient_id — targets everyone).
        await a.send_text(
            session_id=meta.session_id,
            content="hear ye hear ye",
        )
        await asyncio.sleep(0.05)

        bob_got = await asyncio.wait_for(b.notify_queue.get(), timeout=0.5)
        carol_got = await asyncio.wait_for(c.notify_queue.get(), timeout=0.5)

        assert bob_got.event_type == EV_TEXT
        assert carol_got.event_type == EV_TEXT
        assert bob_got.content() == "hear ye hear ye"
        assert carol_got.content() == "hear ye hear ye"
        # Each clone is addressed to its own recipient.
        assert bob_got.recipient_id == bob.actor_id
        assert carol_got.recipient_id == carol.actor_id
        # Alice never gets her own broadcast back.
        assert a.notify_queue.empty()
    finally:
        await a.stop()
        await b.stop()
        await c.stop()
        await link.close()


@pytest.mark.asyncio
async def test_hub_broadcast_does_not_auto_close_after_first_message(hub: Hub) -> None:
    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"))
    link = attach_hub_to_link(hub)
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only)
    a = FakeClient(hub=hub, link=link, actor_id=alice.actor_id or "")
    await a.start()
    await b.start()
    try:
        meta = await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type=SessionType.BROADCAST,
            participant_names=["bob"],
            invite_ack_timeout_s=0.5,
        )
        assert meta.state is SessionState.ACTIVE

        # Send three messages; the session stays ACTIVE throughout.
        for msg in ("one", "two", "three"):
            await a.send_text(session_id=meta.session_id, content=msg)
            await asyncio.sleep(0.02)

        current = hub.peek_session(meta.session_id)
        assert current is not None and current.state is SessionState.ACTIVE

        # Explicit close is what ends it.
        await hub.close_session(meta.session_id)
        closed = hub.peek_session(meta.session_id)
        assert closed is not None and closed.state is SessionState.CLOSED
    finally:
        await a.stop()
        await b.stop()
        await link.close()
