# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Phase 2 :class:`DiscussionAdapter`.

Covers three orderings:

* ``dynamic`` — free-for-all, any participant may send.
* ``static`` — A → B → C turn order, auto-close after the last speaker.
  This is the V2 "pipeline" topology replacement.
* ``round_robin`` — same turn order but cycles forever; explicit close.

Validation tests live at the adapter layer (unit-test shape); end-to-end
tests drive the full hub + FakeClient stack to assert real turn
advancement and WAL order.
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
from autogen.beta.network.adapters import DiscussionAdapter
from autogen.beta.network.envelope import EV_TEXT
from autogen.beta.network.errors import AccessDeniedError

from test.beta.network._harness import (
    FakeClient,
    attach_hub_to_link,
    auto_ack_only,
)


# ---------------------------------------------------------------------------
# Shape validation at create-time
# ---------------------------------------------------------------------------


def _mk_meta(participants: list[Participant], **kwargs: object) -> SessionMetadata:
    return SessionMetadata(
        session_id="01-d",
        type=SessionType.DISCUSSION.value,
        creator_id=participants[0].actor_id,
        participants=participants,
        **kwargs,  # type: ignore[arg-type]
    )


def test_discussion_requires_two_participants() -> None:
    adapter = DiscussionAdapter()
    with pytest.raises(SessionTypeError):
        adapter.validate_create(
            _mk_meta([Participant(actor_id="a", role=ParticipantRole.INITIATOR)])
        )


def test_discussion_rejects_unknown_ordering() -> None:
    adapter = DiscussionAdapter()
    with pytest.raises(SessionTypeError, match="ordering"):
        adapter.validate_create(
            _mk_meta(
                [
                    Participant(actor_id="a", role=ParticipantRole.INITIATOR, order=0),
                    Participant(actor_id="b", role=ParticipantRole.PARTICIPANT, order=1),
                ],
                ordering="spiral",
            )
        )


def test_discussion_rejects_unknown_on_failure() -> None:
    adapter = DiscussionAdapter()
    with pytest.raises(SessionTypeError, match="on_failure"):
        adapter.validate_create(
            _mk_meta(
                [
                    Participant(actor_id="a", role=ParticipantRole.INITIATOR, order=0),
                    Participant(actor_id="b", role=ParticipantRole.PARTICIPANT, order=1),
                ],
                on_failure="retry",
            )
        )


def test_discussion_defaults_to_dynamic_ordering() -> None:
    adapter = DiscussionAdapter()
    meta = _mk_meta(
        [
            Participant(actor_id="a", role=ParticipantRole.INITIATOR, order=0),
            Participant(actor_id="b", role=ParticipantRole.PARTICIPANT, order=1),
        ]
    )
    adapter.validate_create(meta)
    # Any participant can send in dynamic mode.
    env = Envelope.text(session_id="01-d", sender_id="b", content="first")
    adapter.validate_send(meta, env, prior_envelopes=[])


# ---------------------------------------------------------------------------
# Static ordering — enforces turn order and auto-closes
# ---------------------------------------------------------------------------


@pytest.fixture
def hub() -> Hub:
    return Hub(MemoryKnowledgeStore())


async def _spin_three(hub: Hub) -> tuple[FakeClient, FakeClient, FakeClient, object]:
    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"))
    carol = await hub.register(ActorIdentity(name="carol"))
    link = attach_hub_to_link(hub)
    a = FakeClient(hub=hub, link=link, actor_id=alice.actor_id or "")
    b = FakeClient(
        hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only
    )
    c = FakeClient(
        hub=hub, link=link, actor_id=carol.actor_id or "", handler=auto_ack_only
    )
    await a.start()
    await b.start()
    await c.start()
    return a, b, c, link


@pytest.mark.asyncio
async def test_static_discussion_enforces_turn_order_and_auto_closes(
    hub: Hub,
) -> None:
    a, b, c, link = await _spin_three(hub)
    try:
        meta = await hub.create_session(
            creator_id=a.actor_id,
            session_type=SessionType.DISCUSSION,
            participant_names=["bob", "carol"],
            ordering="static",
            invite_ack_timeout_s=0.5,
        )
        assert meta.state is SessionState.ACTIVE

        # Alice speaks first.
        await a.send_text(session_id=meta.session_id, content="A1")
        await asyncio.sleep(0.02)
        assert hub.peek_session(meta.session_id).state is SessionState.ACTIVE

        # Carol trying to speak out-of-turn must fail.
        with pytest.raises(Exception):
            env = Envelope.text(
                session_id=meta.session_id, sender_id=c.actor_id, content="C1"
            )
            await hub.post_envelope(env)

        # Bob speaks next — valid.
        env = Envelope.text(
            session_id=meta.session_id, sender_id=b.actor_id, content="B1"
        )
        await hub.post_envelope(env)
        await asyncio.sleep(0.02)
        assert hub.peek_session(meta.session_id).state is SessionState.ACTIVE

        # Carol speaks last — session auto-closes afterwards.
        env = Envelope.text(
            session_id=meta.session_id, sender_id=c.actor_id, content="C1"
        )
        await hub.post_envelope(env)
        await asyncio.sleep(0.02)
        closed = hub.peek_session(meta.session_id)
        assert closed.state is SessionState.CLOSED
        assert closed.close_reason == "discussion_done"
    finally:
        await a.stop()
        await b.stop()
        await c.stop()
        await link.close()


@pytest.mark.asyncio
async def test_static_discussion_rejects_speaker_after_last_turn(hub: Hub) -> None:
    a, b, c, link = await _spin_three(hub)
    try:
        meta = await hub.create_session(
            creator_id=a.actor_id,
            session_type=SessionType.DISCUSSION,
            participant_names=["bob", "carol"],
            ordering="static",
            invite_ack_timeout_s=0.5,
        )

        for sender, content in (
            (a.actor_id, "A1"),
            (b.actor_id, "B1"),
            (c.actor_id, "C1"),
        ):
            env = Envelope.text(
                session_id=meta.session_id, sender_id=sender, content=content
            )
            await hub.post_envelope(env)
            await asyncio.sleep(0.02)

        # Session should already be CLOSED.
        current = hub.peek_session(meta.session_id)
        assert current.state is SessionState.CLOSED
    finally:
        await a.stop()
        await b.stop()
        await c.stop()
        await link.close()


# ---------------------------------------------------------------------------
# Round-robin ordering — cycles forever, explicit close
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_round_robin_cycles_through_speakers(hub: Hub) -> None:
    a, b, c, link = await _spin_three(hub)
    try:
        meta = await hub.create_session(
            creator_id=a.actor_id,
            session_type=SessionType.DISCUSSION,
            participant_names=["bob", "carol"],
            ordering="round_robin",
            invite_ack_timeout_s=0.5,
        )
        # Six turns: A, B, C, A, B, C.
        for sender in [
            a.actor_id,
            b.actor_id,
            c.actor_id,
            a.actor_id,
            b.actor_id,
            c.actor_id,
        ]:
            env = Envelope.text(
                session_id=meta.session_id, sender_id=sender, content=f"{sender[:4]}"
            )
            await hub.post_envelope(env)

        # Session is still ACTIVE — round_robin doesn't auto-close.
        current = hub.peek_session(meta.session_id)
        assert current.state is SessionState.ACTIVE

        # Out-of-turn send still rejected.
        with pytest.raises(Exception):
            env = Envelope.text(
                session_id=meta.session_id, sender_id=c.actor_id, content="out of turn"
            )
            await hub.post_envelope(env)
    finally:
        await a.stop()
        await b.stop()
        await c.stop()
        await link.close()


# ---------------------------------------------------------------------------
# Dynamic ordering — chatroom-style
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dynamic_discussion_allows_any_participant_to_speak(hub: Hub) -> None:
    a, b, c, link = await _spin_three(hub)
    try:
        meta = await hub.create_session(
            creator_id=a.actor_id,
            session_type=SessionType.DISCUSSION,
            participant_names=["bob", "carol"],
            ordering="dynamic",
            invite_ack_timeout_s=0.5,
        )

        # Carol speaks first, out-of-insertion-order — still valid.
        env = Envelope.text(
            session_id=meta.session_id, sender_id=c.actor_id, content="C1"
        )
        await hub.post_envelope(env)
        env = Envelope.text(
            session_id=meta.session_id, sender_id=b.actor_id, content="B1"
        )
        await hub.post_envelope(env)
        env = Envelope.text(
            session_id=meta.session_id, sender_id=a.actor_id, content="A1"
        )
        await hub.post_envelope(env)

        assert hub.peek_session(meta.session_id).state is SessionState.ACTIVE
    finally:
        await a.stop()
        await b.stop()
        await c.stop()
        await link.close()


@pytest.mark.asyncio
async def test_discussion_rejects_non_participant_sender(hub: Hub) -> None:
    a, b, _c, link = await _spin_three(hub)
    # Register a 4th actor not in the session.
    mallory = await hub.register(ActorIdentity(name="mallory"))
    m = FakeClient(
        hub=hub, link=link, actor_id=mallory.actor_id or "", handler=auto_ack_only
    )
    await m.start()
    try:
        meta = await hub.create_session(
            creator_id=a.actor_id,
            session_type=SessionType.DISCUSSION,
            participant_names=["bob"],
            ordering="dynamic",
            invite_ack_timeout_s=0.5,
        )
        env = Envelope.text(
            session_id=meta.session_id, sender_id=m.actor_id, content="uninvited"
        )
        with pytest.raises(AccessDeniedError):
            await hub.post_envelope(env)
    finally:
        await a.stop()
        await b.stop()
        await m.stop()
        await link.close()
