# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Phase 2 :class:`AuctionAdapter`.

Auction lifecycle:
  1. Initiator posts the RFP (first text envelope).
  2. Other participants post bids (one per bidder).
  3. Initiator posts ``ag2.auction.select`` naming a winner.
  4. Session continues 1:1 between initiator and winner.
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
from autogen.beta.network.adapters import AuctionAdapter
from autogen.beta.network.adapters.auction import EV_AUCTION_SELECT
from autogen.beta.network.envelope import EV_TEXT

from test.beta.network._harness import (
    FakeClient,
    attach_hub_to_link,
    auto_ack_only,
)


# ---------------------------------------------------------------------------
# Adapter unit tests — shape + phase transitions
# ---------------------------------------------------------------------------


def _mk_meta(participants: list[Participant]) -> SessionMetadata:
    return SessionMetadata(
        session_id="01-auc",
        type=SessionType.AUCTION.value,
        creator_id=participants[0].actor_id,
        participants=participants,
    )


def _rfp(sender: str = "a") -> Envelope:
    return Envelope.text(session_id="01-auc", sender_id=sender, content="task spec")


def _bid(sender: str) -> Envelope:
    return Envelope.text(session_id="01-auc", sender_id=sender, content=f"bid from {sender}")


def _select(winner: str, sender: str = "a") -> Envelope:
    return Envelope(
        session_id="01-auc",
        sender_id=sender,
        event_type=EV_AUCTION_SELECT,
        event_data={"winner_id": winner},
    )


def _three_party_meta() -> SessionMetadata:
    return _mk_meta(
        [
            Participant(actor_id="a", role=ParticipantRole.INITIATOR, order=0),
            Participant(actor_id="b", role=ParticipantRole.PARTICIPANT, order=1),
            Participant(actor_id="c", role=ParticipantRole.PARTICIPANT, order=2),
        ]
    )


def test_auction_create_requires_two_participants() -> None:
    adapter = AuctionAdapter()
    with pytest.raises(SessionTypeError):
        adapter.validate_create(
            _mk_meta([Participant(actor_id="a", role=ParticipantRole.INITIATOR)])
        )


def test_auction_first_send_must_be_initiator_rfp() -> None:
    adapter = AuctionAdapter()
    meta = _three_party_meta()
    with pytest.raises(SessionTypeError, match="first send must be the RFP"):
        adapter.validate_send(meta, _bid("b"), prior_envelopes=[])


def test_auction_initiator_rfp_accepted() -> None:
    adapter = AuctionAdapter()
    meta = _three_party_meta()
    adapter.validate_send(meta, _rfp("a"), prior_envelopes=[])


def test_auction_accepts_bid_after_rfp() -> None:
    adapter = AuctionAdapter()
    meta = _three_party_meta()
    adapter.validate_send(meta, _bid("b"), prior_envelopes=[_rfp("a")])


def test_auction_rejects_second_bid_from_same_bidder() -> None:
    adapter = AuctionAdapter()
    meta = _three_party_meta()
    with pytest.raises(SessionTypeError, match="each bidder may post one bid"):
        adapter.validate_send(
            meta,
            _bid("b"),
            prior_envelopes=[_rfp("a"), _bid("b")],
        )


def test_auction_rejects_initiator_send_during_bidding() -> None:
    adapter = AuctionAdapter()
    meta = _three_party_meta()
    with pytest.raises(SessionTypeError, match="initiator cannot speak"):
        adapter.validate_send(
            meta,
            Envelope.text(session_id="01-auc", sender_id="a", content="hmm"),
            prior_envelopes=[_rfp("a"), _bid("b")],
        )


def test_auction_select_must_name_a_participant() -> None:
    adapter = AuctionAdapter()
    meta = _three_party_meta()
    with pytest.raises(SessionTypeError, match="winner_id must be a session participant"):
        adapter.validate_send(
            meta,
            _select("nonparticipant"),
            prior_envelopes=[_rfp("a"), _bid("b")],
        )


def test_auction_select_cannot_be_initiator() -> None:
    adapter = AuctionAdapter()
    meta = _three_party_meta()
    with pytest.raises(SessionTypeError, match="winner cannot be the initiator"):
        adapter.validate_send(
            meta,
            _select("a"),
            prior_envelopes=[_rfp("a"), _bid("b")],
        )


def test_auction_select_requires_rfp_first() -> None:
    adapter = AuctionAdapter()
    meta = _three_party_meta()
    with pytest.raises(SessionTypeError, match="before the RFP is posted"):
        adapter.validate_send(meta, _select("b"), prior_envelopes=[])


def test_auction_select_can_only_be_issued_once() -> None:
    adapter = AuctionAdapter()
    meta = _three_party_meta()
    prior = [_rfp("a"), _bid("b"), _select("b")]
    with pytest.raises(SessionTypeError, match="already posted"):
        adapter.validate_send(meta, _select("c"), prior_envelopes=prior)


def test_auction_select_must_come_from_initiator() -> None:
    adapter = AuctionAdapter()
    meta = _three_party_meta()
    prior = [_rfp("a"), _bid("b")]
    with pytest.raises(SessionTypeError, match="only the initiator may send select"):
        adapter.validate_send(meta, _select("b", sender="c"), prior_envelopes=prior)


def test_auction_post_select_limits_to_initiator_and_winner() -> None:
    adapter = AuctionAdapter()
    meta = _three_party_meta()
    prior = [_rfp("a"), _bid("b"), _bid("c"), _select("b")]

    # Loser (c) cannot speak after select.
    with pytest.raises(
        SessionTypeError, match="only the initiator and the selected winner"
    ):
        adapter.validate_send(
            meta,
            Envelope.text(session_id="01-auc", sender_id="c", content="hey"),
            prior_envelopes=prior,
        )
    # Winner and initiator can.
    adapter.validate_send(
        meta,
        Envelope.text(session_id="01-auc", sender_id="b", content="ok"),
        prior_envelopes=prior,
    )
    adapter.validate_send(
        meta,
        Envelope.text(session_id="01-auc", sender_id="a", content="clarify"),
        prior_envelopes=prior,
    )


# ---------------------------------------------------------------------------
# Hub integration — full auction lifecycle
# ---------------------------------------------------------------------------


@pytest.fixture
def hub() -> Hub:
    return Hub(MemoryKnowledgeStore())


@pytest.mark.asyncio
async def test_hub_auction_full_lifecycle(hub: Hub) -> None:
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
    try:
        meta = await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type=SessionType.AUCTION,
            participant_names=["bob", "carol"],
            invite_ack_timeout_s=0.5,
        )
        assert meta.state is SessionState.ACTIVE

        # RFP
        env = Envelope.text(
            session_id=meta.session_id,
            sender_id=alice.actor_id or "",
            content="build a frobnicator",
        )
        await hub.post_envelope(env)

        # Bob and Carol both bid.
        for bidder in (bob.actor_id, carol.actor_id):
            env = Envelope.text(
                session_id=meta.session_id,
                sender_id=bidder,
                content=f"bid from {bidder[:4]}",
            )
            await hub.post_envelope(env)

        # Alice picks Carol.
        select = Envelope(
            session_id=meta.session_id,
            sender_id=alice.actor_id or "",
            event_type=EV_AUCTION_SELECT,
            event_data={"winner_id": carol.actor_id or ""},
        )
        await hub.post_envelope(select)

        # Carol replies, session stays ACTIVE.
        env = Envelope.text(
            session_id=meta.session_id,
            sender_id=carol.actor_id or "",
            content="sure, give me requirements",
        )
        await hub.post_envelope(env)
        assert hub.peek_session(meta.session_id).state is SessionState.ACTIVE

        # Bob (losing bidder) cannot speak anymore.
        env = Envelope.text(
            session_id=meta.session_id,
            sender_id=bob.actor_id or "",
            content="hey wait",
        )
        with pytest.raises(Exception):
            await hub.post_envelope(env)
    finally:
        await a.stop()
        await b.stop()
        await c.stop()
        await link.close()
