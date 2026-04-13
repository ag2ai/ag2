# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for non-participant subscriptions (Phase 2).

Phase 1 only allowed participants to subscribe to a session. Phase 2
adds ``rule.access.subscribe.sessions`` with three policies:

* ``member-only`` (default, Phase 1 behavior).
* ``public-within-hub`` — any hub-registered actor may observe.
* ``public`` — reserved for federation; currently same as
  public-within-hub.

Rule conflict resolution is "most restrictive wins": even one
participant with ``member-only`` vetoes public observation.
"""

from __future__ import annotations

import asyncio

import pytest

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    AccessBlock,
    ActorIdentity,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Rule,
    SessionType,
    SubscribeAccess,
)
from autogen.beta.network.envelope import EV_TEXT
from autogen.beta.network.rule import (
    SUBSCRIBE_HUB_PUBLIC,
    SUBSCRIBE_MEMBER_ONLY,
    SUBSCRIBE_PUBLIC,
)

from test.beta.network._harness import (
    FakeClient,
    attach_hub_to_link,
    auto_ack_only,
)


# ---------------------------------------------------------------------------
# SubscribeAccess unit tests
# ---------------------------------------------------------------------------


def test_subscribe_access_defaults_to_member_only() -> None:
    access = SubscribeAccess()
    assert access.sessions == SUBSCRIBE_MEMBER_ONLY
    assert not access.allows_session_observer(
        is_participant=False, is_hub_member=True
    )
    assert access.allows_session_observer(is_participant=True, is_hub_member=True)


def test_subscribe_access_hub_public_allows_hub_members() -> None:
    access = SubscribeAccess(sessions=SUBSCRIBE_HUB_PUBLIC)
    assert access.allows_session_observer(is_participant=False, is_hub_member=True)
    assert not access.allows_session_observer(is_participant=False, is_hub_member=False)


def test_subscribe_access_round_trips_through_dict() -> None:
    original = SubscribeAccess(sessions=SUBSCRIBE_HUB_PUBLIC, tasks="owner-or-member")
    restored = SubscribeAccess.from_dict(original.to_dict())
    assert restored.sessions == SUBSCRIBE_HUB_PUBLIC
    assert restored.tasks == "owner-or-member"


def test_subscribe_access_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        SubscribeAccess.from_dict({"sessions": "admins-only"})


def test_rule_access_block_serializes_subscribe_block() -> None:
    rule = Rule(
        access=AccessBlock(
            subscribe=SubscribeAccess(sessions=SUBSCRIBE_HUB_PUBLIC)
        )
    )
    data = rule.to_dict()
    assert data["access"]["subscribe"]["sessions"] == SUBSCRIBE_HUB_PUBLIC
    restored = Rule.from_dict(data)
    assert restored.access.subscribe.sessions == SUBSCRIBE_HUB_PUBLIC


# ---------------------------------------------------------------------------
# End-to-end hub subscription gating
# ---------------------------------------------------------------------------


@pytest.fixture
def hub() -> Hub:
    return Hub(MemoryKnowledgeStore())


async def _spin_session_with_observer(
    hub: Hub, *, public: bool, observer_rule: Rule | None = None
) -> tuple[FakeClient, FakeClient, FakeClient, object, str]:
    """Create alice-bob session and register a mallory observer."""

    if public:
        public_rule = Rule(
            access=AccessBlock(subscribe=SubscribeAccess(sessions=SUBSCRIBE_HUB_PUBLIC))
        )
        alice = await hub.register(ActorIdentity(name="alice"), rule=public_rule)
        bob = await hub.register(ActorIdentity(name="bob"), rule=public_rule)
    else:
        alice = await hub.register(ActorIdentity(name="alice"))
        bob = await hub.register(ActorIdentity(name="bob"))
    mallory = await hub.register(
        ActorIdentity(name="mallory"),
        rule=observer_rule
        or Rule(
            access=AccessBlock(subscribe=SubscribeAccess(sessions=SUBSCRIBE_HUB_PUBLIC))
        ),
    )

    link = attach_hub_to_link(hub)
    a = FakeClient(hub=hub, link=link, actor_id=alice.actor_id or "")
    b = FakeClient(
        hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only
    )
    m = FakeClient(hub=hub, link=link, actor_id=mallory.actor_id or "")
    await a.start()
    await b.start()
    await m.start()

    meta = await hub.create_session(
        creator_id=alice.actor_id or "",
        session_type=SessionType.CONVERSATION,
        participant_names=["bob"],
        invite_ack_timeout_s=0.5,
    )
    return a, b, m, link, meta.session_id


@pytest.mark.asyncio
async def test_non_participant_rejected_with_default_member_only_rule(
    hub: Hub,
) -> None:
    a, b, m, link, session_id = await _spin_session_with_observer(hub, public=False)
    try:
        sub_id = await m.subscribe(session_id=session_id)
        await asyncio.sleep(0.05)
        # Mallory should get an ErrorFrame, and should NOT receive any
        # subsequent events. Our FakeClient ignores ErrorFrames silently,
        # so we verify by sending a text and checking the event queue.
        await a.send_text(
            session_id=session_id,
            content="private chat",
            recipient_id=b.actor_id,
        )
        await asyncio.sleep(0.05)
        assert m.event_queue.empty()
    finally:
        await a.stop()
        await b.stop()
        await m.stop()
        await link.close()


@pytest.mark.asyncio
async def test_non_participant_allowed_when_every_rule_is_public(hub: Hub) -> None:
    a, b, m, link, session_id = await _spin_session_with_observer(hub, public=True)
    try:
        await m.subscribe(session_id=session_id)
        await asyncio.sleep(0.05)
        await a.send_text(
            session_id=session_id,
            content="public broadcast",
            recipient_id=b.actor_id,
        )
        # Drain replayed system envelopes (invite, invite_ack, opened)
        # until we reach the user text envelope we care about.
        text_envelope = None
        for _ in range(10):
            received = await asyncio.wait_for(m.next_event(), timeout=0.5)
            if received.event_type == EV_TEXT:
                text_envelope = received
                break
        assert text_envelope is not None
        assert text_envelope.content() == "public broadcast"
    finally:
        await a.stop()
        await b.stop()
        await m.stop()
        await link.close()


@pytest.mark.asyncio
async def test_non_participant_denied_if_observer_rule_is_member_only(
    hub: Hub,
) -> None:
    """Even with permissive participants, an observer whose own rule is
    member-only cannot subscribe externally."""

    restrictive = Rule(
        access=AccessBlock(subscribe=SubscribeAccess(sessions=SUBSCRIBE_MEMBER_ONLY))
    )
    a, b, m, link, session_id = await _spin_session_with_observer(
        hub, public=True, observer_rule=restrictive
    )
    try:
        await m.subscribe(session_id=session_id)
        await asyncio.sleep(0.05)
        await a.send_text(
            session_id=session_id, content="hi", recipient_id=b.actor_id
        )
        await asyncio.sleep(0.05)
        assert m.event_queue.empty()
    finally:
        await a.stop()
        await b.stop()
        await m.stop()
        await link.close()


@pytest.mark.asyncio
async def test_most_restrictive_participant_vetoes_public_observer(
    hub: Hub,
) -> None:
    """Alice is public-within-hub, Bob is member-only → observer denied."""

    public = Rule(
        access=AccessBlock(subscribe=SubscribeAccess(sessions=SUBSCRIBE_HUB_PUBLIC))
    )
    member_only = Rule(
        access=AccessBlock(subscribe=SubscribeAccess(sessions=SUBSCRIBE_MEMBER_ONLY))
    )
    alice = await hub.register(ActorIdentity(name="alice"), rule=public)
    bob = await hub.register(ActorIdentity(name="bob"), rule=member_only)
    mallory = await hub.register(
        ActorIdentity(name="mallory"),
        rule=public,
    )
    link = attach_hub_to_link(hub)
    a = FakeClient(hub=hub, link=link, actor_id=alice.actor_id or "")
    b = FakeClient(
        hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only
    )
    m = FakeClient(hub=hub, link=link, actor_id=mallory.actor_id or "")
    await a.start()
    await b.start()
    await m.start()
    try:
        meta = await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type=SessionType.CONVERSATION,
            participant_names=["bob"],
            invite_ack_timeout_s=0.5,
        )
        await m.subscribe(session_id=meta.session_id)
        await asyncio.sleep(0.05)
        await a.send_text(
            session_id=meta.session_id, content="secret", recipient_id=b.actor_id
        )
        await asyncio.sleep(0.05)
        assert m.event_queue.empty()
    finally:
        await a.stop()
        await b.stop()
        await m.stop()
        await link.close()
