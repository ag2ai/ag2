# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Hub session lifecycle tests — handshake, adapter enforcement, WAL, close."""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    AccessBlock,
    ActorIdentity,
    Hub,
    LimitsBlock,
    Rule,
    SessionState,
    SessionType,
    SessionTypeAccess,
)
from autogen.beta.network.errors import (
    AccessDeniedError,
    InviteRejectedError,
    LimitExceededError,
    SessionClosedError,
)
from autogen.beta.network.hub import layout

from ._harness import FakeClient, attach_hub_to_link, auto_ack_and_reply, auto_ack_only


# ---------------------------------------------------------------------------
# Fixture: wire a hub + two harness clients
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def wired(hub: Hub):
    """Hub + Alice (ack-only) + Bob (auto-echo)."""

    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"))
    link = attach_hub_to_link(hub)
    a = FakeClient(hub=hub, link=link, actor_id=alice.actor_id or "", handler=auto_ack_only)
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_and_reply)
    await a.start()
    await b.start()
    yield hub, a, b
    await a.stop()
    await b.stop()
    await link.close()


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consulting_handshake_returns_active_session(wired) -> None:
    hub, alice, bob = wired
    metadata = await hub.create_session(
        creator_id=alice.actor_id,
        session_type=SessionType.CONSULTING,
        participant_names=["bob"],
    )
    assert metadata.state is SessionState.ACTIVE
    assert {p.actor_id for p in metadata.participants} == {alice.actor_id, bob.actor_id}


@pytest.mark.asyncio
async def test_session_expires_at_stamped_from_rule_ttl(wired) -> None:
    from datetime import datetime, timezone

    hub, alice, _bob = wired
    metadata = await hub.create_session(
        creator_id=alice.actor_id,
        session_type=SessionType.CONSULTING,
        participant_names=["bob"],
    )
    assert metadata.expires_at is not None
    # Default rule has session_ttl_default="2h" → 7200 seconds.
    created = datetime.strptime(metadata.created_at, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    expires = datetime.strptime(metadata.expires_at, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    assert (expires - created).total_seconds() == 7200


@pytest.mark.asyncio
async def test_session_expires_at_uses_creator_rule_ttl(hub: Hub) -> None:
    from datetime import datetime, timezone

    from autogen.beta.network.errors import UnknownActorError  # noqa: F401

    fast_rule = Rule(limits=LimitsBlock(session_ttl_default="30s"))
    alice = await hub.register(ActorIdentity(name="alice"), fast_rule)
    await hub.register(ActorIdentity(name="bob"))
    link = attach_hub_to_link(hub)
    a = FakeClient(hub=hub, link=link, actor_id=alice.actor_id or "", handler=auto_ack_only)
    bob_id = hub._name_to_id["bob"]  # type: ignore[attr-defined]
    b = FakeClient(hub=hub, link=link, actor_id=bob_id, handler=auto_ack_only)
    await a.start()
    await b.start()
    try:
        metadata = await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type=SessionType.CONSULTING,
            participant_names=["bob"],
        )
        created = datetime.strptime(metadata.created_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        expires = datetime.strptime(metadata.expires_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        assert (expires - created).total_seconds() == 30
    finally:
        await a.stop()
        await b.stop()
        await link.close()


@pytest.mark.asyncio
async def test_session_metadata_written_to_store(wired) -> None:
    hub, alice, _bob = wired
    metadata = await hub.create_session(
        creator_id=alice.actor_id,
        session_type=SessionType.CONSULTING,
        participant_names=["bob"],
    )
    raw = await hub._store.read(layout.session_metadata(metadata.session_id))
    assert raw is not None
    assert metadata.session_id in raw


@pytest.mark.asyncio
async def test_unknown_recipient_raises_before_handshake(wired) -> None:
    hub, alice, _bob = wired
    from autogen.beta.network.errors import UnknownActorError

    with pytest.raises(UnknownActorError):
        await hub.create_session(
            creator_id=alice.actor_id,
            session_type=SessionType.CONSULTING,
            participant_names=["ghost"],
        )


@pytest.mark.asyncio
async def test_handshake_times_out_when_recipient_ignores_invite(hub: Hub) -> None:
    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"))
    link = attach_hub_to_link(hub)
    # Bob does NOT auto-ack anything.
    a = FakeClient(hub=hub, link=link, actor_id=alice.actor_id or "")
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "")
    await a.start()
    await b.start()
    try:
        with pytest.raises(InviteRejectedError):
            await hub.create_session(
                creator_id=alice.actor_id,
                session_type=SessionType.CONSULTING,
                participant_names=["bob"],
                invite_ack_timeout_s=0.25,
            )
    finally:
        await a.stop()
        await b.stop()
        await link.close()


# ---------------------------------------------------------------------------
# Access + limit enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_access_denied_when_creator_rule_forbids_outbound() -> None:
    hub = Hub(MemoryKnowledgeStore())
    alice_rule = Rule(access=AccessBlock(outbound_to=["ag2:*:*"]))
    alice = await hub.register(ActorIdentity(name="alice"), rule=alice_rule)
    bob = await hub.register(ActorIdentity(name="bob"))  # bob doesn't match ag2:*:*
    link = attach_hub_to_link(hub)
    a = FakeClient(hub=hub, link=link, actor_id=alice.actor_id or "", handler=auto_ack_only)
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only)
    await a.start()
    await b.start()
    try:
        with pytest.raises(AccessDeniedError):
            await hub.create_session(
                creator_id=alice.actor_id,
                session_type=SessionType.CONSULTING,
                participant_names=["bob"],
            )
    finally:
        await a.stop()
        await b.stop()
        await link.close()


@pytest.mark.asyncio
async def test_access_denied_when_recipient_rule_forbids_inbound() -> None:
    hub = Hub(MemoryKnowledgeStore())
    bob_rule = Rule(access=AccessBlock(inbound_from=["ag2:*:*"]))
    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"), rule=bob_rule)
    link = attach_hub_to_link(hub)
    a = FakeClient(hub=hub, link=link, actor_id=alice.actor_id or "", handler=auto_ack_only)
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only)
    await a.start()
    await b.start()
    try:
        with pytest.raises(AccessDeniedError):
            await hub.create_session(
                creator_id=alice.actor_id,
                session_type=SessionType.CONSULTING,
                participant_names=["bob"],
            )
    finally:
        await a.stop()
        await b.stop()
        await link.close()


@pytest.mark.asyncio
async def test_access_denied_when_session_type_not_permitted() -> None:
    hub = Hub(MemoryKnowledgeStore())
    bob_rule = Rule(
        access=AccessBlock(
            session_types=SessionTypeAccess(
                accept=[SessionType.CONVERSATION.value]  # not consulting
            ),
        ),
    )
    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"), rule=bob_rule)
    link = attach_hub_to_link(hub)
    a = FakeClient(hub=hub, link=link, actor_id=alice.actor_id or "")
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "")
    await a.start()
    await b.start()
    try:
        with pytest.raises(AccessDeniedError):
            await hub.create_session(
                creator_id=alice.actor_id,
                session_type=SessionType.CONSULTING,
                participant_names=["bob"],
            )
    finally:
        await a.stop()
        await b.stop()
        await link.close()


@pytest.mark.asyncio
async def test_max_concurrent_sessions_limit_enforced() -> None:
    hub = Hub(MemoryKnowledgeStore())
    alice = await hub.register(
        ActorIdentity(name="alice"),
        rule=Rule(limits=LimitsBlock(max_concurrent_sessions=1)),
    )
    bob = await hub.register(ActorIdentity(name="bob"))
    link = attach_hub_to_link(hub)
    a = FakeClient(hub=hub, link=link, actor_id=alice.actor_id or "", handler=auto_ack_and_reply)
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only)
    await a.start()
    await b.start()
    try:
        # Conversation does not auto-close — holds the slot.
        await hub.create_session(
            creator_id=alice.actor_id,
            session_type=SessionType.CONVERSATION,
            participant_names=["bob"],
        )
        with pytest.raises(LimitExceededError):
            await hub.create_session(
                creator_id=alice.actor_id,
                session_type=SessionType.CONVERSATION,
                participant_names=["bob"],
            )
    finally:
        await a.stop()
        await b.stop()
        await link.close()


# ---------------------------------------------------------------------------
# Text exchange via the Hub's adapter path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consulting_exchange_closes_after_reply(wired) -> None:
    hub, alice, bob = wired
    metadata = await hub.create_session(
        creator_id=alice.actor_id,
        session_type=SessionType.CONSULTING,
        participant_names=["bob"],
    )

    await alice.send_text(
        session_id=metadata.session_id,
        content="explain X",
        recipient_id=bob.actor_id,
    )

    # Drain alice's notify queue to see Bob's echo reply.
    reply: str | None = None
    for _ in range(50):
        try:
            envelope = await asyncio.wait_for(alice.notify_queue.get(), timeout=0.25)
        except asyncio.TimeoutError:
            break
        if envelope.event_type == "ag2.msg.text":
            reply = envelope.content()
            break
    assert reply == "echo: explain X"

    # Session has closed by now.
    await asyncio.sleep(0.02)
    meta = await hub.get_session(metadata.session_id)
    assert meta.state is SessionState.CLOSED


@pytest.mark.asyncio
async def test_consulting_second_question_is_rejected(wired) -> None:
    hub, alice, bob = wired
    metadata = await hub.create_session(
        creator_id=alice.actor_id,
        session_type=SessionType.CONSULTING,
        participant_names=["bob"],
    )
    await alice.send_text(
        session_id=metadata.session_id,
        content="first",
        recipient_id=bob.actor_id,
    )
    # Let bob's echo land and session close.
    for _ in range(50):
        meta = await hub.get_session(metadata.session_id)
        if meta.state is SessionState.CLOSED:
            break
        await asyncio.sleep(0.01)
    # Now any further send should fail at the hub.
    with pytest.raises(SessionClosedError):
        await hub.post_envelope(
            _text_envelope(alice.actor_id, bob.actor_id, metadata.session_id, "second"),
        )


def _text_envelope(sender: str, recipient: str, session_id: str, content: str):
    from autogen.beta.network.envelope import Envelope

    return Envelope.text(
        session_id=session_id, sender_id=sender, content=content, recipient_id=recipient
    )


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversation_supports_multi_turn(wired) -> None:
    hub, alice, bob = wired
    metadata = await hub.create_session(
        creator_id=alice.actor_id,
        session_type=SessionType.CONVERSATION,
        participant_names=["bob"],
    )

    collected: list[str] = []

    async def drain_alice() -> None:
        for _ in range(3):
            envelope = await asyncio.wait_for(alice.notify_queue.get(), timeout=1.0)
            if envelope.event_type == "ag2.msg.text":
                collected.append(envelope.content())

    await alice.send_text(
        session_id=metadata.session_id, content="hi", recipient_id=bob.actor_id
    )
    # Bob auto-echoes, and we send another.
    await asyncio.sleep(0.05)
    await alice.send_text(
        session_id=metadata.session_id, content="again", recipient_id=bob.actor_id
    )
    await asyncio.sleep(0.05)
    await alice.send_text(
        session_id=metadata.session_id, content="bye", recipient_id=bob.actor_id
    )

    try:
        await asyncio.wait_for(drain_alice(), timeout=2.0)
    except asyncio.TimeoutError:
        pass

    assert collected == ["echo: hi", "echo: again", "echo: bye"]
    meta = await hub.get_session(metadata.session_id)
    assert meta.state is SessionState.ACTIVE  # still open

    await hub.close_session(metadata.session_id, requested_by=alice.actor_id)
    meta = await hub.get_session(metadata.session_id)
    assert meta.state is SessionState.CLOSED


# ---------------------------------------------------------------------------
# Notification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notification_delivers_once_and_closes() -> None:
    hub = Hub(MemoryKnowledgeStore())
    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"))
    link = attach_hub_to_link(hub)
    a = FakeClient(hub=hub, link=link, actor_id=alice.actor_id or "", handler=auto_ack_only)
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only)
    await a.start()
    await b.start()
    try:
        metadata = await hub.create_session(
            creator_id=alice.actor_id,
            session_type=SessionType.NOTIFICATION,
            participant_names=["bob"],
        )
        await a.send_text(
            session_id=metadata.session_id, content="fyi", recipient_id=bob.actor_id
        )
        text: str | None = None
        for _ in range(10):
            env = await asyncio.wait_for(b.notify_queue.get(), timeout=1.0)
            if env.event_type == "ag2.msg.text":
                text = env.content()
                break
        assert text == "fyi"
        # Session closed immediately.
        await asyncio.sleep(0.02)
        meta = await hub.get_session(metadata.session_id)
        assert meta.state is SessionState.CLOSED
    finally:
        await a.stop()
        await b.stop()
        await link.close()
