# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Consulting adapter integration tests.

Covers:

* Single-recipient consulting handshake (invite → auto-ack → ACTIVE).
* Full LLM-driven turn: initiator sends prompt → respondent's notify
  handler runs ``Agent.ask`` → respondent replies → adapter
  ``on_accepted`` returns CLOSING → session auto-closes with
  ``close_reason="consulting_complete"``.
* Adapter rejects out-of-order sends (``ProtocolError``).
* ``Hub.hydrate()`` re-folds an active session's WAL deterministically
  through ``adapter.fold`` so the in-memory ``AdapterState`` matches
  what's on disk.

This suite uses ``TestConfig`` so it runs offline and fast.
"""

import pytest

from autogen.beta import Agent
from autogen.beta.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from autogen.beta.network import (
    EV_SESSION_CLOSED,
    EV_SESSION_INVITE,
    EV_SESSION_INVITE_ACK,
    EV_SESSION_OPENED,
    EV_TEXT,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)
from autogen.beta.network.adapters.consulting import ConsultingAdapter, ConsultingState
from autogen.beta.network.errors import ProtocolError
from autogen.beta.network.session import SessionState
from autogen.beta.testing import TestConfig


def _agent(name: str, *events: object) -> Agent:
    return Agent(name=name, config=TestConfig(*events))


@pytest.mark.asyncio
async def test_consulting_handshake_transitions_to_active() -> None:
    """alice.open → hub posts invite → bob auto-acks → session ACTIVE."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    session = await alice.open(type="consulting", target="bob")
    assert session.state == SessionState.ACTIVE
    assert len(session.metadata.participants) == 2
    assert session.metadata.creator_id == alice.agent_id
    assert session.metadata.pending_acks == []

    wal = await hub.read_wal(session.session_id)
    event_types = [e.event_type for e in wal]
    assert EV_SESSION_INVITE in event_types
    assert EV_SESSION_INVITE_ACK in event_types
    assert EV_SESSION_OPENED in event_types

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_consulting_full_flow_auto_closes() -> None:
    """Initiator sends prompt → respondent replies via default handler → CLOSED."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(
        _agent("bob", "Hi, this is bob's reply."),
        Passport(name="bob"),
        Resume(),
    )

    session = await alice.open(type="consulting", target="bob")
    await session.send("Hello bob, can you help?", audience=[bob.agent_id])

    # Wait for session close to propagate to alice's inbox.
    close_envelope = await alice.wait_for_session_event(
        session_id=session.session_id,
        predicate=lambda e: e.event_type == EV_SESSION_CLOSED,
        timeout=5.0,
    )
    assert close_envelope.event_data.get("reason") == "consulting_complete"

    final = await hub.get_session(session.session_id)
    assert final.state == SessionState.CLOSED
    assert final.close_reason == "consulting_complete"

    wal = await hub.read_wal(session.session_id)
    text_events = [e for e in wal if e.event_type == EV_TEXT]
    assert len(text_events) == 2
    assert text_events[0].sender_id == alice.agent_id
    assert text_events[0].event_data["text"] == "Hello bob, can you help?"
    assert text_events[1].sender_id == bob.agent_id
    assert text_events[1].event_data["text"] == "Hi, this is bob's reply."

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_consulting_rejects_out_of_order_send() -> None:
    """Respondent cannot send before initiator's first envelope."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    session = await alice.open(type="consulting", target="bob")

    # Bob tries to send first — adapter should reject.
    bad = Envelope(
        session_id=session.session_id,
        sender_id=bob.agent_id,
        audience=[alice.agent_id],
        event_type=EV_TEXT,
        event_data={"text": "I'm jumping the queue"},
    )
    with pytest.raises(ProtocolError, match="initiator"):
        await hub.post_envelope(bad)

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_consulting_rejects_send_after_complete() -> None:
    """Adapter rejects any send after both turns have happened."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(
        _agent("bob", "ok"),
        Passport(name="bob"),
        Resume(),
    )

    session = await alice.open(type="consulting", target="bob")
    await session.send("the question", audience=[bob.agent_id])
    await alice.wait_for_session_event(
        session_id=session.session_id,
        predicate=lambda e: e.event_type == EV_SESSION_CLOSED,
        timeout=5.0,
    )

    # Session is now closed — any further post should fail.
    extra = Envelope(
        session_id=session.session_id,
        sender_id=alice.agent_id,
        audience=[bob.agent_id],
        event_type=EV_TEXT,
        event_data={"text": "follow-up"},
    )
    with pytest.raises(ProtocolError):
        await hub.post_envelope(extra)

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_hub_hydrate_refolds_active_session(tmp_path) -> None:
    """Close hub mid-flight, reopen, verify adapter state survives."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0)
    link1 = LocalLink(hub1)

    alice_hc = HubClient(link1, hub=hub1)
    bob_hc = HubClient(link1, hub=hub1)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    session = await alice.open(type="consulting", target="bob")
    await session.send("the question", audience=[bob.agent_id])

    # Don't wait for bob's reply — close hub mid-flight.
    await alice_hc.close()
    await bob_hc.close()
    await hub1.close()

    # Reopen with a fresh hub against the same store.
    store2 = DiskKnowledgeStore(str(tmp_path))
    hub2 = await Hub.open(store2, ttl_sweep_interval=0)

    # Session metadata reloaded.
    refreshed = await hub2.get_session(session.session_id)
    assert refreshed.session_id == session.session_id
    assert refreshed.manifest.type == "consulting"

    # Adapter state cache rebuilt by re-folding the WAL.
    state = hub2._adapter_states[session.session_id]
    assert isinstance(state, ConsultingState)
    # Alice has sent her prompt; bob has not replied yet.
    assert state.initiator_sent is True
    assert state.respondent_replied is False

    await hub2.close()


@pytest.mark.asyncio
async def test_consulting_invite_timeout_when_no_handler() -> None:
    """If bob has no auto-ack handler, hub times out and fails the open."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, invite_ack_timeout=0.2)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    # Bob registers with attach_plugin=False to skip the default handler.
    bob = await bob_hc.register(
        _agent("bob"),
        Passport(name="bob"),
        Resume(),
        attach_plugin=False,
    )
    # And clear the auto-installed default handler — bob ignores invites.
    bob.on_envelope(_silent_handler)

    with pytest.raises(ProtocolError, match="ack timeout"):
        await alice.open(type="consulting", target="bob")

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


async def _silent_handler(_envelope: Envelope) -> None:
    """No-op handler — used to test invite timeout behaviour."""


@pytest.mark.asyncio
async def test_default_consulting_adapter_registered_on_open() -> None:
    """``Hub.open`` auto-registers ConsultingAdapter for ``consulting@v1``."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    adapter = hub._adapters.get(("consulting", 1))
    assert isinstance(adapter, ConsultingAdapter)
    await hub.close()
