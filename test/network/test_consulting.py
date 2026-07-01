# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Consulting adapter integration tests.

Covers:

* Single-recipient consulting handshake (invite → auto-ack → ACTIVE).
* Full LLM-driven turn: initiator sends prompt → respondent's notify
  handler runs ``Agent.ask`` → respondent replies → adapter
  ``on_accepted`` returns CLOSING → channel auto-closes with
  ``close_reason="consulting_complete"``.
* Adapter rejects out-of-order sends (``ProtocolError``).
* ``Hub.hydrate()`` re-folds an active channel's WAL deterministically
  through ``adapter.fold`` so the in-memory ``AdapterState`` matches
  what's on disk.

This suite uses ``TestConfig`` so it runs offline and fast.
"""

import pytest

from ag2 import Agent
from ag2.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_CLOSED,
    EV_CHANNEL_INVITE,
    EV_CHANNEL_INVITE_ACK,
    EV_CHANNEL_OPENED,
    EV_TEXT,
    Envelope,
    Hub,
)
from ag2.network.adapters.consulting import ConsultingAdapter, ConsultingState
from ag2.network.channel import ChannelState
from ag2.network.errors import ProtocolError
from ag2.testing import TestConfig


def _agent(name: str, *events: object) -> Agent:
    return Agent(name=name, config=TestConfig(*events))


@pytest.mark.asyncio
async def test_consulting_handshake_transitions_to_active() -> None:
    """alice.open → hub posts invite → bob auto-acks → channel ACTIVE."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))

    channel = await alice.open(type="consulting", target="bob")
    assert channel.state == ChannelState.ACTIVE
    assert len(channel.metadata.participants) == 2
    assert channel.metadata.creator_id == alice.agent_id
    assert channel.metadata.pending_acks == []

    wal = await hub.read_wal(channel.channel_id)
    event_types = [e.event_type for e in wal]
    assert EV_CHANNEL_INVITE in event_types
    assert EV_CHANNEL_INVITE_ACK in event_types
    assert EV_CHANNEL_OPENED in event_types

    await hub.close()


@pytest.mark.asyncio
async def test_consulting_full_flow_auto_closes() -> None:
    """Initiator sends prompt → respondent replies via default handler → CLOSED."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(
        _agent("bob", "Hi, this is bob's reply."),
    )

    channel = await alice.open(type="consulting", target="bob")
    await channel.send("Hello bob, can you help?", audience=[bob.agent_id])

    # Wait for channel close to propagate to alice's inbox.
    close_envelope = await alice.wait_for_channel_event(
        channel_id=channel.channel_id,
        predicate=lambda e: e.event_type == EV_CHANNEL_CLOSED,
        timeout=5.0,
    )
    assert close_envelope.event_data.get("reason") == "consulting_complete"

    final = await hub.get_channel(channel.channel_id)
    assert final.state == ChannelState.CLOSED
    assert final.close_reason == "consulting_complete"

    wal = await hub.read_wal(channel.channel_id)
    text_events = [e for e in wal if e.event_type == EV_TEXT]
    assert len(text_events) == 2
    assert text_events[0].sender_id == alice.agent_id
    assert text_events[0].event_data["text"] == "Hello bob, can you help?"
    assert text_events[1].sender_id == bob.agent_id
    assert text_events[1].event_data["text"] == "Hi, this is bob's reply."

    await hub.close()


@pytest.mark.asyncio
async def test_consulting_rejects_out_of_order_send() -> None:
    """Respondent cannot send before initiator's first envelope."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    channel = await alice.open(type="consulting", target="bob")

    # Bob tries to send first — adapter should reject.
    bad = Envelope(
        channel_id=channel.channel_id,
        sender_id=bob.agent_id,
        audience=[alice.agent_id],
        event_type=EV_TEXT,
        event_data={"text": "I'm jumping the queue"},
    )
    with pytest.raises(ProtocolError, match="initiator"):
        await hub.post_envelope(bad)

    await hub.close()


@pytest.mark.asyncio
async def test_consulting_rejects_send_after_complete() -> None:
    """Adapter rejects any send after both turns have happened."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(
        _agent("bob", "ok"),
    )

    channel = await alice.open(type="consulting", target="bob")
    await channel.send("the question", audience=[bob.agent_id])
    await alice.wait_for_channel_event(
        channel_id=channel.channel_id,
        predicate=lambda e: e.event_type == EV_CHANNEL_CLOSED,
        timeout=5.0,
    )

    # Channel is now closed — any further post should fail.
    extra = Envelope(
        channel_id=channel.channel_id,
        sender_id=alice.agent_id,
        audience=[bob.agent_id],
        event_type=EV_TEXT,
        event_data={"text": "follow-up"},
    )
    with pytest.raises(ProtocolError):
        await hub.post_envelope(extra)

    await hub.close()


@pytest.mark.asyncio
async def test_hub_hydrate_refolds_active_channel(tmp_path) -> None:
    """Close hub mid-flight, reopen, verify adapter state survives."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0)

    alice = await hub1.register(_agent("alice"))
    bob = await hub1.register(_agent("bob"))

    channel = await alice.open(type="consulting", target="bob")
    await channel.send("the question", audience=[bob.agent_id])

    # Don't wait for bob's reply — close hub mid-flight.
    await hub1.close()

    # Reopen with a fresh hub against the same store.
    store2 = DiskKnowledgeStore(str(tmp_path))
    hub2 = await Hub.open(store2, ttl_sweep_interval=0)

    # Channel metadata reloaded.
    refreshed = await hub2.get_channel(channel.channel_id)
    assert refreshed.channel_id == channel.channel_id
    assert refreshed.manifest.type == "consulting"

    # Adapter state cache rebuilt by re-folding the WAL.
    state = hub2._adapter_states[channel.channel_id]
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

    alice = await hub.register(_agent("alice"))
    # Bob registers with attach_plugin=False to skip the default handler.
    bob = await hub.register(
        _agent("bob"),
        attach_plugin=False,
    )
    # And clear the auto-installed default handler — bob ignores invites.
    bob.on_envelope(_silent_handler)

    with pytest.raises(ProtocolError, match="ack timeout"):
        await alice.open(type="consulting", target="bob")

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


@pytest.mark.asyncio
async def test_delegate_tool_end_to_end() -> None:
    """End-to-end: Alice's LLM uses ``delegate`` to consult Bob.

    Alice's TestConfig delivers a ``delegate`` tool call followed by a
    final user-facing response. Bob's TestConfig delivers a single
    string reply. The full chain — open consulting, invite/ack, send
    prompt, run Bob's LLM via the default handler, return Bob's reply
    to Alice, Alice's second LLM call to incorporate the reply — runs
    without any real LLM calls.
    """
    from ag2.events import ToolCallEvent

    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    alice_agent = Agent(
        name="alice",
        config=TestConfig(
            [
                ToolCallEvent(
                    name="delegate",
                    arguments='{"target": "bob", "prompt": "what is 2+2?"}',
                ),
            ],
            "The answer is 4.",
        ),
    )
    bob_agent = Agent(name="bob", config=TestConfig("4"))

    await hub.register(alice_agent)
    await hub.register(bob_agent)

    reply = await alice_agent.ask("ask bob to do math for me")

    assert reply.body == "The answer is 4."

    await hub.close()
