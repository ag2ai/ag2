# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Discussion adapter (round_robin) + multi-party handshake tests.

Covers:

* 5-way handshake: hub posts ``EV_CHANNEL_INVITE`` to every invitee,
  collects all-or-nothing acks, transitions to ``ACTIVE``.
* Round-robin speaker rotation: ``state.expected_next_speaker`` cycles
  through ``metadata.participants`` ``order`` after each accepted
  ``EV_TEXT``.
* ``validate_send`` rejects out-of-turn sends.
* Partial reject fails the channel (the handshake is all-or-nothing).
* ``Hub.hydrate()`` re-folds the WAL through ``DiscussionAdapter.fold``
  and recovers ``expected_next_speaker``.
* Hub auto-registers the adapter on ``Hub.open``.
* ``validate_create`` rejects unsupported ordering modes.

This suite uses ``ScriptedConfig`` so it runs offline and fast.
"""

import contextlib
from collections.abc import Awaitable, Callable

import pytest

from ag2 import Agent
from ag2.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_INVITE,
    EV_CHANNEL_INVITE_ACK,
    EV_CHANNEL_INVITE_REJECT,
    EV_CHANNEL_OPENED,
    EV_TEXT,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)
from ag2.network.adapters.discussion import (
    DISCUSSION_TYPE,
    ORDERING_ROUND_ROBIN,
    DiscussionAdapter,
    DiscussionState,
)
from ag2.network.channel import ChannelState
from ag2.network.client.agent_client import AgentClient
from ag2.network.errors import ProtocolError
from ag2.testing import TestConfig

from ._helpers import ScriptedConfig, wait_for_text_count


def _agent(name: str, *events: object) -> Agent:
    return Agent(name=name, config=TestConfig(*events))


def _scripted_agent(name: str, *replies: str) -> Agent:
    return Agent(name=name, config=ScriptedConfig(*replies))


def _make_rejecter(client: AgentClient) -> Callable[[Envelope], Awaitable[None]]:
    """Build a notify handler that explicitly rejects every invite.

    Used to drive the partial-reject path. Defined at module level
    (not inside a test) so the closure is created once at registration.
    """

    async def _reject(envelope: Envelope) -> None:
        if envelope.event_type != EV_CHANNEL_INVITE:
            return
        rejection = Envelope(
            channel_id=envelope.channel_id,
            sender_id=client.agent_id,
            audience=None,
            event_type=EV_CHANNEL_INVITE_REJECT,
            event_data={"reason": "not interested"},
            causation_id=envelope.envelope_id,
        )
        with contextlib.suppress(Exception):
            await client.send_envelope(rejection)

    return _reject


@pytest.mark.asyncio
async def test_default_discussion_adapter_registered_on_open() -> None:
    """``Hub.open`` auto-registers DiscussionAdapter for ``discussion@v1``."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    adapter = hub._adapters.get((DISCUSSION_TYPE, 1))
    assert isinstance(adapter, DiscussionAdapter)
    await hub.close()


@pytest.mark.asyncio
async def test_discussion_validate_create_rejects_unsupported_ordering() -> None:
    """V1 only ships round_robin; dynamic / static raise at create time."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    with pytest.raises(ProtocolError, match="ordering"):
        await hub.create_channel(
            creator_id=alice.agent_id,
            manifest_type=DISCUSSION_TYPE,
            participants=[bob.agent_id],
            knobs={"ordering": "dynamic"},
        )

    await hub.close()


@pytest.mark.asyncio
async def test_discussion_5_way_handshake_transitions_to_active() -> None:
    """5 invitees all auto-ack; channel activates with full participant list."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    names = ["alice", "bob", "carol", "dave", "erin"]
    clients: list[AgentClient] = []
    for name in names:
        hc = HubClient(link, hub=hub)
        client = await hc.register(_agent(name), Passport(name=name), Resume())
        clients.append(client)

    alice = clients[0]
    targets = [c.agent_id for c in clients[1:]]
    channel = await alice.open(
        type=DISCUSSION_TYPE,
        target=targets,
        knobs={"ordering": ORDERING_ROUND_ROBIN},
    )

    assert channel.state == ChannelState.ACTIVE
    assert channel.metadata.pending_acks == []
    assert channel.metadata.rejected_by == []
    assert {p.agent_id for p in channel.metadata.participants} == {c.agent_id for c in clients}

    wal = await hub.read_wal(channel.channel_id)
    invite_count = sum(1 for e in wal if e.event_type == EV_CHANNEL_INVITE)
    ack_count = sum(1 for e in wal if e.event_type == EV_CHANNEL_INVITE_ACK)
    assert invite_count == 4  # one invite per non-creator participant
    assert ack_count == 4
    assert any(e.event_type == EV_CHANNEL_OPENED for e in wal)

    state = hub._adapter_states[channel.channel_id]
    assert isinstance(state, DiscussionState)
    assert state.expected_next_speaker == alice.agent_id
    assert state.participant_order[0] == alice.agent_id

    for hc in [c._hub_client for c in clients]:
        await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_discussion_round_robin_advances_through_participants() -> None:
    """Manual sends in turn order succeed; out-of-turn raises ProtocolError."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    # Use plain TestConfig agents and skip the LLM auto-handler — we
    # only exercise the adapter by posting envelopes manually.
    alice = await hub.register(_agent("alice"), attach_plugin=False)
    bob = await hub.register(_agent("bob"), attach_plugin=False)
    carol = await hub.register(_agent("carol"), attach_plugin=False)

    # Auto-ack invites (handlers are off — install minimal ackers).
    for client in (bob, carol):
        client.on_envelope(_make_auto_acker(client))

    channel = await alice.open(
        type=DISCUSSION_TYPE,
        target=[bob.agent_id, carol.agent_id],
        knobs={"ordering": ORDERING_ROUND_ROBIN},
    )
    assert channel.state == ChannelState.ACTIVE

    # Round 1: alice → bob → carol.
    await channel.send("alice 1")
    state = hub._adapter_states[channel.channel_id]
    assert state.expected_next_speaker == bob.agent_id

    bob_envelope = Envelope(
        channel_id=channel.channel_id,
        sender_id=bob.agent_id,
        audience=None,
        event_type=EV_TEXT,
        event_data={"text": "bob 1"},
    )
    await hub.post_envelope(bob_envelope)
    state = hub._adapter_states[channel.channel_id]
    assert state.expected_next_speaker == carol.agent_id

    carol_envelope = Envelope(
        channel_id=channel.channel_id,
        sender_id=carol.agent_id,
        audience=None,
        event_type=EV_TEXT,
        event_data={"text": "carol 1"},
    )
    await hub.post_envelope(carol_envelope)
    state = hub._adapter_states[channel.channel_id]
    assert state.expected_next_speaker == alice.agent_id  # cycle back
    assert state.turn_count == 3
    assert state.last_speaker_id == carol.agent_id

    await hub.close()


@pytest.mark.asyncio
async def test_discussion_rejects_out_of_turn_send() -> None:
    """Sending out of round-robin order raises ProtocolError before WAL write."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))
    carol = await hub.register(_agent("carol"))

    channel = await alice.open(
        type=DISCUSSION_TYPE,
        target=[bob.agent_id, carol.agent_id],
        knobs={"ordering": ORDERING_ROUND_ROBIN},
    )

    # Carol jumps the queue before alice has spoken.
    bad = Envelope(
        channel_id=channel.channel_id,
        sender_id=carol.agent_id,
        audience=None,
        event_type=EV_TEXT,
        event_data={"text": "I'm jumping the queue"},
    )
    with pytest.raises(ProtocolError, match="expects"):
        await hub.post_envelope(bad)

    await hub.close()


@pytest.mark.asyncio
async def test_discussion_partial_reject_fails_channel() -> None:
    """V1 all-or-nothing: any reject during handshake closes the channel."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, invite_ack_timeout=2.0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))
    # Carol rejects every invite.
    carol = await hub.register(_agent("carol"), attach_plugin=False)
    carol.on_envelope(_make_rejecter(carol))

    with pytest.raises(ProtocolError, match="rejected"):
        await alice.open(
            type=DISCUSSION_TYPE,
            target=[bob.agent_id, carol.agent_id],
            knobs={"ordering": ORDERING_ROUND_ROBIN},
        )

    await hub.close()


@pytest.mark.asyncio
async def test_discussion_hydrate_refolds_round_robin_state(tmp_path) -> None:
    """Re-opening hub mid-discussion recovers expected_next_speaker."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0)

    alice = await hub1.register(_agent("alice"), attach_plugin=False)
    bob = await hub1.register(_agent("bob"), attach_plugin=False)
    carol = await hub1.register(_agent("carol"), attach_plugin=False)
    bob.on_envelope(_make_auto_acker(bob))
    carol.on_envelope(_make_auto_acker(carol))

    channel = await alice.open(
        type=DISCUSSION_TYPE,
        target=[bob.agent_id, carol.agent_id],
        knobs={"ordering": ORDERING_ROUND_ROBIN},
    )
    await channel.send("alice opens the floor")

    # Mid-discussion (bob is up next) → tear down the hub.
    await hub1.close()

    # Reopen against the same store.
    store2 = DiskKnowledgeStore(str(tmp_path))
    hub2 = await Hub.open(store2, ttl_sweep_interval=0)

    state = hub2._adapter_states[channel.channel_id]
    assert isinstance(state, DiscussionState)
    assert state.expected_next_speaker == bob.agent_id
    assert state.last_speaker_id == alice.agent_id
    assert state.turn_count == 1
    assert state.participant_order == [alice.agent_id, bob.agent_id, carol.agent_id]

    await hub2.close()


@pytest.mark.asyncio
async def test_discussion_llm_driven_round_robin_3_way() -> None:
    """End-to-end LLM-driven 3-way round-robin: each agent's handler fires
    only when the adapter's validate_send would accept their reply."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    # alice's manual send opens turn 1; her LLM speaks again on her turn 2.
    alice = await hub.register(_scripted_agent("alice", "alice 2"))
    bob = await hub.register(_scripted_agent("bob", "bob 1"))
    carol = await hub.register(_scripted_agent("carol", "carol 1"))

    channel = await alice.open(
        type=DISCUSSION_TYPE,
        target=[bob.agent_id, carol.agent_id],
        knobs={"ordering": ORDERING_ROUND_ROBIN},
    )
    await channel.send("alice 1")

    # 4 EV_TEXT envelopes total: alice 1 / bob 1 / carol 1 / alice 2.
    # On alice's turn 3, bob has no script left → empty body → halt.
    wal = await wait_for_text_count(hub, channel.channel_id, expected=4)
    text_envelopes = [e for e in wal if e.event_type == EV_TEXT]
    assert [e.event_data["text"] for e in text_envelopes] == [
        "alice 1",
        "bob 1",
        "carol 1",
        "alice 2",
    ]
    assert [e.sender_id for e in text_envelopes] == [
        alice.agent_id,
        bob.agent_id,
        carol.agent_id,
        alice.agent_id,
    ]

    state = hub._adapter_states[channel.channel_id]
    assert state.expected_next_speaker == bob.agent_id
    assert state.turn_count == 4

    await hub.close()


def _make_auto_acker(client: AgentClient) -> Callable[[Envelope], Awaitable[None]]:
    """Minimal handler that auto-acks invites and ignores everything else.

    Used by tests that disable the default handler (which also runs
    LLM logic) but still need the multi-party handshake to complete.
    """

    async def _ack(envelope: Envelope) -> None:
        if envelope.event_type != EV_CHANNEL_INVITE:
            return
        ack = Envelope(
            channel_id=envelope.channel_id,
            sender_id=client.agent_id,
            audience=None,
            event_type=EV_CHANNEL_INVITE_ACK,
            event_data={"channel_id": envelope.channel_id},
            causation_id=envelope.envelope_id,
        )
        with contextlib.suppress(Exception):
            await client.send_envelope(ack)

    return _ack
