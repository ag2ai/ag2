# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Conversation adapter + windowed summary view tests.

Covers:

* ``ConversationAdapter`` 1+1 bidirectional handshake.
* Multi-turn LLM-driven back-and-forth (no auto-close).
* Explicit ``channel.close()`` ends the conversation.
* ``validate_send`` rejects sends from non-participants.
* Adapter survives ``Hub.hydrate()`` mid-conversation.
* ``WindowedSummary`` view: short history passes through; long history
  prepends a ``CompactionSummary``; respects ``audience`` visibility.

This suite uses ``TestConfig`` so it runs offline and fast.
"""

import pytest

from ag2 import Agent
from ag2.compact import CompactionSummary
from ag2.events import ModelMessage, ModelRequest, TextInput
from ag2.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_INVITE,
    EV_CHANNEL_INVITE_ACK,
    EV_CHANNEL_OPENED,
    EV_TEXT,
    Envelope,
    Hub,
)
from ag2.network.adapters.base import default_render_envelope
from ag2.network.adapters.conversation import (
    CONVERSATION_TYPE,
    ConversationAdapter,
    ConversationState,
)
from ag2.network.channel import (
    ChannelMetadata,
    ChannelState,
    Participant,
    ParticipantRole,
)
from ag2.network.errors import ProtocolError
from ag2.network.views.builtin import WindowedSummary
from ag2.testing import TestConfig

from ._helpers import ScriptedConfig, wait_for_text_count


def _agent(name: str, *events: object) -> Agent:
    return Agent(name=name, config=TestConfig(*events))


def _scripted_agent(name: str, *replies: str) -> Agent:
    return Agent(name=name, config=ScriptedConfig(*replies))


@pytest.mark.asyncio
async def test_conversation_handshake_transitions_to_active() -> None:
    """alice.open(type=conversation) → bob auto-acks → ACTIVE."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    channel = await alice.open(type=CONVERSATION_TYPE, target="bob")
    assert channel.state == ChannelState.ACTIVE
    assert channel.metadata.creator_id == alice.agent_id
    assert {p.agent_id for p in channel.metadata.participants} == {
        alice.agent_id,
        bob.agent_id,
    }
    assert channel.metadata.pending_acks == []

    wal = await hub.read_wal(channel.channel_id)
    event_types = [e.event_type for e in wal]
    assert EV_CHANNEL_INVITE in event_types
    assert EV_CHANNEL_INVITE_ACK in event_types
    assert EV_CHANNEL_OPENED in event_types

    await hub.close()


@pytest.mark.asyncio
async def test_conversation_back_and_forth_multi_turn() -> None:
    """Bidirectional LLM-driven exchange runs until one side returns empty."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    # alice replies to bob's first answer with a follow-up, then halts.
    alice = await hub.register(_scripted_agent("alice", "follow up question"))
    # bob answers alice twice; second reply is empty → halts the chain.
    bob = await hub.register(_scripted_agent("bob", "initial reply", "second answer"))

    channel = await alice.open(type=CONVERSATION_TYPE, target="bob")
    await channel.send("hello bob")

    # Expect 4 EV_TEXT envelopes total: alice → bob → alice → bob.
    wal = await wait_for_text_count(hub, channel.channel_id, expected=4)
    text_events = [e for e in wal if e.event_type == EV_TEXT]
    assert [e.event_data["text"] for e in text_events] == [
        "hello bob",
        "initial reply",
        "follow up question",
        "second answer",
    ]
    assert [e.sender_id for e in text_events] == [
        alice.agent_id,
        bob.agent_id,
        alice.agent_id,
        bob.agent_id,
    ]

    # Adapter state reflects the final speaker + count.
    state = hub._adapter_states[channel.channel_id]
    assert isinstance(state, ConversationState)
    assert state.turn_count == 4
    assert state.last_speaker_id == bob.agent_id

    await hub.close()


@pytest.mark.asyncio
async def test_conversation_explicit_close_terminates() -> None:
    """``channel.close()`` transitions to ``CLOSED`` regardless of activity."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    channel = await alice.open(type=CONVERSATION_TYPE, target="bob")
    closed = await channel.close(reason="done")

    assert closed.state == ChannelState.CLOSED
    assert closed.close_reason == "done"

    # Subsequent sends rejected — channel is terminal.
    extra = Envelope(
        channel_id=channel.channel_id,
        sender_id=alice.agent_id,
        audience=[bob.agent_id],
        event_type=EV_TEXT,
        event_data={"text": "after close"},
    )
    with pytest.raises(ProtocolError):
        await hub.post_envelope(extra)

    await hub.close()


@pytest.mark.asyncio
async def test_conversation_rejects_send_from_non_participant() -> None:
    """``validate_send`` blocks sends from agents not in ``participants``."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))
    eve = await hub.register(_agent("eve"))

    channel = await alice.open(type=CONVERSATION_TYPE, target="bob")

    intruder = Envelope(
        channel_id=channel.channel_id,
        sender_id=eve.agent_id,
        audience=[alice.agent_id],
        event_type=EV_TEXT,
        event_data={"text": "I'm gatecrashing"},
    )
    with pytest.raises(ProtocolError, match="participants"):
        await hub.post_envelope(intruder)

    await hub.close()


@pytest.mark.asyncio
async def test_conversation_hydrate_refolds_active_channel(tmp_path) -> None:
    """Re-opening the hub mid-conversation rebuilds adapter state from WAL."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0)

    alice = await hub1.register(_agent("alice"))
    await hub1.register(_agent("bob"))

    channel = await alice.open(type=CONVERSATION_TYPE, target="bob")
    await channel.send("first turn")
    # No LLM responses configured — bob's handler exhausts; conversation
    # quiesces with one EV_TEXT in WAL.
    await wait_for_text_count(hub1, channel.channel_id, expected=1)

    await hub1.close()

    store2 = DiskKnowledgeStore(str(tmp_path))
    hub2 = await Hub.open(store2, ttl_sweep_interval=0)

    refreshed = await hub2.get_channel(channel.channel_id)
    assert refreshed.manifest.type == CONVERSATION_TYPE
    assert refreshed.state == ChannelState.ACTIVE

    state = hub2._adapter_states[channel.channel_id]
    assert isinstance(state, ConversationState)
    assert state.turn_count == 1
    assert state.last_speaker_id == alice.agent_id

    await hub2.close()


@pytest.mark.asyncio
async def test_default_conversation_adapter_registered_on_open() -> None:
    """``Hub.open`` auto-registers ConversationAdapter for ``conversation@v1``."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    adapter = hub._adapters.get((CONVERSATION_TYPE, 1))
    assert isinstance(adapter, ConversationAdapter)
    await hub.close()


@pytest.mark.asyncio
async def test_windowed_summary_short_history_passes_through() -> None:
    """When WAL ≤ recent_n, projection is full transcript without summary."""
    metadata = _two_party_metadata("alice", "bob")
    wal = [
        _text_envelope("alice", "bob", "hi"),
        _text_envelope("bob", "alice", "hello"),
        _text_envelope("alice", "bob", "how's it going"),
    ]

    view = WindowedSummary(recent_n=10)
    projection = await view.project(
        wal,
        participant_id="bob",
        channel=metadata,
        render_envelope=default_render_envelope,
    )

    assert projection == [
        ModelRequest([TextInput("hi")]),
        ModelMessage("hello"),
        ModelRequest([TextInput("how's it going")]),
    ]


@pytest.mark.asyncio
async def test_windowed_summary_long_history_prepends_compaction_summary() -> None:
    """When WAL > recent_n, older slice collapses to a CompactionSummary."""
    metadata = _two_party_metadata("alice", "bob")
    wal = [_text_envelope("alice", "bob", f"msg {i}") for i in range(8)]

    view = WindowedSummary(recent_n=3)
    projection = await view.project(
        wal,
        participant_id="bob",
        channel=metadata,
        render_envelope=default_render_envelope,
    )

    assert len(projection) == 4  # 1 summary + 3 recent
    head = projection[0]
    assert isinstance(head, CompactionSummary)
    assert head.event_count == 5
    assert "alice" in head.summary
    # Tail preserves recent envelopes verbatim.
    assert projection[1:] == [
        ModelRequest([TextInput("msg 5")]),
        ModelRequest([TextInput("msg 6")]),
        ModelRequest([TextInput("msg 7")]),
    ]


@pytest.mark.asyncio
async def test_windowed_summary_respects_audience_visibility() -> None:
    """Envelopes targeted at others (audience) are filtered out per participant."""
    metadata = _three_party_metadata("alice", "bob", "carol")
    wal = [
        _text_envelope("alice", None, "broadcast"),
        _text_envelope("alice", "bob", "private to bob", audience=["bob"]),
        _text_envelope("bob", None, "bob's reply"),
    ]

    view = WindowedSummary(recent_n=10)
    carol_projection = await view.project(
        wal,
        participant_id="carol",
        channel=metadata,
        render_envelope=default_render_envelope,
    )

    # Carol cannot see alice's private message to bob.
    assert carol_projection == [
        ModelRequest([TextInput("broadcast")]),
        ModelRequest([TextInput("bob's reply")]),
    ]


def _text_envelope(
    sender: str,
    recipient: str | None,
    text: str,
    *,
    audience: list[str] | None = None,
) -> Envelope:
    if audience is None and recipient is not None:
        audience = [recipient]
    return Envelope(
        envelope_id=f"env-{sender}-{text}",
        channel_id="channel-1",
        sender_id=sender,
        audience=audience,
        event_type=EV_TEXT,
        event_data={"text": text},
    )


def _two_party_metadata(initiator: str, respondent: str) -> ChannelMetadata:
    return ChannelMetadata(
        channel_id="channel-1",
        manifest=ConversationAdapter().manifest,
        creator_id=initiator,
        participants=[
            Participant(agent_id=initiator, role=ParticipantRole.INITIATOR, order=0),
            Participant(agent_id=respondent, role=ParticipantRole.RESPONDENT, order=1),
        ],
        state=ChannelState.ACTIVE,
        created_at="2026-05-03T00:00:00Z",
    )


def _three_party_metadata(initiator: str, *others: str) -> ChannelMetadata:
    participants = [
        Participant(agent_id=initiator, role=ParticipantRole.INITIATOR, order=0),
    ]
    for i, name in enumerate(others, start=1):
        participants.append(Participant(agent_id=name, role=ParticipantRole.PARTICIPANT, order=i))
    return ChannelMetadata(
        channel_id="channel-1",
        manifest=ConversationAdapter().manifest,
        creator_id=initiator,
        participants=participants,
        state=ChannelState.ACTIVE,
        created_at="2026-05-03T00:00:00Z",
    )
