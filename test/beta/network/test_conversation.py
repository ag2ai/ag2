# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Conversation adapter + windowed summary view tests.

Covers:

* ``ConversationAdapter`` 1+1 bidirectional handshake.
* Multi-turn LLM-driven back-and-forth (no auto-close).
* Explicit ``session.close()`` ends the conversation.
* ``validate_send`` rejects sends from non-participants.
* Adapter survives ``Hub.hydrate()`` mid-conversation.
* ``WindowedSummary`` view: short history passes through; long history
  prepends a ``CompactionSummary``; respects ``audience`` visibility.

This suite uses ``TestConfig`` so it runs offline and fast.
"""

import pytest

from autogen.beta import Agent
from autogen.beta.compact import CompactionSummary
from autogen.beta.events import ModelMessage, ModelRequest, TextInput
from autogen.beta.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from autogen.beta.network import (
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
from autogen.beta.network.adapters.base import default_render_envelope
from autogen.beta.network.adapters.conversation import (
    CONVERSATION_TYPE,
    ConversationAdapter,
    ConversationState,
)
from autogen.beta.network.errors import ProtocolError
from autogen.beta.network.session import (
    Participant,
    ParticipantRole,
    SessionMetadata,
    SessionState,
)
from autogen.beta.network.views.builtin import WindowedSummary
from autogen.beta.testing import TestConfig

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
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    session = await alice.open(type=CONVERSATION_TYPE, target="bob")
    assert session.state == SessionState.ACTIVE
    assert session.metadata.creator_id == alice.agent_id
    assert {p.agent_id for p in session.metadata.participants} == {
        alice.agent_id,
        bob.agent_id,
    }
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
async def test_conversation_back_and_forth_multi_turn() -> None:
    """Bidirectional LLM-driven exchange runs until one side returns empty."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    # alice replies to bob's first answer with a follow-up, then halts.
    alice = await alice_hc.register(
        _scripted_agent("alice", "follow up question"),
        Passport(name="alice"),
        Resume(),
    )
    # bob answers alice twice; second reply is empty → halts the chain.
    bob = await bob_hc.register(
        _scripted_agent("bob", "initial reply", "second answer"),
        Passport(name="bob"),
        Resume(),
    )

    session = await alice.open(type=CONVERSATION_TYPE, target="bob")
    await session.send("hello bob")

    # Expect 4 EV_TEXT envelopes total: alice → bob → alice → bob.
    wal = await wait_for_text_count(hub, session.session_id, expected=4)
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
    state = hub._adapter_states[session.session_id]
    assert isinstance(state, ConversationState)
    assert state.turn_count == 4
    assert state.last_speaker_id == bob.agent_id

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_conversation_explicit_close_terminates() -> None:
    """``session.close()`` transitions to ``CLOSED`` regardless of activity."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    session = await alice.open(type=CONVERSATION_TYPE, target="bob")
    closed = await session.close(reason="done")

    assert closed.state == SessionState.CLOSED
    assert closed.close_reason == "done"

    # Subsequent sends rejected — session is terminal.
    extra = Envelope(
        session_id=session.session_id,
        sender_id=alice.agent_id,
        audience=[bob.agent_id],
        event_type=EV_TEXT,
        event_data={"text": "after close"},
    )
    with pytest.raises(ProtocolError):
        await hub.post_envelope(extra)

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_conversation_rejects_send_from_non_participant() -> None:
    """``validate_send`` blocks sends from agents not in ``participants``."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    eve_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())
    eve = await eve_hc.register(_agent("eve"), Passport(name="eve"), Resume())

    session = await alice.open(type=CONVERSATION_TYPE, target="bob")

    intruder = Envelope(
        session_id=session.session_id,
        sender_id=eve.agent_id,
        audience=[alice.agent_id],
        event_type=EV_TEXT,
        event_data={"text": "I'm gatecrashing"},
    )
    with pytest.raises(ProtocolError, match="participants"):
        await hub.post_envelope(intruder)

    await alice_hc.close()
    await bob_hc.close()
    await eve_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_conversation_hydrate_refolds_active_session(tmp_path) -> None:
    """Re-opening the hub mid-conversation rebuilds adapter state from WAL."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0)
    link1 = LocalLink(hub1)

    alice_hc = HubClient(link1, hub=hub1)
    bob_hc = HubClient(link1, hub=hub1)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    session = await alice.open(type=CONVERSATION_TYPE, target="bob")
    await session.send("first turn")
    # No LLM responses configured — bob's handler exhausts; conversation
    # quiesces with one EV_TEXT in WAL.
    await wait_for_text_count(hub1, session.session_id, expected=1)

    await alice_hc.close()
    await bob_hc.close()
    await hub1.close()

    store2 = DiskKnowledgeStore(str(tmp_path))
    hub2 = await Hub.open(store2, ttl_sweep_interval=0)

    refreshed = await hub2.get_session(session.session_id)
    assert refreshed.manifest.type == CONVERSATION_TYPE
    assert refreshed.state == SessionState.ACTIVE

    state = hub2._adapter_states[session.session_id]
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
        session=metadata,
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
        session=metadata,
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
        session=metadata,
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
        session_id="session-1",
        sender_id=sender,
        audience=audience,
        event_type=EV_TEXT,
        event_data={"text": text},
    )


def _two_party_metadata(initiator: str, respondent: str) -> SessionMetadata:
    return SessionMetadata(
        session_id="session-1",
        manifest=ConversationAdapter().manifest,
        creator_id=initiator,
        participants=[
            Participant(agent_id=initiator, role=ParticipantRole.INITIATOR, order=0),
            Participant(agent_id=respondent, role=ParticipantRole.RESPONDENT, order=1),
        ],
        state=SessionState.ACTIVE,
        created_at="2026-05-03T00:00:00Z",
    )


def _three_party_metadata(initiator: str, *others: str) -> SessionMetadata:
    participants = [
        Participant(agent_id=initiator, role=ParticipantRole.INITIATOR, order=0),
    ]
    for i, name in enumerate(others, start=1):
        participants.append(Participant(agent_id=name, role=ParticipantRole.PARTICIPANT, order=i))
    return SessionMetadata(
        session_id="session-1",
        manifest=ConversationAdapter().manifest,
        creator_id=initiator,
        participants=participants,
        state=SessionState.ACTIVE,
        created_at="2026-05-03T00:00:00Z",
    )
