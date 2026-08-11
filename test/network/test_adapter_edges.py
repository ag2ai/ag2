# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Adapter validate_send / validate_create / fold edge cases.

Each adapter (consulting, conversation, discussion, workflow) declares
its own protocol shape. The hub gates substantive sends through
``adapter.validate_send`` under a per-channel lock; rejection raises
``ProtocolError`` and the WAL must remain untouched. Hydrate re-folds
the WAL through ``adapter.fold`` so determinism is required (replaying
the same envelope sequence must produce the same state).

These tests exercise the rejection paths and fold determinism that
the existing per-adapter integration tests don't cover directly.
"""

import pytest

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_TEXT,
    Envelope,
    Hub,
    ProtocolError,
)
from ag2.network.adapters.consulting import ConsultingAdapter, ConsultingState
from ag2.network.adapters.conversation import ConversationAdapter
from ag2.network.adapters.discussion import DiscussionAdapter, DiscussionState
from ag2.network.channel import (
    ChannelMetadata,
    ChannelState,
    Participant,
    ParticipantRole,
)

from ._helpers import ScriptedConfig


def _agent(name: str) -> Agent:
    return Agent(name=name, config=ScriptedConfig())


def _make_metadata(
    *,
    manifest,
    creator: str,
    participants: list[tuple[str, ParticipantRole]],
    knobs: dict | None = None,
) -> ChannelMetadata:
    parts = [
        Participant(agent_id=aid, role=role, order=i, joined_at="2026-01-01T00:00:00+00:00")
        for i, (aid, role) in enumerate(participants)
    ]
    return ChannelMetadata(
        channel_id="s1",
        manifest=manifest,
        creator_id=creator,
        participants=parts,
        state=ChannelState.ACTIVE,
        created_at="2026-01-01T00:00:00+00:00",
        knobs=knobs or {},
    )


class TestConsultingAdapter:
    """ConsultingAdapter unit-level edge cases."""

    def test_validate_create_rejects_three_participants(self) -> None:
        adapter = ConsultingAdapter()
        meta = _make_metadata(
            manifest=adapter.manifest,
            creator="alice",
            participants=[
                ("alice", ParticipantRole.INITIATOR),
                ("bob", ParticipantRole.RESPONDENT),
                ("carol", ParticipantRole.PARTICIPANT),
            ],
        )
        with pytest.raises(ProtocolError, match="exactly 2"):
            adapter.validate_create(meta)

    def test_validate_create_rejects_missing_respondent(self) -> None:
        adapter = ConsultingAdapter()
        meta = _make_metadata(
            manifest=adapter.manifest,
            creator="alice",
            participants=[
                ("alice", ParticipantRole.INITIATOR),
                ("bob", ParticipantRole.PARTICIPANT),
            ],
        )
        with pytest.raises(ProtocolError, match="respondent"):
            adapter.validate_create(meta)

    def test_validate_send_rejects_respondent_first(self) -> None:
        """Initiator must send the first substantive envelope."""
        adapter = ConsultingAdapter()
        meta = _make_metadata(
            manifest=adapter.manifest,
            creator="alice",
            participants=[
                ("alice", ParticipantRole.INITIATOR),
                ("bob", ParticipantRole.RESPONDENT),
            ],
        )
        state = adapter.initial_state(meta)
        envelope = Envelope(
            channel_id="s1",
            sender_id="bob",
            audience=["alice"],
            event_type=EV_TEXT,
            event_data={"text": "first?"},
        )
        with pytest.raises(ProtocolError, match="initiator"):
            adapter.validate_send(meta, envelope, state)

    def test_validate_send_rejects_after_both_replied(self) -> None:
        adapter = ConsultingAdapter()
        meta = _make_metadata(
            manifest=adapter.manifest,
            creator="alice",
            participants=[
                ("alice", ParticipantRole.INITIATOR),
                ("bob", ParticipantRole.RESPONDENT),
            ],
        )
        state = ConsultingState(initiator_sent=True, respondent_replied=True)
        envelope = Envelope(
            channel_id="s1",
            sender_id="alice",
            audience=["bob"],
            event_type=EV_TEXT,
            event_data={"text": "third send"},
        )
        with pytest.raises(ProtocolError, match="already complete"):
            adapter.validate_send(meta, envelope, state)

    def test_fold_is_deterministic_under_replay(self) -> None:
        """Replaying the same WAL through initial_state + fold reproduces state."""
        adapter = ConsultingAdapter()
        meta = _make_metadata(
            manifest=adapter.manifest,
            creator="alice",
            participants=[
                ("alice", ParticipantRole.INITIATOR),
                ("bob", ParticipantRole.RESPONDENT),
            ],
        )
        envelopes = [
            Envelope(
                channel_id="s1",
                sender_id="alice",
                audience=["bob"],
                event_type=EV_TEXT,
                event_data={"text": "q"},
            ),
            Envelope(
                channel_id="s1",
                sender_id="bob",
                audience=["alice"],
                event_type=EV_TEXT,
                event_data={"text": "a"},
            ),
        ]
        # Stamp envelope_ids so fold can record them.
        envelopes[0].envelope_id = "env-1"
        envelopes[1].envelope_id = "env-2"

        s1 = adapter.initial_state(meta)
        for env in envelopes:
            s1 = adapter.fold(env, s1)

        s2 = adapter.initial_state(meta)
        for env in envelopes:
            s2 = adapter.fold(env, s2)

        assert s1 == s2

    def test_on_accepted_returns_closed_after_both_replied(self) -> None:
        adapter = ConsultingAdapter()
        meta = _make_metadata(
            manifest=adapter.manifest,
            creator="alice",
            participants=[
                ("alice", ParticipantRole.INITIATOR),
                ("bob", ParticipantRole.RESPONDENT),
            ],
        )
        state = ConsultingState(initiator_sent=True, respondent_replied=True)
        envelope = Envelope(
            channel_id="s1",
            sender_id="bob",
            audience=["alice"],
            event_type=EV_TEXT,
            event_data={"text": "reply"},
        )
        result = adapter.on_accepted(meta, envelope, state)
        assert result.next_state == ChannelState.CLOSED
        assert result.auto_close_reason == "consulting_complete"


class TestConversationAdapter:
    def test_validate_send_rejects_non_participant(self) -> None:
        adapter = ConversationAdapter()
        meta = _make_metadata(
            manifest=adapter.manifest,
            creator="alice",
            participants=[
                ("alice", ParticipantRole.INITIATOR),
                ("bob", ParticipantRole.RESPONDENT),
            ],
        )
        envelope = Envelope(
            channel_id="s1",
            sender_id="eve",
            audience=["alice"],
            event_type=EV_TEXT,
            event_data={"text": "x"},
        )
        with pytest.raises(ProtocolError, match="only accepts sends from participants"):
            adapter.validate_send(meta, envelope, adapter.initial_state(meta))

    def test_fold_advances_turn_count_and_speaker(self) -> None:
        adapter = ConversationAdapter()
        meta = _make_metadata(
            manifest=adapter.manifest,
            creator="alice",
            participants=[
                ("alice", ParticipantRole.INITIATOR),
                ("bob", ParticipantRole.RESPONDENT),
            ],
        )
        state = adapter.initial_state(meta)
        envelopes = [
            Envelope(
                channel_id="s1", sender_id="alice", audience=["bob"], event_type=EV_TEXT, event_data={"text": "1"}
            ),
            Envelope(
                channel_id="s1", sender_id="bob", audience=["alice"], event_type=EV_TEXT, event_data={"text": "2"}
            ),
            Envelope(
                channel_id="s1", sender_id="alice", audience=["bob"], event_type=EV_TEXT, event_data={"text": "3"}
            ),
        ]
        for i, env in enumerate(envelopes, 1):
            env.envelope_id = f"e{i}"
            state = adapter.fold(env, state)

        assert state.turn_count == 3
        assert state.last_speaker_id == "alice"
        assert state.last_envelope_id == "e3"

    def test_fold_ignores_protocol_envelopes(self) -> None:
        from ag2.network.envelope import EV_CHANNEL_OPENED

        adapter = ConversationAdapter()
        meta = _make_metadata(
            manifest=adapter.manifest,
            creator="alice",
            participants=[
                ("alice", ParticipantRole.INITIATOR),
                ("bob", ParticipantRole.RESPONDENT),
            ],
        )
        state = adapter.initial_state(meta)
        opened = Envelope(
            channel_id="s1",
            sender_id="alice",
            audience=["alice", "bob"],
            event_type=EV_CHANNEL_OPENED,
            event_data={"channel_id": "s1"},
        )
        opened.envelope_id = "p1"
        new = adapter.fold(opened, state)
        assert new == state  # untouched


class TestDiscussionAdapter:
    def test_validate_create_rejects_unknown_ordering(self) -> None:
        adapter = DiscussionAdapter()
        meta = _make_metadata(
            manifest=adapter.manifest,
            creator="alice",
            participants=[
                ("alice", ParticipantRole.INITIATOR),
                ("bob", ParticipantRole.PARTICIPANT),
            ],
            knobs={"ordering": "weighted"},  # not supported
        )
        with pytest.raises(ProtocolError, match="ordering"):
            adapter.validate_create(meta)

    def test_validate_create_rejects_solo(self) -> None:
        adapter = DiscussionAdapter()
        meta = _make_metadata(
            manifest=adapter.manifest,
            creator="alice",
            participants=[("alice", ParticipantRole.INITIATOR)],
        )
        with pytest.raises(ProtocolError, match="at least 2"):
            adapter.validate_create(meta)

    def test_validate_send_rejects_out_of_turn(self) -> None:
        adapter = DiscussionAdapter()
        meta = _make_metadata(
            manifest=adapter.manifest,
            creator="alice",
            participants=[
                ("alice", ParticipantRole.INITIATOR),
                ("bob", ParticipantRole.PARTICIPANT),
                ("carol", ParticipantRole.PARTICIPANT),
            ],
        )
        state = adapter.initial_state(meta)
        # alice is expected_next_speaker; bob trying to speak is out-of-turn
        envelope = Envelope(
            channel_id="s1",
            sender_id="bob",
            audience=None,
            event_type=EV_TEXT,
            event_data={"text": "jump in"},
        )
        with pytest.raises(ProtocolError, match="expects 'alice' to speak"):
            adapter.validate_send(meta, envelope, state)

    def test_round_robin_rotation_wraps_after_n_turns(self) -> None:
        """After N turns where N == participant count, expected_next_speaker
        cycles back to the initiator."""
        adapter = DiscussionAdapter()
        meta = _make_metadata(
            manifest=adapter.manifest,
            creator="alice",
            participants=[
                ("alice", ParticipantRole.INITIATOR),
                ("bob", ParticipantRole.PARTICIPANT),
                ("carol", ParticipantRole.PARTICIPANT),
            ],
        )
        state = adapter.initial_state(meta)
        speakers = ["alice", "bob", "carol"]
        for i, speaker in enumerate(speakers, 1):
            assert state.expected_next_speaker == speaker
            env = Envelope(
                channel_id="s1",
                sender_id=speaker,
                audience=None,
                event_type=EV_TEXT,
                event_data={"text": f"turn-{i}"},
            )
            env.envelope_id = f"e{i}"
            state = adapter.fold(env, state)
        # After carol, rotation wraps back to alice.
        assert state.expected_next_speaker == "alice"
        assert state.turn_count == 3

    def test_fold_with_unknown_sender_does_not_crash(self) -> None:
        """Sender not in participant_order — fold returns unchanged state.

        validate_send normally gates this, but fold must be defensive
        because hydrate replays the WAL deterministically and any
        in-memory ParticipantOrder change between create and hydrate
        could leave a stale envelope.
        """
        adapter = DiscussionAdapter()
        meta = _make_metadata(
            manifest=adapter.manifest,
            creator="alice",
            participants=[
                ("alice", ParticipantRole.INITIATOR),
                ("bob", ParticipantRole.PARTICIPANT),
            ],
        )
        state = adapter.initial_state(meta)
        ghost_env = Envelope(
            channel_id="s1",
            sender_id="ghost",
            audience=None,
            event_type=EV_TEXT,
            event_data={"text": "x"},
        )
        ghost_env.envelope_id = "g1"
        new = adapter.fold(ghost_env, state)
        assert new == state


@pytest.mark.asyncio
async def test_validate_send_rejection_does_not_append_to_wal() -> None:
    """A rejected send must NOT touch the WAL.

    post_envelope holds the WAL lock and runs validate_send BEFORE
    appending. A rejected envelope must leave the WAL exactly as it was.
    """
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))
    await hub.register(_agent("carol"))

    channel = await alice.open(type="discussion", target=["bob", "carol"])
    pre_wal = await hub.read_wal(channel.channel_id)
    pre_count = len(pre_wal)

    # Bob tries to speak out-of-turn (alice is initiator, alice goes first).
    envelope = Envelope(
        channel_id=channel.channel_id,
        sender_id=bob.agent_id,
        audience=None,
        event_type=EV_TEXT,
        event_data={"text": "out of turn"},
    )
    with pytest.raises(ProtocolError):
        await bob.send_envelope(envelope)

    post_wal = await hub.read_wal(channel.channel_id)
    assert len(post_wal) == pre_count
    # And no envelope from bob with the rejected text.
    assert all(e.event_data.get("text") != "out of turn" for e in post_wal)

    await hub.close()


@pytest.mark.asyncio
async def test_hydrate_refolds_discussion_state_deterministically() -> None:
    """Close hub, reopen → refolded DiscussionState matches what fold
    would produce live."""
    import tempfile

    from ag2.knowledge import DiskKnowledgeStore

    with tempfile.TemporaryDirectory() as tmp:
        store = DiskKnowledgeStore(tmp)
        hub1 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
        alice = await hub1.register(_agent("alice"))
        bob = await hub1.register(_agent("bob"))
        carol = await hub1.register(_agent("carol"))

        channel = await alice.open(type="discussion", target=["bob", "carol"])
        sid = channel.channel_id

        await channel.send("a-1", audience=None)
        # bob's default handler does NOT auto-respond (ScriptedConfig
        # returns ""), so we send manually for each speaker in turn.
        await bob._hub_client.post_envelope(
            Envelope(
                channel_id=sid,
                sender_id=bob.agent_id,
                audience=None,
                event_type=EV_TEXT,
                event_data={"text": "b-1"},
            )
        )
        await carol._hub_client.post_envelope(
            Envelope(
                channel_id=sid,
                sender_id=carol.agent_id,
                audience=None,
                event_type=EV_TEXT,
                event_data={"text": "c-1"},
            )
        )

        live_state: DiscussionState = hub1._adapter_states[sid]
        live_summary = (
            live_state.expected_next_speaker,
            live_state.last_speaker_id,
            live_state.turn_count,
            live_state.participant_order,
        )

        await hub1.close()

        hub2 = await Hub.open(DiskKnowledgeStore(tmp), ttl_sweep_interval=0, expectation_sweep_interval=0)
        rehydrated: DiscussionState = hub2._adapter_states[sid]
        rehydrated_summary = (
            rehydrated.expected_next_speaker,
            rehydrated.last_speaker_id,
            rehydrated.turn_count,
            rehydrated.participant_order,
        )

        assert live_summary == rehydrated_summary
        await hub2.close()
