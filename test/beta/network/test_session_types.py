# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for SessionType enum and SessionMetadata round-trips."""

from __future__ import annotations

from autogen.beta.network.session_types import (
    Participant,
    ParticipantRole,
    SessionMetadata,
    SessionState,
    SessionType,
)


def test_session_type_enum_carries_all_six_types() -> None:
    names = {t.value for t in SessionType}
    assert names == {
        "notification",
        "broadcast",
        "consulting",
        "conversation",
        "discussion",
        "auction",
    }


def _sample() -> SessionMetadata:
    return SessionMetadata(
        session_id="01-ses",
        type=SessionType.CONSULTING,
        creator_id="01-alice",
        participants=[
            Participant(actor_id="01-alice", role=ParticipantRole.INITIATOR, joined_at="t0"),
            Participant(actor_id="01-bob", role=ParticipantRole.RESPONDENT, joined_at="t0"),
        ],
        created_at="t0",
        expires_at="t1",
        labels={"project": "research"},
    )


def test_session_metadata_roundtrip_via_dict() -> None:
    meta = _sample()
    restored = SessionMetadata.from_dict(meta.to_dict())
    assert restored.session_id == meta.session_id
    # Phase 2: SessionMetadata.type is a plain string on the wire and in
    # memory so operator-shipped adapters can register arbitrary type
    # names via ``Hub.register_adapter``. The value still compares equal
    # to the built-in ``SessionType`` enum member.
    assert restored.type == SessionType.CONSULTING
    assert restored.type == "consulting"
    assert isinstance(restored.type, str)
    assert not isinstance(restored.type, SessionType)
    assert restored.state is SessionState.PENDING
    assert restored.labels == {"project": "research"}
    assert len(restored.participants) == 2
    assert restored.participants[0].role is ParticipantRole.INITIATOR


def test_session_metadata_json_roundtrip() -> None:
    meta = _sample()
    restored = SessionMetadata.from_json(meta.to_json())
    assert restored.to_dict() == meta.to_dict()


def test_has_participant_and_participant_lookup() -> None:
    meta = _sample()
    assert meta.has_participant("01-alice")
    assert meta.has_participant("01-bob")
    assert not meta.has_participant("01-carol")
    p = meta.participant("01-alice")
    assert p is not None and p.role is ParticipantRole.INITIATOR
    assert meta.participant("01-carol") is None


def test_participant_ids_returns_stable_order() -> None:
    meta = _sample()
    assert meta.participant_ids() == ["01-alice", "01-bob"]


def test_session_state_default_is_pending() -> None:
    meta = SessionMetadata(
        session_id="s",
        type=SessionType.NOTIFICATION,
        creator_id="a",
        participants=[Participant(actor_id="b")],
    )
    assert meta.state is SessionState.PENDING


def test_session_metadata_copy_is_deep_for_mutable_fields() -> None:
    meta = _sample()
    copy = meta.copy()
    copy.labels["project"] = "other"
    copy.participants.append(Participant(actor_id="01-carol"))
    assert meta.labels == {"project": "research"}
    assert meta.participant_ids() == ["01-alice", "01-bob"]
