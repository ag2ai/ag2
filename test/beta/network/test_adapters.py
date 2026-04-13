# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure tests for session adapters (no Hub, no store)."""

from __future__ import annotations

import pytest

from autogen.beta.network.adapters import (
    ConsultingAdapter,
    ConversationAdapter,
    NotificationAdapter,
)
from autogen.beta.network.envelope import EV_SESSION_INVITE, EV_TEXT, Envelope
from autogen.beta.network.errors import SessionTypeError
from autogen.beta.network.session_types import (
    Participant,
    ParticipantRole,
    SessionMetadata,
    SessionState,
    SessionType,
)


def _consulting_meta() -> SessionMetadata:
    return SessionMetadata(
        session_id="s1",
        type=SessionType.CONSULTING,
        creator_id="alice",
        participants=[
            Participant(actor_id="alice", role=ParticipantRole.INITIATOR),
            Participant(actor_id="bob", role=ParticipantRole.RESPONDENT),
        ],
        state=SessionState.ACTIVE,
    )


def _text(sender: str, recipient: str, content: str) -> Envelope:
    return Envelope.text(
        session_id="s1",
        sender_id=sender,
        content=content,
        recipient_id=recipient,
    )


# ---------------------------------------------------------------------------
# Consulting
# ---------------------------------------------------------------------------


def test_consulting_validate_create_requires_two_participants() -> None:
    adapter = ConsultingAdapter()
    meta = SessionMetadata(
        session_id="s",
        type=SessionType.CONSULTING,
        creator_id="a",
        participants=[Participant(actor_id="a", role=ParticipantRole.INITIATOR)],
    )
    with pytest.raises(SessionTypeError):
        adapter.validate_create(meta)


def test_consulting_validate_create_requires_initiator_and_respondent() -> None:
    adapter = ConsultingAdapter()
    meta = SessionMetadata(
        session_id="s",
        type=SessionType.CONSULTING,
        creator_id="a",
        participants=[
            Participant(actor_id="a", role=ParticipantRole.INITIATOR),
            Participant(actor_id="b", role=ParticipantRole.INITIATOR),
        ],
    )
    with pytest.raises(SessionTypeError):
        adapter.validate_create(meta)


def test_consulting_rejects_text_from_respondent_first() -> None:
    adapter = ConsultingAdapter()
    meta = _consulting_meta()
    env = _text("bob", "alice", "hi")
    with pytest.raises(SessionTypeError):
        adapter.validate_send(meta, env, [])


def test_consulting_accepts_first_text_from_initiator() -> None:
    adapter = ConsultingAdapter()
    meta = _consulting_meta()
    env = _text("alice", "bob", "explain")
    adapter.validate_send(meta, env, [])  # no error


def test_consulting_second_send_must_come_from_respondent() -> None:
    adapter = ConsultingAdapter()
    meta = _consulting_meta()
    prior = [_text("alice", "bob", "question")]
    with pytest.raises(SessionTypeError):
        adapter.validate_send(meta, _text("alice", "bob", "again"), prior)
    adapter.validate_send(meta, _text("bob", "alice", "answer"), prior)


def test_consulting_closes_after_reply() -> None:
    adapter = ConsultingAdapter()
    meta = _consulting_meta()
    prior = [_text("alice", "bob", "q")]
    result = adapter.on_accepted(meta, _text("bob", "alice", "a"), prior)
    assert result.next_state is SessionState.CLOSED
    assert result.close_reason == "consulting_done"


def test_consulting_third_message_is_rejected() -> None:
    adapter = ConsultingAdapter()
    meta = _consulting_meta()
    prior = [_text("alice", "bob", "q"), _text("bob", "alice", "a")]
    with pytest.raises(SessionTypeError):
        adapter.validate_send(meta, _text("alice", "bob", "again"), prior)


def test_consulting_rejects_non_text_event() -> None:
    adapter = ConsultingAdapter()
    meta = _consulting_meta()
    env = Envelope(
        session_id="s1",
        sender_id="alice",
        recipient_id="bob",
        event_type=EV_SESSION_INVITE,
        event_data={"reason": "x"},
    )
    with pytest.raises(SessionTypeError):
        adapter.validate_send(meta, env, [])


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


def test_conversation_allows_either_direction() -> None:
    adapter = ConversationAdapter()
    meta = SessionMetadata(
        session_id="s",
        type=SessionType.CONVERSATION,
        creator_id="alice",
        participants=[
            Participant(actor_id="alice", role=ParticipantRole.INITIATOR),
            Participant(actor_id="bob", role=ParticipantRole.RESPONDENT),
        ],
        state=SessionState.ACTIVE,
    )
    # Alice → Bob, Bob → Alice, Alice → Bob, ... all allowed.
    prior: list[Envelope] = []
    for sender, recipient in [("alice", "bob"), ("bob", "alice"), ("alice", "bob")]:
        env = Envelope.text(session_id="s", sender_id=sender, content="hi", recipient_id=recipient)
        adapter.validate_send(meta, env, prior)
        prior.append(env)


def test_conversation_rejects_non_participant_sender() -> None:
    adapter = ConversationAdapter()
    meta = SessionMetadata(
        session_id="s",
        type=SessionType.CONVERSATION,
        creator_id="alice",
        participants=[
            Participant(actor_id="alice", role=ParticipantRole.INITIATOR),
            Participant(actor_id="bob", role=ParticipantRole.RESPONDENT),
        ],
        state=SessionState.ACTIVE,
    )
    env = Envelope.text(session_id="s", sender_id="carol", content="hi", recipient_id="alice")
    with pytest.raises(SessionTypeError):
        adapter.validate_send(meta, env, [])


def test_conversation_on_accepted_transitions_pending_to_active() -> None:
    adapter = ConversationAdapter()
    meta = SessionMetadata(
        session_id="s",
        type=SessionType.CONVERSATION,
        creator_id="alice",
        participants=[
            Participant(actor_id="alice", role=ParticipantRole.INITIATOR),
            Participant(actor_id="bob", role=ParticipantRole.RESPONDENT),
        ],
        state=SessionState.PENDING,
    )
    env = Envelope.text(session_id="s", sender_id="alice", content="hi", recipient_id="bob")
    result = adapter.on_accepted(meta, env, [])
    assert result.next_state is SessionState.ACTIVE


# ---------------------------------------------------------------------------
# Notification
# ---------------------------------------------------------------------------


def test_notification_accepts_single_message_and_closes() -> None:
    adapter = NotificationAdapter()
    meta = SessionMetadata(
        session_id="s",
        type=SessionType.NOTIFICATION,
        creator_id="alice",
        participants=[
            Participant(actor_id="alice", role=ParticipantRole.INITIATOR),
            Participant(actor_id="bob", role=ParticipantRole.RESPONDENT),
        ],
        state=SessionState.ACTIVE,
    )
    env = Envelope.text(session_id="s", sender_id="alice", content="fyi", recipient_id="bob")
    adapter.validate_send(meta, env, [])
    result = adapter.on_accepted(meta, env, [])
    assert result.next_state is SessionState.CLOSED
    assert result.close_reason == "notification_done"


def test_notification_rejects_second_message() -> None:
    adapter = NotificationAdapter()
    meta = SessionMetadata(
        session_id="s",
        type=SessionType.NOTIFICATION,
        creator_id="alice",
        participants=[
            Participant(actor_id="alice", role=ParticipantRole.INITIATOR),
            Participant(actor_id="bob", role=ParticipantRole.RESPONDENT),
        ],
        state=SessionState.ACTIVE,
    )
    prior = [Envelope.text(session_id="s", sender_id="alice", content="one", recipient_id="bob")]
    second = Envelope.text(session_id="s", sender_id="alice", content="two", recipient_id="bob")
    with pytest.raises(SessionTypeError):
        adapter.validate_send(meta, second, prior)


def test_notification_rejects_recipient_reply() -> None:
    adapter = NotificationAdapter()
    meta = SessionMetadata(
        session_id="s",
        type=SessionType.NOTIFICATION,
        creator_id="alice",
        participants=[
            Participant(actor_id="alice", role=ParticipantRole.INITIATOR),
            Participant(actor_id="bob", role=ParticipantRole.RESPONDENT),
        ],
        state=SessionState.ACTIVE,
    )
    reply = Envelope.text(session_id="s", sender_id="bob", content="ok", recipient_id="alice")
    with pytest.raises(SessionTypeError):
        adapter.validate_send(meta, reply, [])
