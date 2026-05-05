# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``ConversationAdapter`` — bidirectional multi-turn 1+1 session.

Two participants: initiator and respondent. Either side may send at
any turn. The session runs until an explicit ``close()`` call or TTL
expiry — there is no auto-close based on message content.

Default expectations:
* ``max_silence(3600s, audit)`` — light enforcement; conversations may
  legitimately span hours of idle time.

Task envelopes (``ag2.task.*``) and session-protocol envelopes
(``ag2.session.*``) bypass conversation's send rule the same way
consulting does — they're hub bookkeeping or task lifecycle observed
on the session.
"""

from dataclasses import dataclass

from ..envelope import (
    EV_SESSION_CLOSED,
    EV_SESSION_EXPIRED,
    EV_SESSION_INVITE,
    EV_SESSION_INVITE_ACK,
    EV_SESSION_INVITE_REJECT,
    EV_SESSION_OPENED,
    EV_TEXT,
    Envelope,
)
from ..errors import ProtocolError
from ..session import (
    Expectation,
    ParticipantRole,
    ParticipantSchema,
    SessionManifest,
    SessionMetadata,
)
from ..views.base import ViewPolicy
from ..views.builtin import WindowedSummary
from .base import AdapterResult

__all__ = ("CONVERSATION_TYPE", "ConversationAdapter", "ConversationState")


CONVERSATION_TYPE = "conversation"
_DEFAULT_RECENT_N = 10


_SESSION_PROTOCOL_EVENTS: frozenset[str] = frozenset({
    EV_SESSION_INVITE,
    EV_SESSION_INVITE_ACK,
    EV_SESSION_INVITE_REJECT,
    EV_SESSION_OPENED,
    EV_SESSION_CLOSED,
    EV_SESSION_EXPIRED,
})


def _is_session_protocol_event(envelope: Envelope) -> bool:
    return envelope.event_type in _SESSION_PROTOCOL_EVENTS


def _is_task_event(envelope: Envelope) -> bool:
    return envelope.event_type.startswith("ag2.task.")


@dataclass(slots=True)
class ConversationState:
    """Folded state for a conversation session.

    Conversations have no per-turn ordering constraint, so the state
    only tracks bookkeeping for views and observability — the adapter
    never returns a ``next_state`` from ``on_accepted``.
    """

    turn_count: int = 0
    last_speaker_id: str | None = None
    last_envelope_id: str | None = None


class ConversationAdapter:
    """Bidirectional multi-turn 1+1 session.

    Knobs: none. Participants: exactly 2 (initiator + respondent).
    Default view: :class:`WindowedSummary(recent_n=10)` — bounded
    prompt size at any turn count.
    """

    def __init__(self) -> None:
        self.manifest = SessionManifest(
            type=CONVERSATION_TYPE,
            version=1,
            participants=ParticipantSchema(
                min=2,
                max=2,
                roles=[ParticipantRole.INITIATOR.value, ParticipantRole.RESPONDENT.value],
            ),
            knobs_schema={},
            default_view_policy=WindowedSummary.name,
            expectations=[
                Expectation(
                    name="max_silence",
                    on_violation="audit",
                    params={"seconds": 3600},
                ),
            ],
        )

    # ── Adapter Protocol ────────────────────────────────────────────────────

    def initial_state(self, metadata: SessionMetadata) -> ConversationState:
        return ConversationState()

    def fold(self, envelope: Envelope, state: ConversationState) -> ConversationState:
        if _is_session_protocol_event(envelope) or _is_task_event(envelope):
            return state
        if envelope.event_type != EV_TEXT:
            return state
        return ConversationState(
            turn_count=state.turn_count + 1,
            last_speaker_id=envelope.sender_id,
            last_envelope_id=envelope.envelope_id,
        )

    def validate_create(self, metadata: SessionMetadata) -> None:
        roles = {p.role for p in metadata.participants}
        if ParticipantRole.INITIATOR not in roles:
            raise ProtocolError("conversation requires exactly one initiator")
        if ParticipantRole.RESPONDENT not in roles:
            raise ProtocolError("conversation requires exactly one respondent")
        if len(metadata.participants) != 2:
            raise ProtocolError(f"conversation requires exactly 2 participants, got {len(metadata.participants)}")

    def validate_send(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        state: ConversationState,
    ) -> None:
        if _is_session_protocol_event(envelope) or _is_task_event(envelope):
            return
        if envelope.event_type != EV_TEXT:
            # Unknown event types accepted as informational data — same
            # convention as consulting.
            return
        participant_ids = {p.agent_id for p in metadata.participants}
        if envelope.sender_id not in participant_ids:
            raise ProtocolError(
                f"conversation session {metadata.session_id!r} only accepts "
                f"sends from participants, got {envelope.sender_id!r}"
            )

    def on_accepted(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        state: ConversationState,
    ) -> AdapterResult:
        # Conversations end via explicit ``Hub.close_session`` or TTL
        # — never via adapter-initiated transitions on accepted content.
        return AdapterResult()

    def default_view_policy(
        self,
        metadata: SessionMetadata,
        participant_id: str,
    ) -> ViewPolicy:
        return WindowedSummary(recent_n=_DEFAULT_RECENT_N)
