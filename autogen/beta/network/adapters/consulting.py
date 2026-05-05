# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``ConsultingAdapter`` — strict 1Q1R session.

Two participants: an initiator and a respondent. The initiator sends
exactly one substantive envelope, the respondent sends exactly one
reply, and the session auto-closes on the reply.

Default expectations:
* ``acks_within(30s, auto_close)`` — invitee must ack within 30s
* ``reply_within(600s, auto_close)`` — respondent must reply within
  10 minutes of the initiator's send

Task envelopes (``ag2.task.*``) bypass the 1Q1R contract: an LLM
running as the respondent can emit progress / result envelopes
mid-reply without the consulting adapter auto-closing on the first
non-respondent send.
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
    SessionState,
)
from ..views.base import ViewPolicy
from ..views.builtin import FullTranscript
from .base import AdapterResult

__all__ = ("CONSULTING_TYPE", "ConsultingAdapter", "ConsultingState")


CONSULTING_TYPE = "consulting"


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
class ConsultingState:
    """Folded state for a consulting session.

    ``initiator_sent`` flips True on the first ``EV_TEXT`` from the
    initiator. ``respondent_replied`` flips True on the first
    ``EV_TEXT`` from the respondent following the initiator's send.
    Both flags True means the session is complete and should
    auto-close.
    """

    initiator_sent: bool = False
    respondent_replied: bool = False
    last_envelope_id: str | None = None


class ConsultingAdapter:
    """Strict 1Q1R: initiator → 1 envelope, respondent → 1 reply, auto-close.

    Knobs: none. Participants: exactly 2 (initiator + respondent).

    Default view: :class:`FullTranscript` (transcripts are short).
    """

    def __init__(self) -> None:
        # __init__ stores params; manifest is a class-level constant
        # constructed once.
        self.manifest = SessionManifest(
            type=CONSULTING_TYPE,
            version=1,
            participants=ParticipantSchema(
                min=2,
                max=2,
                roles=[ParticipantRole.INITIATOR.value, ParticipantRole.RESPONDENT.value],
            ),
            knobs_schema={},
            default_view_policy=FullTranscript.name,
            expectations=[
                Expectation(
                    name="acks_within",
                    on_violation="auto_close",
                    params={"seconds": 30},
                ),
                Expectation(
                    name="reply_within",
                    on_violation="auto_close",
                    params={"seconds": 600},
                ),
            ],
        )

    # ── Static helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _initiator_id(metadata: SessionMetadata) -> str:
        return metadata.creator_id

    @staticmethod
    def _respondent_id(metadata: SessionMetadata) -> str:
        for participant in metadata.participants:
            if participant.role is ParticipantRole.RESPONDENT:
                return participant.agent_id
        raise ProtocolError(f"consulting session {metadata.session_id!r} has no respondent")

    # ── Adapter Protocol ────────────────────────────────────────────────────

    def initial_state(self, metadata: SessionMetadata) -> ConsultingState:
        return ConsultingState()

    def fold(self, envelope: Envelope, state: ConsultingState) -> ConsultingState:
        if _is_session_protocol_event(envelope) or _is_task_event(envelope):
            return state
        if envelope.event_type != EV_TEXT:
            return state
        if not state.initiator_sent:
            return ConsultingState(
                initiator_sent=True,
                respondent_replied=False,
                last_envelope_id=envelope.envelope_id,
            )
        if not state.respondent_replied:
            return ConsultingState(
                initiator_sent=True,
                respondent_replied=True,
                last_envelope_id=envelope.envelope_id,
            )
        return state

    def validate_create(self, metadata: SessionMetadata) -> None:
        roles = {p.role for p in metadata.participants}
        if ParticipantRole.INITIATOR not in roles:
            raise ProtocolError("consulting requires exactly one initiator")
        if ParticipantRole.RESPONDENT not in roles:
            raise ProtocolError("consulting requires exactly one respondent")
        if len(metadata.participants) != 2:
            raise ProtocolError(
                f"consulting requires exactly 2 participants, got {len(metadata.participants)}"
            )

    def validate_send(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        state: ConsultingState,
    ) -> None:
        # Hub-emitted protocol events and task envelopes bypass the 1Q1R rule.
        if _is_session_protocol_event(envelope) or _is_task_event(envelope):
            return
        if envelope.event_type != EV_TEXT:
            # Unknown event types are accepted as informational data on the
            # session.
            return
        if state.initiator_sent and state.respondent_replied:
            raise ProtocolError(
                f"consulting session {metadata.session_id!r} already complete"
            )
        if not state.initiator_sent:
            initiator_id = self._initiator_id(metadata)
            if envelope.sender_id != initiator_id:
                raise ProtocolError(
                    f"consulting session {metadata.session_id!r} expects first send "
                    f"from initiator {initiator_id!r}, got {envelope.sender_id!r}"
                )
        else:
            respondent_id = self._respondent_id(metadata)
            if envelope.sender_id != respondent_id:
                raise ProtocolError(
                    f"consulting session {metadata.session_id!r} expects reply "
                    f"from respondent {respondent_id!r}, got {envelope.sender_id!r}"
                )

    def on_accepted(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        state: ConsultingState,
    ) -> AdapterResult:
        if envelope.event_type != EV_TEXT:
            return AdapterResult()
        if state.initiator_sent and state.respondent_replied:
            # Direct to CLOSED — consulting has no async cleanup phase.
            # The transitional ``CLOSING`` state is reserved for adapters
            # that need a quiescence window (e.g., draining streamed
            # chunks before close).
            return AdapterResult(
                next_state=SessionState.CLOSED,
                auto_close_reason="consulting_complete",
            )
        return AdapterResult()

    def default_view_policy(
        self,
        metadata: SessionMetadata,
        participant_id: str,
    ) -> ViewPolicy:
        return FullTranscript()
