# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""ConsultingAdapter — strict 1Q1R session type.

Rules:

* Exactly two participants (initiator + respondent).
* First user envelope must flow initiator → respondent.
* Exactly one reply is allowed, respondent → initiator.
* Session closes immediately after the reply.
"""

from __future__ import annotations

from ..envelope import EV_TEXT, Envelope
from ..errors import SessionTypeError
from ..session_types import ParticipantRole, SessionMetadata, SessionState, SessionType
from .base import AdapterResult


class ConsultingAdapter:
    session_type = SessionType.CONSULTING

    def validate_create(self, metadata: SessionMetadata) -> None:
        if len(metadata.participants) != 2:
            raise SessionTypeError("consulting requires exactly two participants")
        roles = {p.role for p in metadata.participants}
        if ParticipantRole.INITIATOR not in roles:
            raise SessionTypeError("consulting requires an initiator participant")
        if ParticipantRole.RESPONDENT not in roles:
            raise SessionTypeError("consulting requires a respondent participant")

    def _initiator(self, metadata: SessionMetadata) -> str:
        for p in metadata.participants:
            if p.role is ParticipantRole.INITIATOR:
                return p.actor_id
        raise SessionTypeError("no initiator")

    def _respondent(self, metadata: SessionMetadata) -> str:
        for p in metadata.participants:
            if p.role is ParticipantRole.RESPONDENT:
                return p.actor_id
        raise SessionTypeError("no respondent")

    def validate_send(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        prior_envelopes: list[Envelope],
    ) -> None:
        if envelope.event_type != EV_TEXT:
            raise SessionTypeError(
                f"consulting only carries {EV_TEXT!r} events, got {envelope.event_type!r}"
            )
        initiator = self._initiator(metadata)
        respondent = self._respondent(metadata)

        if not prior_envelopes:
            # First envelope must be the initiator's question.
            if envelope.sender_id != initiator:
                raise SessionTypeError("consulting: first send must come from the initiator")
            if envelope.recipient_id not in (None, respondent):
                raise SessionTypeError("consulting: first send must target the respondent")
            return

        if len(prior_envelopes) == 1:
            # Reply must come from the respondent and target the initiator.
            if envelope.sender_id != respondent:
                raise SessionTypeError("consulting: reply must come from the respondent")
            if envelope.recipient_id not in (None, initiator):
                raise SessionTypeError("consulting: reply must target the initiator")
            return

        raise SessionTypeError(
            "consulting is strict 1Q1R — the session is exhausted after the reply"
        )

    def on_accepted(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        prior_envelopes: list[Envelope],
    ) -> AdapterResult:
        # ``prior_envelopes`` is the tail BEFORE this envelope. After writing
        # the new envelope, the total count becomes len(prior_envelopes) + 1.
        total = len(prior_envelopes) + 1
        if total == 1:
            return AdapterResult(next_state=SessionState.ACTIVE)
        if total == 2:
            return AdapterResult(next_state=SessionState.CLOSED, close_reason="consulting_done")
        return AdapterResult()
