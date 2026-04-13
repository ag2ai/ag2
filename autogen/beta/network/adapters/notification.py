# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""NotificationAdapter — fire-and-ack one-shot delivery.

Rules:

* Exactly two participants.
* Exactly one text envelope allowed, sender → recipient.
* The recipient acks by posting no further content; the session closes
  after the hub records the recipient's receipt.
"""

from __future__ import annotations

from ..envelope import EV_TEXT, Envelope
from ..errors import SessionTypeError
from ..session_types import ParticipantRole, SessionMetadata, SessionState, SessionType
from .base import AdapterResult


class NotificationAdapter:
    session_type = SessionType.NOTIFICATION

    def validate_create(self, metadata: SessionMetadata) -> None:
        if len(metadata.participants) != 2:
            raise SessionTypeError("notification requires exactly two participants")
        roles = {p.role for p in metadata.participants}
        if ParticipantRole.INITIATOR not in roles:
            raise SessionTypeError("notification requires an initiator participant")
        if ParticipantRole.RESPONDENT not in roles:
            raise SessionTypeError("notification requires a recipient participant")

    def _initiator(self, metadata: SessionMetadata) -> str:
        for p in metadata.participants:
            if p.role is ParticipantRole.INITIATOR:
                return p.actor_id
        raise SessionTypeError("no initiator")

    def validate_send(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        prior_envelopes: list[Envelope],
    ) -> None:
        if envelope.event_type != EV_TEXT:
            raise SessionTypeError(
                f"notification only carries {EV_TEXT!r} events, got {envelope.event_type!r}"
            )
        if envelope.sender_id != self._initiator(metadata):
            raise SessionTypeError("notification: only the initiator may send")
        if prior_envelopes:
            raise SessionTypeError("notification carries exactly one message")

    def on_accepted(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        prior_envelopes: list[Envelope],
    ) -> AdapterResult:
        # Single message lands — close immediately.
        return AdapterResult(next_state=SessionState.CLOSED, close_reason="notification_done")
