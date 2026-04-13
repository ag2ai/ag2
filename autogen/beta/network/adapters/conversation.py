# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""ConversationAdapter — bidirectional multi-turn between two actors.

Either participant may send text envelopes. There is no automatic close —
callers issue an explicit ``close_session`` once the conversation is done.
"""

from __future__ import annotations

from ..envelope import EV_TEXT, Envelope
from ..errors import SessionTypeError
from ..session_types import SessionMetadata, SessionState, SessionType
from .base import AdapterResult


class ConversationAdapter:
    session_type = SessionType.CONVERSATION

    def validate_create(self, metadata: SessionMetadata) -> None:
        if len(metadata.participants) != 2:
            raise SessionTypeError("conversation requires exactly two participants")

    def validate_send(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        prior_envelopes: list[Envelope],
    ) -> None:
        if envelope.event_type != EV_TEXT:
            raise SessionTypeError(
                f"conversation only carries {EV_TEXT!r} events, got {envelope.event_type!r}"
            )
        if not metadata.has_participant(envelope.sender_id):
            raise SessionTypeError("conversation: sender is not a participant")
        if envelope.recipient_id is not None and not metadata.has_participant(envelope.recipient_id):
            raise SessionTypeError("conversation: recipient is not a participant")

    def on_accepted(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        prior_envelopes: list[Envelope],
    ) -> AdapterResult:
        if metadata.state is SessionState.PENDING:
            return AdapterResult(next_state=SessionState.ACTIVE)
        return AdapterResult()
