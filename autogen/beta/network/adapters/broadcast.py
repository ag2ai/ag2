# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""BroadcastAdapter — 1 → N fan-out session.

Delivery rules:

* **Cardinality.** Exactly one initiator plus one or more recipients.
* **Direction.** Only the initiator may post text envelopes; recipients
  consume them passively. This is the delivery semantics V2 used to
  carry as a one-shot ``TopicMessage`` plus a separate subscription
  concept — V3 collapses both into a single session type.
* **Lifetime.** Broadcasts do **not** auto-close after the first
  message. The caller uses ``close_session`` explicitly when the stream
  of messages is done; short-lived broadcasts simply post one envelope
  and then close. This is why topics (§5.1) map to the same adapter —
  a topic is a long-lived broadcast session with persistent
  subscription and per-subscriber cursors.
* **Quorum.** Handshake defaults to *all* recipients acking before the
  session enters ACTIVE; callers that want earlier progress pass
  ``required_acks`` on :meth:`Hub.create_session`.

Fan-out across participants is the hub's job — the adapter only
validates direction and cardinality.
"""

from __future__ import annotations

from ..envelope import EV_TEXT, Envelope
from ..errors import SessionTypeError
from ..session_types import (
    ParticipantRole,
    SessionMetadata,
    SessionState,
    SessionType,
)
from .base import AdapterResult


class BroadcastAdapter:
    session_type = SessionType.BROADCAST

    def validate_create(self, metadata: SessionMetadata) -> None:
        if len(metadata.participants) < 2:
            raise SessionTypeError(
                "broadcast requires an initiator and at least one recipient"
            )
        roles = [p.role for p in metadata.participants]
        if roles.count(ParticipantRole.INITIATOR) != 1:
            raise SessionTypeError("broadcast requires exactly one initiator")

    def _initiator(self, metadata: SessionMetadata) -> str:
        for p in metadata.participants:
            if p.role is ParticipantRole.INITIATOR:
                return p.actor_id
        raise SessionTypeError("broadcast session has no initiator")

    def validate_send(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        prior_envelopes: list[Envelope],
    ) -> None:
        if envelope.event_type != EV_TEXT:
            raise SessionTypeError(
                f"broadcast only carries {EV_TEXT!r} events, got {envelope.event_type!r}"
            )
        initiator = self._initiator(metadata)
        if envelope.sender_id != initiator:
            raise SessionTypeError("broadcast: only the initiator may send")
        # ``recipient_id`` is optional on a broadcast envelope — omit it
        # to target every non-initiator. If explicitly set, it must name
        # a participant (partial broadcasts are allowed).
        if envelope.recipient_id is not None and not metadata.has_participant(
            envelope.recipient_id
        ):
            raise SessionTypeError(
                "broadcast: explicit recipient is not a participant"
            )

    def on_accepted(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        prior_envelopes: list[Envelope],
    ) -> AdapterResult:
        # Broadcasts do not auto-close — the initiator decides. The only
        # transition we own here is pending → active, which normally
        # happens at handshake; validate_send enforces direction for
        # anything that slips through in PENDING.
        if metadata.state is SessionState.PENDING:
            return AdapterResult(next_state=SessionState.ACTIVE)
        return AdapterResult()
