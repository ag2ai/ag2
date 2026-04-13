# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""DiscussionAdapter — multi-participant turn-taking session.

Discussions carry three orderings that cover the common multi-actor
patterns:

* ``dynamic`` — any participant may send at any time (chat room). This
  is the adapter's default and the most permissive mode. It subsumes
  every V2 "multi-agent discussion" flow that didn't require a
  pre-declared speaking order.
* ``static`` — speakers take turns in the order declared at session
  creation. The initiator speaks first, then the next participant in
  ``SessionMetadata.participants`` order, and so on until the last
  participant's turn completes. After the last participant the
  discussion closes (unless ``on_failure="continue"`` kept an earlier
  participant in the queue — see below). This mode is the V2 "pipeline"
  topology replacement: A → B → C, each stage sees only the previous
  stage's output (the assembly policy lives in framework core; the
  delivery rule lives here).
* ``round_robin`` — like ``static``, but speaking wraps around the
  participant list and the session never auto-closes. The initiator
  decides when to close.

``on_failure`` controls what happens when a send from the expected
speaker violates a rule:

* ``abort`` — any rule / type violation closes the session with
  ``close_reason="discussion_failed"``.
* ``continue`` — (static / round-robin only) skip the failing speaker
  and advance to the next expected one. Useful for best-effort
  pipelines where a single stage's failure shouldn't tank the whole
  discussion.

Turn advancement in static / round-robin is computed as "the participant
whose order index comes next after the current speaker." The WAL is the
authoritative record of who has spoken — we re-derive the current
expected speaker from the prior user envelopes on every send, so the
adapter stays stateless (important for cold-restart hydrate and
post-incident replays).
"""

from __future__ import annotations

from ..envelope import EV_TEXT, Envelope
from ..errors import SessionTypeError
from ..session_types import (
    Participant,
    ParticipantRole,
    SessionMetadata,
    SessionState,
    SessionType,
)
from .base import AdapterResult

ORDERING_DYNAMIC = "dynamic"
ORDERING_STATIC = "static"
ORDERING_ROUND_ROBIN = "round_robin"


_VALID_ORDERINGS: frozenset[str] = frozenset(
    {ORDERING_DYNAMIC, ORDERING_STATIC, ORDERING_ROUND_ROBIN}
)

ON_FAILURE_ABORT = "abort"
ON_FAILURE_CONTINUE = "continue"

_VALID_FAILURE_POLICIES: frozenset[str] = frozenset(
    {ON_FAILURE_ABORT, ON_FAILURE_CONTINUE}
)


class DiscussionAdapter:
    session_type = SessionType.DISCUSSION

    # ------------------------------------------------------------------
    # Create-time validation
    # ------------------------------------------------------------------

    def validate_create(self, metadata: SessionMetadata) -> None:
        if len(metadata.participants) < 2:
            raise SessionTypeError("discussion requires at least two participants")
        roles = [p.role for p in metadata.participants]
        if roles.count(ParticipantRole.INITIATOR) != 1:
            raise SessionTypeError("discussion requires exactly one initiator")
        ordering = metadata.ordering or ORDERING_DYNAMIC
        if ordering not in _VALID_ORDERINGS:
            raise SessionTypeError(
                f"discussion ordering must be one of {sorted(_VALID_ORDERINGS)}, "
                f"got {ordering!r}"
            )
        on_failure = metadata.on_failure or ON_FAILURE_ABORT
        if on_failure not in _VALID_FAILURE_POLICIES:
            raise SessionTypeError(
                f"discussion on_failure must be one of "
                f"{sorted(_VALID_FAILURE_POLICIES)}, got {on_failure!r}"
            )

    # ------------------------------------------------------------------
    # Turn ordering helpers
    # ------------------------------------------------------------------

    def _ordered_participants(self, metadata: SessionMetadata) -> list[Participant]:
        """Return participants in their declared turn order.

        ``SessionMetadata.participants`` is created in insertion order by
        :meth:`Hub.create_session`, with the initiator first and every
        other participant following in ``participant_names`` order. Each
        participant also has a numeric ``.order`` stamped on creation;
        we re-sort by that value so any later code that rewrites the
        list keeps the turn sequence stable.
        """

        return sorted(
            metadata.participants,
            key=lambda p: (p.order if p.order is not None else 0, p.actor_id),
        )

    def _expected_speaker(
        self,
        metadata: SessionMetadata,
        prior_envelopes: list[Envelope],
    ) -> str | None:
        ordering = metadata.ordering or ORDERING_DYNAMIC
        if ordering == ORDERING_DYNAMIC:
            return None
        ordered = self._ordered_participants(metadata)
        seq = [p.actor_id for p in ordered]
        turn_count = len(prior_envelopes)
        if ordering == ORDERING_STATIC:
            if turn_count >= len(seq):
                return None  # signals "session should already be closed"
            return seq[turn_count]
        # round-robin
        return seq[turn_count % len(seq)]

    # ------------------------------------------------------------------
    # Per-envelope validation
    # ------------------------------------------------------------------

    def validate_send(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        prior_envelopes: list[Envelope],
    ) -> None:
        if envelope.event_type != EV_TEXT:
            raise SessionTypeError(
                f"discussion only carries {EV_TEXT!r} events, got {envelope.event_type!r}"
            )
        if not metadata.has_participant(envelope.sender_id):
            raise SessionTypeError("discussion: sender is not a participant")

        ordering = metadata.ordering or ORDERING_DYNAMIC
        if ordering == ORDERING_DYNAMIC:
            return

        expected = self._expected_speaker(metadata, prior_envelopes)
        if expected is None:
            raise SessionTypeError("discussion: static session has no remaining turns")
        if envelope.sender_id != expected:
            raise SessionTypeError(
                f"discussion: expected {expected} to speak next, got {envelope.sender_id}"
            )

    def on_accepted(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        prior_envelopes: list[Envelope],
    ) -> AdapterResult:
        if metadata.state is SessionState.PENDING:  # noqa: SIM108
            # First real send flips the session to ACTIVE. Fall through
            # to the close check below so a 1-participant-past-initiator
            # pipeline still closes immediately.
            pending_to_active = True
        else:
            pending_to_active = False

        ordering = metadata.ordering or ORDERING_DYNAMIC
        if ordering == ORDERING_STATIC:
            total_turns = len(prior_envelopes) + 1
            seq = self._ordered_participants(metadata)
            if total_turns >= len(seq):
                return AdapterResult(
                    next_state=SessionState.CLOSED,
                    close_reason="discussion_done",
                )
        if pending_to_active:
            return AdapterResult(next_state=SessionState.ACTIVE)
        return AdapterResult()
