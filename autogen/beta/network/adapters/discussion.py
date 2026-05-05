# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``DiscussionAdapter`` — multi-participant turn-taking session.

Ships ``round_robin`` ordering. Unsupported orderings are rejected at
create time so manifests on disk stay consistent with the adapter
that's actually loaded.

Default expectations:
* ``turn_within(120s, warn)`` — the expected next speaker should post
  within 2 minutes of being expected.
* ``turn_within(600s, hide)`` — silenced from the session if quiet for
  10 minutes.
"""

from dataclasses import dataclass, field

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
    ParticipantSchema,
    SessionManifest,
    SessionMetadata,
)
from ..views.base import ViewPolicy
from ..views.builtin import WindowedSummary
from .base import AdapterResult

__all__ = (
    "DISCUSSION_TYPE",
    "ORDERING_ROUND_ROBIN",
    "DiscussionAdapter",
    "DiscussionState",
)


DISCUSSION_TYPE = "discussion"
ORDERING_ROUND_ROBIN = "round_robin"
_SUPPORTED_ORDERINGS: frozenset[str] = frozenset({ORDERING_ROUND_ROBIN})
_DEFAULT_ORDERING = ORDERING_ROUND_ROBIN


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
class DiscussionState:
    """Folded state for a discussion session.

    ``participant_order`` is snapshotted at ``initial_state`` so
    ``fold`` can compute the next speaker without access to
    ``SessionMetadata`` (the adapter Protocol passes only state +
    envelope into ``fold``). The hub's ``hydrate()`` re-folds from the
    WAL, which is deterministic because ``initial_state`` reads the
    same ``metadata.participants`` snapshot every time.
    """

    participant_order: list[str] = field(default_factory=list)
    expected_next_speaker: str | None = None
    last_speaker_id: str | None = None
    last_envelope_id: str | None = None
    turn_count: int = 0


class DiscussionAdapter:
    """Multi-participant turn-taking session with ``round_robin`` ordering.

    Participants: 2+ (no upper bound). Initiator (``order=0``) speaks
    first; turns rotate through ``metadata.participants`` in ``order``,
    cycling back after the last participant.

    Knobs: ``{"ordering": "round_robin"}`` (default). Unsupported
    orderings are rejected at create time.

    Default view: :class:`WindowedSummary(recent_n=N*2)` where N =
    participant count — keeps prompt size bounded at any turn count.
    """

    def __init__(self) -> None:
        self.manifest = SessionManifest(
            type=DISCUSSION_TYPE,
            version=1,
            participants=ParticipantSchema(min=2),
            knobs_schema={"ordering": "str"},
            default_view_policy=WindowedSummary.name,
            expectations=[
                Expectation(
                    name="turn_within",
                    on_violation="warn",
                    params={"seconds": 120},
                ),
                Expectation(
                    name="turn_within",
                    on_violation="hide",
                    params={"seconds": 600},
                ),
            ],
        )

    # ── Adapter Protocol ────────────────────────────────────────────────────

    def initial_state(self, metadata: SessionMetadata) -> DiscussionState:
        order = [
            p.agent_id
            for p in sorted(metadata.participants, key=lambda p: p.order)
        ]
        return DiscussionState(
            participant_order=order,
            expected_next_speaker=metadata.creator_id,
        )

    def fold(self, envelope: Envelope, state: DiscussionState) -> DiscussionState:
        if _is_session_protocol_event(envelope) or _is_task_event(envelope):
            return state
        if envelope.event_type != EV_TEXT:
            return state
        try:
            idx = state.participant_order.index(envelope.sender_id)
        except ValueError:
            # Sender not in the rotation (shouldn't happen — validate_send
            # gates this) — leave state untouched rather than crash.
            return state
        next_idx = (idx + 1) % len(state.participant_order)
        return DiscussionState(
            participant_order=state.participant_order,
            expected_next_speaker=state.participant_order[next_idx],
            last_speaker_id=envelope.sender_id,
            last_envelope_id=envelope.envelope_id,
            turn_count=state.turn_count + 1,
        )

    def validate_create(self, metadata: SessionMetadata) -> None:
        if len(metadata.participants) < 2:
            raise ProtocolError(
                f"discussion requires at least 2 participants, got {len(metadata.participants)}"
            )
        ordering = metadata.knobs.get("ordering", _DEFAULT_ORDERING)
        if ordering not in _SUPPORTED_ORDERINGS:
            raise ProtocolError(
                f"discussion knobs.ordering={ordering!r} not supported; "
                f"choose from {sorted(_SUPPORTED_ORDERINGS)}"
            )

    def validate_send(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        state: DiscussionState,
    ) -> None:
        if _is_session_protocol_event(envelope) or _is_task_event(envelope):
            return
        if envelope.event_type != EV_TEXT:
            return
        if envelope.sender_id != state.expected_next_speaker:
            raise ProtocolError(
                f"discussion session {metadata.session_id!r} expects "
                f"{state.expected_next_speaker!r} to speak, got {envelope.sender_id!r}"
            )

    def on_accepted(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        state: DiscussionState,
    ) -> AdapterResult:
        # Discussions end via explicit ``Hub.close_session`` or TTL.
        # Speaker rotation happens entirely in ``fold``.
        return AdapterResult()

    def default_view_policy(
        self,
        metadata: SessionMetadata,
        participant_id: str,
    ) -> ViewPolicy:
        recent_n = max(len(metadata.participants) * 2, 4)
        return WindowedSummary(recent_n=recent_n)
