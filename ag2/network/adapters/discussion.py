# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``DiscussionAdapter`` — multi-participant turn-taking channel.

Ships ``round_robin`` ordering. Unsupported orderings are rejected at
create time so manifests on disk stay consistent with the adapter
that's actually loaded.

Default expectations:
* ``turn_within(120s, warn)`` — the expected next speaker should post
  within 2 minutes of being expected.
* ``turn_within(600s, hide)`` — silenced from the channel if quiet for
  10 minutes.
"""

from dataclasses import dataclass, field

from ..channel import (
    ChannelManifest,
    ChannelMetadata,
    Expectation,
    ParticipantSchema,
)
from ..envelope import (
    EV_CHANNEL_CLOSED,
    EV_CHANNEL_EXPIRED,
    EV_CHANNEL_INVITE,
    EV_CHANNEL_INVITE_ACK,
    EV_CHANNEL_INVITE_REJECT,
    EV_CHANNEL_OPENED,
    EV_TEXT,
    Envelope,
)
from ..errors import ProtocolError
from ..views.base import ViewPolicy
from ..views.builtin import NamedWindowedSummary
from .base import (
    AdapterResult,
    ExpectedTurn,
    default_build_packet_envelope,
    default_build_round_envelope,
    default_build_text_envelope,
    default_extract_turn_input,
    default_render_envelope,
)

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


_CHANNEL_PROTOCOL_EVENTS: frozenset[str] = frozenset({
    EV_CHANNEL_INVITE,
    EV_CHANNEL_INVITE_ACK,
    EV_CHANNEL_INVITE_REJECT,
    EV_CHANNEL_OPENED,
    EV_CHANNEL_CLOSED,
    EV_CHANNEL_EXPIRED,
})


def _is_channel_protocol_event(envelope: Envelope) -> bool:
    return envelope.event_type in _CHANNEL_PROTOCOL_EVENTS


def _is_task_event(envelope: Envelope) -> bool:
    return envelope.event_type.startswith("ag2.task.")


@dataclass(slots=True)
class DiscussionState:
    """Folded state for a discussion channel.

    ``participant_order`` is snapshotted at ``initial_state`` so
    ``fold`` can compute the next speaker without access to
    ``ChannelMetadata`` (the adapter Protocol passes only state +
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
    """Multi-participant turn-taking channel with ``round_robin`` ordering.

    Participants: 2+ (no upper bound). Initiator (``order=0``) speaks
    first; turns rotate through ``metadata.participants`` in ``order``,
    cycling back after the last participant.

    Knobs: ``{"ordering": "round_robin"}`` (default). Unsupported
    orderings are rejected at create time.

    Default view: :class:`NamedWindowedSummary(recent_n=N*2)` where
    N = participant count — keeps prompt size bounded at any turn
    count AND prefixes each non-self projection line with the
    sender's name so the LLM can tell its peers apart in a 3+ party
    chat (the assistant/user role bit alone collapses every "other"
    into one indistinguishable stream).
    """

    def __init__(self) -> None:
        self.manifest = ChannelManifest(
            type=DISCUSSION_TYPE,
            version=1,
            participants=ParticipantSchema(min=2),
            knobs_schema={"ordering": "str"},
            default_view_policy=NamedWindowedSummary.name,
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

    def initial_state(self, metadata: ChannelMetadata) -> DiscussionState:
        order = [p.agent_id for p in sorted(metadata.participants, key=lambda p: p.order)]
        return DiscussionState(
            participant_order=order,
            expected_next_speaker=metadata.creator_id,
        )

    def fold(self, envelope: Envelope, state: DiscussionState) -> DiscussionState:
        if _is_channel_protocol_event(envelope) or _is_task_event(envelope):
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

    def validate_create(self, metadata: ChannelMetadata) -> None:
        if len(metadata.participants) < 2:
            raise ProtocolError(f"discussion requires at least 2 participants, got {len(metadata.participants)}")
        ordering = metadata.knobs.get("ordering", _DEFAULT_ORDERING)
        if ordering not in _SUPPORTED_ORDERINGS:
            raise ProtocolError(
                f"discussion knobs.ordering={ordering!r} not supported; choose from {sorted(_SUPPORTED_ORDERINGS)}"
            )

    def validate_send(
        self,
        metadata: ChannelMetadata,
        envelope: Envelope,
        state: DiscussionState,
    ) -> None:
        if _is_channel_protocol_event(envelope) or _is_task_event(envelope):
            return
        if envelope.event_type != EV_TEXT:
            return
        if envelope.sender_id != state.expected_next_speaker:
            raise ProtocolError(
                f"discussion channel {metadata.channel_id!r} expects "
                f"{state.expected_next_speaker!r} to speak, got {envelope.sender_id!r}"
            )

    def on_accepted(
        self,
        metadata: ChannelMetadata,
        envelope: Envelope,
        state: DiscussionState,
    ) -> AdapterResult:
        # Discussions end via explicit ``Hub.close_channel`` or TTL.
        # Speaker rotation happens entirely in ``fold``.
        return AdapterResult()

    def expected_next(
        self,
        metadata: ChannelMetadata,
        state: DiscussionState,
    ) -> ExpectedTurn | None:
        if state.expected_next_speaker is None:
            return None
        return ExpectedTurn(
            agent_id=state.expected_next_speaker,
            triggering_envelope_id=state.last_envelope_id,
        )

    def default_view_policy(
        self,
        metadata: ChannelMetadata,
        participant_id: str,
    ) -> ViewPolicy:
        recent_n = max(len(metadata.participants) * 2, 4)
        return NamedWindowedSummary(recent_n=recent_n)

    def extract_turn_input(self, envelope):
        return default_extract_turn_input(envelope)

    def build_round_envelope(self, metadata, sender_id, reply, events, state, hub):
        return default_build_round_envelope(metadata, sender_id, reply, events, state, hub)

    def render_envelope(self, envelope):
        return default_render_envelope(envelope)

    def tools_for(self, client, metadata, state, participant_id):
        return []

    def build_text_envelope(self, channel_id, sender_id, text, *, audience=None, causation_id=None):
        return default_build_text_envelope(channel_id, sender_id, text, audience=audience, causation_id=causation_id)

    def build_packet_envelope(
        self,
        channel_id,
        sender_id,
        body,
        *,
        handoff=None,
        context_set=None,
        audience=None,
        causation_id=None,
    ):
        return default_build_packet_envelope(
            channel_id,
            sender_id,
            body,
            handoff=handoff,
            context_set=context_set,
            audience=audience,
            causation_id=causation_id,
        )
