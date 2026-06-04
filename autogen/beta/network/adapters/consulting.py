# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``ConsultingAdapter`` — strict 1Q1R channel.

Two participants: an initiator and a respondent. The initiator sends
exactly one substantive envelope, the respondent sends exactly one
reply, and the channel auto-closes on the reply.

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

from ..channel import (
    ChannelManifest,
    ChannelMetadata,
    ChannelState,
    Expectation,
    ParticipantRole,
    ParticipantSchema,
)
from ..client.tools.say import make_say_tool
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
from ..views.builtin import FullTranscript
from .base import (
    AdapterResult,
    ExpectedTurn,
    default_build_packet_envelope,
    default_build_round_envelope,
    default_build_text_envelope,
    default_extract_turn_input,
    default_render_envelope,
)

__all__ = ("CONSULTING_TYPE", "ConsultingAdapter", "ConsultingState")


CONSULTING_TYPE = "consulting"


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
class ConsultingState:
    """Folded state for a consulting channel.

    ``initiator_sent`` flips True on the first ``EV_TEXT`` from the
    initiator. ``respondent_replied`` flips True on the first
    ``EV_TEXT`` from the respondent following the initiator's send.
    Both flags True means the channel is complete and should
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
        # Per-client say-tool cache: avoids paying the ``fast_depends``
        # JSON-schema build cost on every notify-handler turn. Keyed by
        # ``client.agent_id`` (stable across the client's lifetime). An
        # unregister + re-register yields a new agent_id, so the dead
        # entry from the prior generation is leaked memory bounded by
        # the total registration count — acceptable for V1.
        self._say_tool_cache: dict[str, object] = {}
        self.manifest = ChannelManifest(
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
    def _initiator_id(metadata: ChannelMetadata) -> str:
        return metadata.creator_id

    @staticmethod
    def _respondent_id(metadata: ChannelMetadata) -> str:
        for participant in metadata.participants:
            if participant.role is ParticipantRole.RESPONDENT:
                return participant.agent_id
        raise ProtocolError(f"consulting channel {metadata.channel_id!r} has no respondent")

    # ── Adapter Protocol ────────────────────────────────────────────────────

    def initial_state(self, metadata: ChannelMetadata) -> ConsultingState:
        return ConsultingState()

    def fold(self, envelope: Envelope, state: ConsultingState) -> ConsultingState:
        if _is_channel_protocol_event(envelope) or _is_task_event(envelope):
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

    def validate_create(self, metadata: ChannelMetadata) -> None:
        roles = {p.role for p in metadata.participants}
        if ParticipantRole.INITIATOR not in roles:
            raise ProtocolError("consulting requires exactly one initiator")
        if ParticipantRole.RESPONDENT not in roles:
            raise ProtocolError("consulting requires exactly one respondent")
        if len(metadata.participants) != 2:
            raise ProtocolError(f"consulting requires exactly 2 participants, got {len(metadata.participants)}")

    def validate_send(
        self,
        metadata: ChannelMetadata,
        envelope: Envelope,
        state: ConsultingState,
    ) -> None:
        # Hub-emitted protocol events and task envelopes bypass the 1Q1R rule.
        if _is_channel_protocol_event(envelope) or _is_task_event(envelope):
            return
        if envelope.event_type != EV_TEXT:
            # Unknown event types are accepted as informational data on the
            # channel.
            return
        if state.initiator_sent and state.respondent_replied:
            raise ProtocolError(f"consulting channel {metadata.channel_id!r} already complete")
        if not state.initiator_sent:
            initiator_id = self._initiator_id(metadata)
            if envelope.sender_id != initiator_id:
                raise ProtocolError(
                    f"consulting channel {metadata.channel_id!r} expects first send "
                    f"from initiator {initiator_id!r}, got {envelope.sender_id!r}"
                )
        else:
            respondent_id = self._respondent_id(metadata)
            if envelope.sender_id != respondent_id:
                raise ProtocolError(
                    f"consulting channel {metadata.channel_id!r} expects reply "
                    f"from respondent {respondent_id!r}, got {envelope.sender_id!r}"
                )

    def on_accepted(
        self,
        metadata: ChannelMetadata,
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
                next_state=ChannelState.CLOSED,
                auto_close_reason="consulting_complete",
            )
        return AdapterResult()

    def expected_next(
        self,
        metadata: ChannelMetadata,
        state: ConsultingState,
    ) -> ExpectedTurn | None:
        # 1Q1R cycle: initiator first, then respondent. Once both have
        # spoken the cycle is complete and no participant is expected.
        if not state.initiator_sent:
            return ExpectedTurn(
                agent_id=self._initiator_id(metadata),
                triggering_envelope_id=state.last_envelope_id,
            )
        if not state.respondent_replied:
            return ExpectedTurn(
                agent_id=self._respondent_id(metadata),
                triggering_envelope_id=state.last_envelope_id,
            )
        return None

    def default_view_policy(
        self,
        metadata: ChannelMetadata,
        participant_id: str,
    ) -> ViewPolicy:
        return FullTranscript()

    def extract_turn_input(self, envelope):
        return default_extract_turn_input(envelope)

    def build_round_envelope(self, metadata, sender_id, reply, events, state, hub):
        return default_build_round_envelope(metadata, sender_id, reply, events, state, hub)

    def render_envelope(self, envelope):
        return default_render_envelope(envelope)

    def tools_for(self, client, metadata, state, participant_id):
        """Consulting offers ``say`` to the participant whose turn it is.

        State gating: the initiator has the floor until they send the
        prompt; the respondent has the floor after the prompt lands and
        before they reply. Once the respondent replies the channel
        auto-closes — no further turns.

        The resolved tool is memoized per-client on the adapter (see
        :meth:`_cached_say_tool`) so the per-turn ``fast_depends`` schema
        build cost is paid once.
        """
        initiator_id = next(
            (p.agent_id for p in metadata.participants if p.role == ParticipantRole.INITIATOR),
            None,
        )
        if participant_id == initiator_id and not state.initiator_sent:
            return [self._cached_say_tool(client)]
        if participant_id != initiator_id and state.initiator_sent and not state.respondent_replied:
            return [self._cached_say_tool(client)]
        return []

    def _cached_say_tool(self, client):
        """Memoize the per-client ``say`` tool.

        The ``fast_depends`` schema build inside the ``@tool`` decorator
        is not free; building the tool fresh on every notify-handler
        turn would dominate per-turn latency at scale. Cache by
        ``client.agent_id`` (stable for the client's lifetime).
        """
        cached = self._say_tool_cache.get(client.agent_id)
        if cached is not None:
            return cached
        tool = make_say_tool(client)
        self._say_tool_cache[client.agent_id] = tool
        return tool

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
