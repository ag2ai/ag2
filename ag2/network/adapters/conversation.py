# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``ConversationAdapter`` — bidirectional multi-turn 1+1 channel.

Two participants: initiator and respondent. Either side may send at
any turn. The channel runs until an explicit ``close()`` call or TTL
expiry — there is no auto-close based on message content.

Default expectations:
* ``max_silence(3600s, audit)`` — light enforcement; conversations may
  legitimately span hours of idle time.

Task envelopes (``ag2.task.*``) and channel-protocol envelopes
(``ag2.channel.*``) bypass conversation's send rule the same way
consulting does — they're hub bookkeeping or task lifecycle observed
on the channel.
"""

from dataclasses import dataclass

from ..channel import (
    ChannelManifest,
    ChannelMetadata,
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
from ..views.builtin import WindowedSummary
from .base import (
    AdapterResult,
    ExpectedTurn,
    default_build_packet_envelope,
    default_build_round_envelope,
    default_build_text_envelope,
    default_expected_next,
    default_extract_turn_input,
    default_render_envelope,
)

__all__ = ("CONVERSATION_TYPE", "ConversationAdapter", "ConversationState")


CONVERSATION_TYPE = "conversation"
_DEFAULT_RECENT_N = 10


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
class ConversationState:
    """Folded state for a conversation channel.

    Conversations have no per-turn ordering constraint, so the state
    only tracks bookkeeping for views and observability — the adapter
    never returns a ``next_state`` from ``on_accepted``.
    """

    turn_count: int = 0
    last_speaker_id: str | None = None
    last_envelope_id: str | None = None


class ConversationAdapter:
    """Bidirectional multi-turn 1+1 channel.

    Knobs: none. Participants: exactly 2 (initiator + respondent).
    Default view: :class:`WindowedSummary(recent_n=10)` — bounded
    prompt size at any turn count.
    """

    def __init__(self) -> None:
        self._say_tool_cache: dict[str, object] = {}
        self.manifest = ChannelManifest(
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

    def initial_state(self, metadata: ChannelMetadata) -> ConversationState:
        return ConversationState()

    def fold(self, envelope: Envelope, state: ConversationState) -> ConversationState:
        if _is_channel_protocol_event(envelope) or _is_task_event(envelope):
            return state
        if envelope.event_type != EV_TEXT:
            return state
        return ConversationState(
            turn_count=state.turn_count + 1,
            last_speaker_id=envelope.sender_id,
            last_envelope_id=envelope.envelope_id,
        )

    def validate_create(self, metadata: ChannelMetadata) -> None:
        roles = {p.role for p in metadata.participants}
        if ParticipantRole.INITIATOR not in roles:
            raise ProtocolError("conversation requires exactly one initiator")
        if ParticipantRole.RESPONDENT not in roles:
            raise ProtocolError("conversation requires exactly one respondent")
        if len(metadata.participants) != 2:
            raise ProtocolError(f"conversation requires exactly 2 participants, got {len(metadata.participants)}")

    def validate_send(
        self,
        metadata: ChannelMetadata,
        envelope: Envelope,
        state: ConversationState,
    ) -> None:
        if _is_channel_protocol_event(envelope) or _is_task_event(envelope):
            return
        if envelope.event_type != EV_TEXT:
            # Unknown event types accepted as informational data — same
            # convention as consulting.
            return
        participant_ids = {p.agent_id for p in metadata.participants}
        if envelope.sender_id not in participant_ids:
            raise ProtocolError(
                f"conversation channel {metadata.channel_id!r} only accepts "
                f"sends from participants, got {envelope.sender_id!r}"
            )

    def on_accepted(
        self,
        metadata: ChannelMetadata,
        envelope: Envelope,
        state: ConversationState,
    ) -> AdapterResult:
        # Conversations end via explicit ``Hub.close_channel`` or TTL
        # — never via adapter-initiated transitions on accepted content.
        return AdapterResult()

    def expected_next(
        self,
        metadata: ChannelMetadata,
        state: ConversationState,
    ) -> ExpectedTurn | None:
        # Conversation has no turn ordering — any participant may
        # speak at any time, so there is no expected next speaker.
        return default_expected_next(metadata, state)

    def default_view_policy(
        self,
        metadata: ChannelMetadata,
        participant_id: str,
    ) -> ViewPolicy:
        return WindowedSummary(recent_n=_DEFAULT_RECENT_N)

    def extract_turn_input(self, envelope):
        return default_extract_turn_input(envelope)

    def build_round_envelope(self, metadata, sender_id, reply, events, state, hub):
        return default_build_round_envelope(metadata, sender_id, reply, events, state, hub)

    def render_envelope(self, envelope):
        return default_render_envelope(envelope)

    def tools_for(self, client, metadata, state, participant_id):
        """Conversation has no turn order — both participants always
        see ``say``. Tool resolution is memoized per-client.
        """
        return [self._cached_say_tool(client)]

    def _cached_say_tool(self, client):
        """Memoize ``make_say_tool`` per ``client.agent_id``."""
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
