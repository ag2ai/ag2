# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``ChannelAdapter`` Protocol + ``AdapterState`` marker + ``AdapterResult``.

Key invariants:

* Adapters are stateless and pure.
* Every decision derives from ``(metadata, AdapterState)``.
* ``validate_send`` and ``on_accepted`` are O(1), not O(WAL) — the hub
  passes the cached state in.
* ``fold`` is called once per WAL append by the hub, and called
  repeatedly during ``Hub.hydrate()`` to rebuild state from disk. It
  must be a pure function.

Three layers of surface per adapter:

* **Capabilities** — what the hub calls (``validate_create`` /
  ``validate_send`` / ``fold`` / ``on_accepted`` / ``initial_state``).
* **Envelope helpers** (``build_text_envelope`` /
  ``build_packet_envelope``) — pure constructors any client uses to
  produce correctly-shaped envelopes for this adapter's protocol.
  Framework-agnostic; not LLM-specific.
* **LLM tools** (``tools_for``) — the AG2-LLM-facing presentation
  layer. The default notify handler resolves these per turn and merges
  them with the identity-level cross-cutting tools attached by
  :class:`NetworkPlugin`. Adapters that take no LLM input (e.g.
  workflow, where handoff tools are user-authored) return ``[]``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ag2.events import Input

from ..channel import ChannelManifest, ChannelMetadata, ChannelState
from ..envelope import EV_PACKET, EV_TEXT, Envelope
from ..handoff import Handoff
from ..views.base import ViewPolicy

if TYPE_CHECKING:
    from ag2.agent import AgentReply
    from ag2.events import BaseEvent
    from ag2.tools import Tool

    from ..client.agent_client import AgentClient
    from ..hub.core import Hub

__all__ = (
    "AdapterResult",
    "AdapterState",
    "ChannelAdapter",
    "ExpectedTurn",
    "default_build_packet_envelope",
    "default_build_round_envelope",
    "default_build_text_envelope",
    "default_expected_next",
    "default_extract_turn_input",
    "default_render_envelope",
    "default_tools_for",
)


class AdapterState(Protocol):
    """Marker Protocol — concrete adapters define their own dataclass.

    Empty by design: the hub treats adapter state opaquely and only
    passes it back into ``fold`` / ``validate_send`` / ``on_accepted``.
    """


@dataclass(slots=True)
class AdapterResult:
    """What an adapter wants the hub to do after accepting an envelope.

    ``next_state=None`` leaves the channel in its current state. The
    hub broadcasts ``EV_CHANNEL_CLOSED`` / ``EV_CHANNEL_EXPIRED`` when
    transitioning to a terminal state.
    """

    next_state: ChannelState | None = None
    auto_close_reason: str = ""


@dataclass(slots=True, frozen=True)
class ExpectedTurn:
    """Who the protocol expects to act next, and what put the turn on them.

    ``agent_id`` is the participant the adapter expects to send the
    next substantive envelope. ``triggering_envelope_id`` names the
    envelope that put the turn on this participant — typically the
    previous speaker's send. ``None`` when no specific envelope drove
    the expectation (e.g. fresh channel, expected speaker is the
    creator with nothing yet posted).
    """

    agent_id: str
    triggering_envelope_id: str | None = None


class ChannelAdapter(Protocol):
    """Code half of the manifest/adapter split.

    Adapters are looked up at channel-create time by
    ``(manifest.type, manifest.version)``. Re-registering an adapter
    at a new version does not retroactively change in-flight channels
    — they keep their original manifest snapshot.
    """

    manifest: ChannelManifest

    def initial_state(self, metadata: ChannelMetadata) -> AdapterState:
        """Empty state for a fresh channel."""
        ...

    def fold(self, envelope: Envelope, state: AdapterState) -> AdapterState:
        """Append ``envelope`` into the derived state. Pure function.

        Called once per WAL append by the hub. Must be deterministic so
        ``Hub.hydrate()`` can re-fold from disk.
        """
        ...

    def validate_create(self, metadata: ChannelMetadata) -> None:
        """Raise on invalid creation (bad participant count, missing knobs, ...)."""
        ...

    def validate_send(
        self,
        metadata: ChannelMetadata,
        envelope: Envelope,
        state: AdapterState,
    ) -> None:
        """Raise if this envelope is not allowed by the protocol at this point.

        Receives state BEFORE ``fold(envelope, ...)`` runs.
        """
        ...

    def on_accepted(
        self,
        metadata: ChannelMetadata,
        envelope: Envelope,
        state: AdapterState,
    ) -> AdapterResult:
        """Decide post-accept transitions.

        Receives state AFTER ``fold(envelope, ...)`` has run.
        """
        ...

    def expected_next(
        self,
        metadata: ChannelMetadata,
        state: AdapterState,
    ) -> "ExpectedTurn | None":
        """Identify the participant the protocol expects to act next.

        Free-form channels (no turn ordering) and channels that have
        completed their protocol cycle return ``None``. Turn-taking
        adapters return ``ExpectedTurn(agent_id, triggering_envelope_id)``
        so the hub can answer ``pending_turns_for(agent_id)`` and a
        reconnecting client can re-fire its notify handler against the
        triggering envelope.
        """
        ...

    def default_view_policy(
        self,
        metadata: ChannelMetadata,
        participant_id: str,
    ) -> ViewPolicy:
        """Per-participant default projection for this channel type."""
        ...

    def extract_turn_input(self, envelope: Envelope) -> "str | Input | list[Input] | None":
        """Decode an inbound substantive envelope into the input the
        next speaker's LLM should receive on its turn.

        Return ``None`` (or empty string) for envelopes this adapter
        doesn't act on — the handler will skip the round.

        The default behaviour (``default_extract_turn_input``) handles
        ``EV_TEXT``. Adapters that emit additional substantive event
        types (e.g. ``EV_PACKET`` for the workflow adapter) override
        to decode those.
        """
        ...

    def build_round_envelope(
        self,
        metadata: ChannelMetadata,
        sender_id: str,
        reply: "AgentReply",
        events: "list[BaseEvent]",
        state: AdapterState,
        hub: "Hub",
    ) -> Envelope | None:
        """Build the envelope that captures one ``Agent.ask`` round.

        Called by the handler after ``Agent.ask`` returns. Adapters
        encode the round result into the envelope shape they expect:
        the default (``default_build_round_envelope``) emits
        ``EV_TEXT(reply.body)`` if non-empty, else ``None`` (silent
        round, no envelope posted).

        Returning ``None`` means "this round produced nothing worth
        recording" — the caller skips the post.
        """
        ...

    def render_envelope(self, envelope: Envelope) -> str | None:
        """Project ``envelope`` to its LLM-visible string for view policies.

        Called by ``ViewPolicy.project`` once per envelope in the WAL
        slice the participant should see. Adapters that emit only
        ``EV_TEXT`` delegate to ``default_render_envelope``; adapters
        with richer round-end shapes (e.g. ``WorkflowAdapter`` with
        ``EV_PACKET``) handle their own types and fall through to
        the default for the universal cases.

        Returning ``None`` means "skip this envelope in the projection"
        (non-substantive event types, malformed payload, etc.).
        """
        ...

    def tools_for(
        self,
        client: "AgentClient",
        metadata: ChannelMetadata,
        state: AdapterState,
        participant_id: str,
    ) -> "list[Tool]":
        """Return the LLM tools this adapter offers a participant.

        Resolved per turn by the default notify handler and merged
        into the per-call ``tools=`` override passed to ``agent.ask``.
        Adapters that take no LLM input (e.g. workflow, where handoff
        tools are user-authored) return ``[]``.

        Adapters gate on ``state`` and ``participant_id`` — e.g. the
        discussion adapter offers ``say`` only when it is this
        participant's round; consulting offers ``say`` only to the
        participant whose turn it is in the 1Q1R handshake.
        """
        ...

    def build_text_envelope(
        self,
        channel_id: str,
        sender_id: str,
        text: str,
        *,
        audience: list[str] | None = None,
        causation_id: str | None = None,
    ) -> Envelope:
        """Construct an ``EV_TEXT`` envelope shaped for this adapter.

        Layer-2 helper — any client (AG2 ``AgentClient``, ``HumanClient``,
        non-AG2 bridge) uses this to build correctly-shaped envelopes
        without going through the LLM tool decorator system. Default
        impl (``default_build_text_envelope``) is what consulting /
        conversation / discussion adapters use; workflow overrides to
        wrap text in ``EV_PACKET``.
        """
        ...

    def build_packet_envelope(
        self,
        channel_id: str,
        sender_id: str,
        body: str,
        *,
        handoff: "Handoff | None" = None,
        context_set: dict | None = None,
        audience: list[str] | None = None,
        causation_id: str | None = None,
    ) -> Envelope:
        """Construct an ``EV_PACKET`` envelope shaped for this adapter.

        Layer-2 helper. Workflow uses ``handoff`` / ``context_set`` to
        encode routing + variable mutations into one atomic round
        capture; other adapters typically delegate to
        ``default_build_packet_envelope`` (no handoff, no context_set).
        """
        ...


def default_extract_turn_input(envelope: Envelope) -> str | None:
    """Default ``extract_turn_input``: decode ``EV_TEXT`` only.

    Adapters that don't handle additional substantive event types
    delegate to this helper from their ``extract_turn_input``.
    """
    if envelope.event_type == EV_TEXT:
        text = envelope.event_data.get("text", "")
        return text if isinstance(text, str) else None
    return None


def default_build_round_envelope(
    metadata: ChannelMetadata,
    sender_id: str,
    reply: "AgentReply",
    events: "list[BaseEvent]",
    state: AdapterState,
    hub: "Hub",
) -> Envelope | None:
    """Default ``build_round_envelope``: emit ``EV_TEXT(body)`` or
    ``None``.

    Adapters that don't have a richer round-end shape delegate to
    this helper from their ``build_round_envelope``.
    """
    body = reply.body or ""
    if not body:
        return None
    return Envelope(
        channel_id=metadata.channel_id,
        sender_id=sender_id,
        audience=None,
        event_type=EV_TEXT,
        event_data={"text": body},
    )


def default_render_envelope(envelope: Envelope) -> str | None:
    """Default ``render_envelope``: project ``EV_TEXT`` only.

    Returns the envelope's text payload, or ``None`` for any other
    event type (so the view skips it). Adapters that emit additional
    substantive event types override ``render_envelope`` and fall
    through to this helper for ``EV_TEXT``.
    """
    if envelope.event_type == EV_TEXT:
        text = envelope.event_data.get("text", "")
        return text if isinstance(text, str) else None
    return None


def default_tools_for(
    client: "AgentClient",
    metadata: ChannelMetadata,
    state: AdapterState,
    participant_id: str,
) -> "list[Tool]":
    """Default ``tools_for``: return ``[]``.

    Adapters that accept LLM-driven sends override to return the
    appropriate per-channel tool set (e.g. ``[make_say_tool(client)]``
    for free-form text channels).
    """
    return []


def default_expected_next(
    metadata: ChannelMetadata,
    state: AdapterState,
) -> "ExpectedTurn | None":
    """Default ``expected_next``: no specific participant expected.

    Free-form adapters (e.g. conversation) delegate to this from
    ``expected_next``. Adapters that enforce turn ordering override
    to surface the expected speaker.
    """
    return None


def default_build_text_envelope(
    channel_id: str,
    sender_id: str,
    text: str,
    *,
    audience: list[str] | None = None,
    causation_id: str | None = None,
) -> Envelope:
    """Default ``build_text_envelope``: emit ``EV_TEXT(text)``."""
    return Envelope(
        channel_id=channel_id,
        sender_id=sender_id,
        audience=audience,
        event_type=EV_TEXT,
        event_data={"text": text},
        causation_id=causation_id,
    )


def default_build_packet_envelope(
    channel_id: str,
    sender_id: str,
    body: str,
    *,
    handoff: "Handoff | None" = None,
    context_set: dict | None = None,
    audience: list[str] | None = None,
    causation_id: str | None = None,
) -> Envelope:
    """Default ``build_packet_envelope``: emit ``EV_PACKET`` with body +
    optional routing.

    ``routing`` carries ``{"kind": "handoff", "target": ..., "reason": ...}``
    when ``handoff`` is set. ``context_set`` populates ``context``.
    """
    event_data: dict[str, object] = {"body": body}
    if handoff is not None:
        routing: dict[str, object] = {"kind": "handoff", "target": handoff.target}
        if handoff.reason:
            routing["reason"] = handoff.reason
        event_data["routing"] = routing
    if context_set:
        event_data["context"] = dict(context_set)
    return Envelope(
        channel_id=channel_id,
        sender_id=sender_id,
        audience=audience,
        event_type=EV_PACKET,
        event_data=event_data,
        causation_id=causation_id,
    )
