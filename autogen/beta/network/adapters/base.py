# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``SessionAdapter`` Protocol + ``AdapterState`` marker + ``AdapterResult``.

Key invariants:

* Adapters are stateless and pure.
* Every decision derives from ``(metadata, AdapterState)``.
* ``validate_send`` and ``on_accepted`` are O(1), not O(WAL) — the hub
  passes the cached state in.
* ``fold`` is called once per WAL append by the hub, and called
  repeatedly during ``Hub.hydrate()`` to rebuild state from disk. It
  must be a pure function.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from autogen.beta.events.input_events import Input

from ..envelope import EV_TEXT, Envelope
from ..session import SessionManifest, SessionMetadata, SessionState
from ..views.base import ViewPolicy

if TYPE_CHECKING:
    from autogen.beta.agent import AgentReply
    from autogen.beta.events import BaseEvent

    from ..hub.core import Hub

__all__ = (
    "AdapterResult",
    "AdapterState",
    "SessionAdapter",
    "default_build_round_envelope",
    "default_extract_turn_input",
    "default_render_envelope",
)


class AdapterState(Protocol):
    """Marker Protocol — concrete adapters define their own dataclass.

    Empty by design: the hub treats adapter state opaquely and only
    passes it back into ``fold`` / ``validate_send`` / ``on_accepted``.
    """


@dataclass(slots=True)
class AdapterResult:
    """What an adapter wants the hub to do after accepting an envelope.

    ``next_state=None`` leaves the session in its current state. The
    hub broadcasts ``EV_SESSION_CLOSED`` / ``EV_SESSION_EXPIRED`` when
    transitioning to a terminal state.
    """

    next_state: SessionState | None = None
    auto_close_reason: str = ""


class SessionAdapter(Protocol):
    """Code half of the manifest/adapter split.

    Adapters are looked up at session-create time by
    ``(manifest.type, manifest.version)``. Re-registering an adapter
    at a new version does not retroactively change in-flight sessions
    — they keep their original manifest snapshot.
    """

    manifest: SessionManifest

    def initial_state(self, metadata: SessionMetadata) -> AdapterState:
        """Empty state for a fresh session."""
        ...

    def fold(self, envelope: Envelope, state: AdapterState) -> AdapterState:
        """Append ``envelope`` into the derived state. Pure function.

        Called once per WAL append by the hub. Must be deterministic so
        ``Hub.hydrate()`` can re-fold from disk.
        """
        ...

    def validate_create(self, metadata: SessionMetadata) -> None:
        """Raise on invalid creation (bad participant count, missing knobs, ...)."""
        ...

    def validate_send(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        state: AdapterState,
    ) -> None:
        """Raise if this envelope is not allowed by the protocol at this point.

        Receives state BEFORE ``fold(envelope, ...)`` runs.
        """
        ...

    def on_accepted(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        state: AdapterState,
    ) -> AdapterResult:
        """Decide post-accept transitions.

        Receives state AFTER ``fold(envelope, ...)`` has run.
        """
        ...

    def default_view_policy(
        self,
        metadata: SessionMetadata,
        participant_id: str,
    ) -> ViewPolicy:
        """Per-participant default projection for this session type."""
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
        metadata: SessionMetadata,
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
    metadata: SessionMetadata,
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
        session_id=metadata.session_id,
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
