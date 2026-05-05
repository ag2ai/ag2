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
from typing import Protocol

from ..envelope import Envelope
from ..session import SessionManifest, SessionMetadata, SessionState
from ..views.base import ViewPolicy

__all__ = ("AdapterResult", "AdapterState", "SessionAdapter")


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
