# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""SessionAdapter protocol.

A session adapter owns the per-session-type delivery rules: who may send
when, when the session is considered done, and how many participants it
requires. Adapters are pure — they read session metadata and the candidate
envelope, raise on violations, and return a structured result describing
state transitions. The hub is responsible for everything else
(persistence, dispatch, notify frames).

:attr:`SessionAdapter.session_type` is a plain string. Built-in adapters
use the :class:`SessionType` enum members (which subclass ``str``) for the
canonical names; operator-shipped custom types pass arbitrary string
values — see :meth:`Hub.register_adapter` (§5.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..envelope import Envelope
from ..session_types import SessionMetadata, SessionState


@dataclass(slots=True)
class AdapterResult:
    """What the adapter wants the hub to do after accepting an envelope.

    ``follow_ups`` carries hub-generated system envelopes (e.g. the next
    speaker's turn signal in a ``discussion(ordering="static")``) that the
    hub should append and deliver after the accepted envelope's fan-out.
    """

    next_state: SessionState | None = None
    close_reason: str | None = None
    follow_ups: list[Envelope] = field(default_factory=list)


@runtime_checkable
class SessionAdapter(Protocol):
    """Per-type session delivery rules."""

    session_type: str

    def validate_create(self, metadata: SessionMetadata) -> None:
        """Raise :class:`SessionTypeError` if participant shape is wrong."""
        ...

    def validate_send(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        prior_envelopes: list[Envelope],
    ) -> None:
        """Raise :class:`SessionTypeError` if the envelope is not allowed.

        ``prior_envelopes`` is the current session WAL tail (user envelopes
        only — system invites/acks are filtered out by the hub before the
        adapter sees it).
        """
        ...

    def on_accepted(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        prior_envelopes: list[Envelope],
    ) -> AdapterResult:
        """Called after the hub has written the envelope to the WAL."""
        ...
