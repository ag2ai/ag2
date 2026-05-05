# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Envelope — the wire shape for every message between agents.

Every Agent-to-Agent exchange (post-M2) happens inside a session and
the carrier is an ``Envelope``. Envelopes are JSON-serialisable, hub-
stamped at ``post_envelope``, and persisted to the per-session WAL.
``audience`` is the addressing primitive: ``None`` broadcasts within
the session, a list targets a subset.

Streaming chunks use a separate transport-level ``chunk`` frame (Phase
2 surface) and are not envelopes — they bypass the WAL entirely.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

__all__ = (
    "EV_ERROR",
    "EV_EXPECTATION_VIOLATED",
    "EV_PARTICIPANT_REMOVED",
    "EV_PEER_RECONNECTED",
    "EV_PEER_UNREACHABLE",
    "EV_SESSION_CLOSED",
    "EV_SESSION_EXPIRED",
    "EV_SESSION_IDLE",
    "EV_SESSION_INVITE",
    "EV_SESSION_INVITE_ACK",
    "EV_SESSION_INVITE_REJECT",
    "EV_SESSION_OPENED",
    "EV_SESSION_QUORUM_CHANGED",
    "EV_TASK_ERROR",
    "EV_TASK_EXPIRED",
    "EV_TASK_PROGRESS",
    "EV_TASK_RESULT",
    "EV_TASK_STALLED",
    "EV_TASK_STARTED",
    "EV_TEXT",
    "Envelope",
    "Priority",
    "visible_to",
)


Priority = Literal["background", "normal", "urgent"]


# ── Stable event-type names ──────────────────────────────────────────────────
# V1 ships a fixed set; new names are added in code, not at runtime. User-
# defined event types may be posted with arbitrary strings (no namespace
# check in V1) — the framework only special-cases the names below.

EV_TEXT = "ag2.msg.text"

EV_SESSION_INVITE = "ag2.session.invite"
EV_SESSION_INVITE_ACK = "ag2.session.invite.ack"
EV_SESSION_INVITE_REJECT = "ag2.session.invite.reject"
EV_SESSION_OPENED = "ag2.session.opened"
EV_SESSION_CLOSED = "ag2.session.closed"
EV_SESSION_EXPIRED = "ag2.session.expired"
EV_SESSION_IDLE = "ag2.session.idle"
EV_SESSION_QUORUM_CHANGED = "ag2.session.quorum_changed"

EV_TASK_STARTED = "ag2.task.started"
EV_TASK_PROGRESS = "ag2.task.progress"
EV_TASK_RESULT = "ag2.task.result"
EV_TASK_ERROR = "ag2.task.error"
EV_TASK_EXPIRED = "ag2.task.expired"
EV_TASK_STALLED = "ag2.task.stalled"

EV_PEER_UNREACHABLE = "ag2.peer.unreachable"
EV_PEER_RECONNECTED = "ag2.peer.reconnected"

EV_EXPECTATION_VIOLATED = "ag2.expectation.violated"
EV_PARTICIPANT_REMOVED = "ag2.participant.removed"

EV_ERROR = "ag2.error"


@dataclass(slots=True)
class Envelope:
    """Wire shape for every Agent-to-Agent message.

    Field semantics:

    * ``envelope_id`` — hub-stamped on accept (UUID7-like). Sender-side
      construction leaves this empty; ``Hub.post_envelope`` populates.
    * ``audience`` — ``None`` broadcasts within the session; a list
      targets a subset. Hub WAL stores the full envelope regardless of
      addressing (audit + debug); ``notify`` lands only on listed peers.
    * ``causation_id`` — envelope this is responding to. Used by view
      policies to thread replies to their prompts.
    * ``depth`` — delegation hop count. Hub auto-increments on the
      reply path; ``Rule.limits.delegation_depth`` caps it.
    * ``ttl_seconds`` — per-envelope TTL. ``None`` defers to the
      session's ``expires_at``.
    * ``idempotency_key`` — Phase 3 dedup key; ignored in V1.
    """

    session_id: str
    sender_id: str
    audience: list[str] | None
    event_type: str
    event_data: dict[str, Any]

    envelope_id: str = ""  # hub-stamped on accept
    task_id: str | None = None
    causation_id: str | None = None
    trace_id: str | None = None
    priority: Priority = "normal"
    depth: int = 0
    idempotency_key: str | None = None  # Phase 3

    created_at: str = ""  # ISO-Z, hub-stamped on accept
    ttl_seconds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict (every field round-trips byte-stable)."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialise to JSON. Sort keys so cross-process hashes match."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Envelope":
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> "Envelope":
        return cls.from_dict(json.loads(text))


def visible_to(envelope: Envelope, participant_id: str) -> bool:
    """Pure delivery / view-filtering predicate.

    Sender always sees their own envelope; broadcasts (``audience=None``)
    are visible to all session participants; subset addressing is
    visible only to listed peers.
    """
    if envelope.sender_id == participant_id:
        return True
    if envelope.audience is None:
        return True
    return participant_id in envelope.audience
