# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Envelope — the wire shape for every message between agents.

Every Agent-to-Agent exchange happens inside a session and the carrier
is an ``Envelope``. Envelopes are JSON-serialisable, hub-stamped at
``post_envelope``, and persisted to the per-session WAL. ``audience``
is the addressing primitive: ``None`` broadcasts within the session,
a list targets a subset.
"""

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

__all__ = (
    "EV_EXPECTATION_VIOLATED",
    "EV_HANDOFF",
    "EV_SESSION_CLOSED",
    "EV_SESSION_EXPIRED",
    "EV_SESSION_INVITE",
    "EV_SESSION_INVITE_ACK",
    "EV_SESSION_INVITE_REJECT",
    "EV_SESSION_OPENED",
    "EV_TEXT",
    "Envelope",
    "Priority",
    "visible_to",
)


Priority = Literal["background", "normal", "urgent"]


# ── Stable event-type names ──────────────────────────────────────────────────
# Fixed set; new names are added in code, not at runtime. User-defined event
# types may be posted with arbitrary strings — the framework only
# special-cases the names below.

EV_TEXT = "ag2.msg.text"

# Tool-driven workflow transition signal. ``event_data`` carries
# ``{"tool": <tool_name>, "reason": <free-form>}``. Read by
# ``WorkflowAdapter``'s ``ToolCalled`` condition; any adapter that wants
# tool-driven transitions can read it the same way.
EV_HANDOFF = "ag2.handoff"

EV_SESSION_INVITE = "ag2.session.invite"
EV_SESSION_INVITE_ACK = "ag2.session.invite.ack"
EV_SESSION_INVITE_REJECT = "ag2.session.invite.reject"
EV_SESSION_OPENED = "ag2.session.opened"
EV_SESSION_CLOSED = "ag2.session.closed"
EV_SESSION_EXPIRED = "ag2.session.expired"

EV_EXPECTATION_VIOLATED = "ag2.expectation.violated"

# Idle-detection rides on ``max_silence`` expectations; task lifecycle is
# mirrored as Python events on the agent's own stream
# (see :mod:`autogen.beta.network.task_mirror`).


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
    * ``idempotency_key`` — dedup key reserved for cross-process
      transports; the in-process hub serialises under a per-session lock
      and ignores it.
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
    idempotency_key: str | None = None

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
