# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Envelope — the wire shape for every message between agents.

Every Agent-to-Agent exchange happens inside a channel and the carrier
is an ``Envelope``. Envelopes are JSON-serialisable, hub-stamped at
``post_envelope``, and persisted to the per-channel WAL. ``audience``
is the addressing primitive: ``None`` broadcasts within the channel,
a list targets a subset.
"""

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

__all__ = (
    "EV_CHANNEL_CLOSED",
    "EV_CHANNEL_EXPIRED",
    "EV_CHANNEL_INVITE",
    "EV_CHANNEL_INVITE_ACK",
    "EV_CHANNEL_INVITE_REJECT",
    "EV_CHANNEL_OPENED",
    "EV_CONTEXT_SET",
    "EV_EXPECTATION_VIOLATED",
    "EV_PACKET",
    "EV_TASK_CANCELLED",
    "EV_TASK_CANCEL_REQUEST",
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

# For the WorkflowAdapter, this is one agent's full ``Agent.ask`` round,
# captured atomically. The ``event_data`` shape:
#
#   {
#     "routing": {
#         "kind": "handoff" | "text",
#         "tool"?: str,    # the routing tool's name (handoff kind only)
#         "reason"?: str,  # human-readable trigger reason
#         "target"?: str,  # resolved next-speaker agent_id (handoff kind)
#     },
#     "context_updates": {"set": {...}, "delete": [...]},
#     "body": str,         # the agent's final text response, if any
#   }
#
EV_PACKET = "ag2.packet"

EV_CHANNEL_INVITE = "ag2.channel.invite"
EV_CHANNEL_INVITE_ACK = "ag2.channel.invite.ack"
EV_CHANNEL_INVITE_REJECT = "ag2.channel.invite.reject"
EV_CHANNEL_OPENED = "ag2.channel.opened"
EV_CHANNEL_CLOSED = "ag2.channel.closed"
EV_CHANNEL_EXPIRED = "ag2.channel.expired"

EV_EXPECTATION_VIOLATED = "ag2.expectation.violated"

# Channel-scoped context variable mutation.
# The ``event_data`` shape:
# ``{"set": {<key>: <value>, ...}, "delete": [<key>, ...]}``.
EV_CONTEXT_SET = "ag2.context.set"

# Task lifecycle is mirrored from the agent's stream into the hub's
# ``TaskMetadata`` cache directly (see
# :mod:`ag2.network.task_mirror`). The two wire envelopes
# below are the exception — cancellation has a peer-driven side that
# needs an addressable envelope.
#
# * ``EV_TASK_CANCEL_REQUEST`` — peer asks the owner to cancel.
#   ``event_data`` carries ``{"task_id": str, "reason": str}``.
#   The owner is free to honour by calling ``Task.cancel`` or to
#   ignore the request entirely.
# * ``EV_TASK_CANCELLED`` — owner-emitted terminal envelope a peer
#   handler can match on if it forwards cancellations into a channel.
#   ``event_data`` carries ``{"task_id": str, "reason": str}``.
EV_TASK_CANCEL_REQUEST = "ag2.task.cancel_request"
EV_TASK_CANCELLED = "ag2.task.cancelled"


@dataclass(slots=True)
class Envelope:
    """Wire shape for every Agent-to-Agent message.

    Field semantics:

    * ``envelope_id`` — hub-stamped on accept (UUID7-like). Sender-side
      construction leaves this empty; ``Hub.post_envelope`` populates.
    * ``audience`` — ``None`` broadcasts within the channel; a list
      targets a subset. Hub WAL stores the full envelope regardless of
      addressing (audit + debug); ``notify`` lands only on listed peers.
    * ``causation_id`` — envelope this is responding to. Used by view
      policies to thread replies to their prompts.
    * ``depth`` — delegation hop count. Hub auto-increments on the
      reply path; ``Rule.limits.delegation_depth`` caps it.
    * ``ttl_seconds`` — per-envelope TTL. ``None`` defers to the
      channel's ``expires_at``.
    * ``idempotency_key`` — dedup key reserved for cross-process
      transports; the in-process hub serialises under a per-channel lock
      and ignores it.
    """

    channel_id: str
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
    are visible to all channel participants; subset addressing is
    visible only to listed peers.
    """
    if envelope.sender_id == participant_id:
        return True
    if envelope.audience is None:
        return True
    return participant_id in envelope.audience
