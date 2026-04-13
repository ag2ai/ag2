# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Envelope — wire format for every hub-routed message.

An envelope wraps an **event** (a stable-named dict payload) with the network
metadata needed to thread it through sessions, inboxes, and subscribers.
Event types use stable registered names (``ag2.msg.text``, ``ag2.session.invite``
...) instead of Python-qualified class names so the wire format is safe
across language boundaries.

Phase 1 ships a fixed set of event types; Phase 2 introduces a proper
``EventRegistry`` for user-defined types.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Built-in event type names
# ---------------------------------------------------------------------------

EV_TEXT = "ag2.msg.text"
EV_SESSION_INVITE = "ag2.session.invite"
EV_SESSION_INVITE_ACK = "ag2.session.invite_ack"
EV_SESSION_INVITE_REJECT = "ag2.session.invite_reject"
EV_SESSION_CLOSED = "ag2.session.closed"
EV_SESSION_OPENED = "ag2.session.opened"
EV_ERROR = "ag2.error"

# Phase 4 — Task events. Task creation is a direct ``Hub.create_task`` call
# (no wire envelope), but every subsequent state transition rides the session
# WAL so subscribers and cold-restart hydrate both see the full trail.
EV_TASK_ASSIGNED = "ag2.task.assigned"
EV_TASK_PHASE_ENTERED = "ag2.task.phase_entered"
EV_TASK_PHASE_COMPLETED = "ag2.task.phase_completed"
EV_TASK_PROGRESS = "ag2.task.progress"
EV_TASK_RESULT = "ag2.task.result"
EV_TASK_ERROR = "ag2.task.error"
EV_TASK_CANCELLED = "ag2.task.cancelled"
EV_TASK_EXPIRED = "ag2.task.expired"

TASK_EVENT_TYPES: frozenset[str] = frozenset(
    {
        EV_TASK_ASSIGNED,
        EV_TASK_PHASE_ENTERED,
        EV_TASK_PHASE_COMPLETED,
        EV_TASK_PROGRESS,
        EV_TASK_RESULT,
        EV_TASK_ERROR,
        EV_TASK_CANCELLED,
        EV_TASK_EXPIRED,
    }
)

TASK_TERMINAL_EVENT_TYPES: frozenset[str] = frozenset(
    {
        EV_TASK_RESULT,
        EV_TASK_ERROR,
        EV_TASK_CANCELLED,
        EV_TASK_EXPIRED,
    }
)


# Priority is a tagged enum — the wire format is a plain string so Phase 3
# WsLink and custom backends don't need to teach an Enum decoder. Operators
# that want custom priority schemes register them per-hub; the three
# built-in values below are the defaults Phase 1 ships.
Priority = Literal["background", "normal", "urgent"]
_VALID_PRIORITIES: frozenset[str] = frozenset({"background", "normal", "urgent"})


@dataclass(slots=True)
class Envelope:
    """Network-level wrapper around an event payload.

    ``envelope_id`` is hub-stamped at accept time; callers leave it ``None``
    on construction and receive a filled copy from the hub accept response.
    """

    session_id: str
    sender_id: str
    event_type: str
    event_data: dict[str, Any] = field(default_factory=dict)

    envelope_id: str | None = None
    recipient_id: str | None = None
    task_id: str | None = None
    causation_id: str | None = None
    trace_id: str | None = None
    priority: Priority = "normal"
    created_at: str | None = None
    ttl_seconds: int | None = None
    idempotency_key: str | None = None
    depth: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.priority not in _VALID_PRIORITIES:
            raise ValueError(
                f"priority must be one of {sorted(_VALID_PRIORITIES)}, got {self.priority!r}"
            )

    # ---------------------- convenience constructors ----------------------

    @classmethod
    def text(
        cls,
        *,
        session_id: str,
        sender_id: str,
        content: str,
        recipient_id: str | None = None,
        causation_id: str | None = None,
        trace_id: str | None = None,
        priority: Priority = "normal",
        metadata: dict[str, Any] | None = None,
    ) -> Envelope:
        return cls(
            session_id=session_id,
            sender_id=sender_id,
            event_type=EV_TEXT,
            event_data={"content": content},
            recipient_id=recipient_id,
            causation_id=causation_id,
            trace_id=trace_id,
            priority=priority,
            metadata=dict(metadata or {}),
        )

    # --------------------------- (de)serialization ------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "envelope_id": self.envelope_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "causation_id": self.causation_id,
            "trace_id": self.trace_id,
            "priority": self.priority,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "idempotency_key": self.idempotency_key,
            "depth": self.depth,
            "event": {"type": self.event_type, "data": dict(self.event_data)},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Envelope:
        event = data.get("event") or {}
        return cls(
            envelope_id=data.get("envelope_id"),
            session_id=data["session_id"],
            sender_id=data["sender_id"],
            recipient_id=data.get("recipient_id"),
            task_id=data.get("task_id"),
            causation_id=data.get("causation_id"),
            trace_id=data.get("trace_id"),
            priority=data.get("priority", "normal"),
            created_at=data.get("created_at"),
            ttl_seconds=data.get("ttl_seconds"),
            idempotency_key=data.get("idempotency_key"),
            depth=int(data.get("depth", 0)),
            event_type=event.get("type", EV_TEXT),
            event_data=dict(event.get("data", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> Envelope:
        return cls.from_dict(json.loads(payload))

    # ------------------------------ helpers --------------------------------

    def content(self) -> str:
        """Return the text content for ``ag2.msg.text`` envelopes.

        Raises ``KeyError`` if the event type is not text — callers that want
        a best-effort preview should reach into ``event_data`` themselves.
        """

        if self.event_type != EV_TEXT:
            raise KeyError(f"Envelope is not a text event: {self.event_type}")
        return str(self.event_data["content"])
