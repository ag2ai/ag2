# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Session types and per-session metadata.

Six session types are declared up front so rules and handlers can reference
them by stable name. Phase 1 ships adapters for three of them (consulting,
conversation, notification); Phase 2 adds broadcast / discussion / auction
and opens the set to operator-registered custom types via
:meth:`Hub.register_adapter` (§5.5). Every session — regardless of type —
shares the same :class:`SessionMetadata` on disk so adapters are strictly
swappable.

:class:`SessionType` is a canonical namespace of built-in type names. It
subclasses ``str`` so members double as the wire value, and
:attr:`SessionMetadata.type` is annotated as ``str`` — the hub dispatches
adapters by string, letting operators register types that are not members
of the built-in enum.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SessionType(str, Enum):
    """Canonical namespace of built-in session type names.

    Members subclass ``str``, so ``SessionType.CONSULTING == "consulting"``
    and the enum members are interchangeable with their string values
    anywhere a session type is expected. Operators that ship custom types
    via :meth:`Hub.register_adapter` pass plain strings — the enum is a
    convenience, not a closed set.
    """

    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    CONSULTING = "consulting"
    CONVERSATION = "conversation"
    DISCUSSION = "discussion"
    AUCTION = "auction"


class SessionState(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSING = "closing"
    CLOSED = "closed"
    EXPIRED = "expired"


class ParticipantRole(str, Enum):
    INITIATOR = "initiator"
    RESPONDENT = "respondent"
    PARTICIPANT = "participant"
    OBSERVER = "observer"


@dataclass(slots=True)
class Participant:
    actor_id: str
    role: ParticipantRole = ParticipantRole.PARTICIPANT
    joined_at: str | None = None
    order: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "actor_id": self.actor_id,
            "role": self.role.value,
        }
        if self.joined_at is not None:
            data["joined_at"] = self.joined_at
        if self.order is not None:
            data["order"] = self.order
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Participant:
        return cls(
            actor_id=data["actor_id"],
            role=ParticipantRole(data.get("role", ParticipantRole.PARTICIPANT.value)),
            joined_at=data.get("joined_at"),
            order=data.get("order"),
        )


@dataclass(slots=True)
class SessionMetadata:
    session_id: str
    type: str
    creator_id: str
    participants: list[Participant]
    state: SessionState = SessionState.PENDING
    visibility: str = "members-only"
    created_at: str | None = None
    expires_at: str | None = None
    labels: dict[str, Any] = field(default_factory=dict)
    ordering: str | None = None
    on_failure: str | None = None
    parent_session_id: str | None = None
    closed_at: str | None = None
    close_reason: str | None = None
    # Phase 3b archival. Non-null ``archived_at`` means the WAL has
    # been moved to ``hub/archive/sessions/{id}/wal.jsonl`` and the
    # session no longer owns an active WAL in ``hub/sessions/``.
    archived_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "type": _type_value(self.type),
            "creator_id": self.creator_id,
            "participants": [p.to_dict() for p in self.participants],
            "state": self.state.value,
            "visibility": self.visibility,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "labels": dict(self.labels),
            "ordering": self.ordering,
            "on_failure": self.on_failure,
            "parent_session_id": self.parent_session_id,
            "closed_at": self.closed_at,
            "close_reason": self.close_reason,
            "archived_at": self.archived_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMetadata:
        return cls(
            session_id=data["session_id"],
            type=_type_value(data["type"]),
            creator_id=data["creator_id"],
            participants=[Participant.from_dict(p) for p in data.get("participants", [])],
            state=SessionState(data.get("state", SessionState.PENDING.value)),
            visibility=data.get("visibility", "members-only"),
            created_at=data.get("created_at"),
            expires_at=data.get("expires_at"),
            labels=dict(data.get("labels", {})),
            ordering=data.get("ordering"),
            on_failure=data.get("on_failure"),
            parent_session_id=data.get("parent_session_id"),
            closed_at=data.get("closed_at"),
            close_reason=data.get("close_reason"),
            archived_at=data.get("archived_at"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> SessionMetadata:
        return cls.from_dict(json.loads(payload))

    def participant_ids(self) -> list[str]:
        return [p.actor_id for p in self.participants]

    def has_participant(self, actor_id: str) -> bool:
        return any(p.actor_id == actor_id for p in self.participants)

    def participant(self, actor_id: str) -> Participant | None:
        for p in self.participants:
            if p.actor_id == actor_id:
                return p
        return None

    def copy(self) -> SessionMetadata:
        data = asdict(self)
        data["type"] = _type_value(data["type"])
        data["state"] = SessionState(data["state"])
        data["participants"] = [
            Participant(
                actor_id=p["actor_id"],
                role=ParticipantRole(p["role"]),
                joined_at=p["joined_at"],
                order=p["order"],
            )
            for p in data["participants"]
        ]
        return SessionMetadata(**data)


def _type_value(session_type: SessionType | str) -> str:
    """Return the canonical string value for a session type.

    Python 3.11+ makes ``str(SessionType.CONSULTING)`` return the
    qualified name (``"SessionType.CONSULTING"``) instead of the
    underlying value (``"consulting"``); we go through ``.value`` for
    enum members and accept any other string as-is.
    """

    if isinstance(session_type, SessionType):
        return session_type.value
    return str(session_type)
