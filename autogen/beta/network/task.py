# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Task primitives — Phase 4 network `Task` type.

A network :class:`Task` is a session-owned, hub-tracked unit of work with a
first-class state machine, optional phases, TTL enforcement, and a durable
metadata record on the extended :class:`KnowledgeStore`. Tasks live *inside*
exactly one session (§6) — they are never free-floating. Every task event
carries ``task_id`` on the envelope and is routed through the hub's task-event
branch in :meth:`Hub.post_envelope`, which bypasses the session adapter's
delivery rules so the session-type lifecycle (e.g. consulting's 1Q1R
auto-close) stays orthogonal to task lifecycle.

Framework-core :func:`run_subtask` / :func:`run_subtasks` (see
``autogen/beta/actor.py``) are deliberately unrelated: they spawn a private
child ``Agent`` with no hub, no envelopes, and no observability. See §6.4 of
``design/network_v3_redesign.md`` for the full distinction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskState(str, Enum):
    """Hub-owned task lifecycle state.

    Transition rules (enforced by the hub in :meth:`Hub._validate_task_event`):

    * ``created → running``  — first phase_entered / progress / result.
    * ``running → running``  — phase_entered / phase_completed / progress.
    * ``running → paused``   — reserved for Phase 6 pause support.
    * ``running → completed`` — on ``ag2.task.result``.
    * ``running → failed``   — on ``ag2.task.error``.
    * ``running → cancelled`` — on ``ag2.task.cancelled``.
    * ``running → expired``  — on ``ag2.task.expired`` (hub-emitted TTL sweep).
    * ``paused → running``   — reserved for Phase 6 resume.
    * ``paused → cancelled`` — requester cancel on a paused task.
    * Terminal → anything    — rejected. Terminal states are final.
    """

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_TASK_STATES: frozenset[TaskState] = frozenset(
    {
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELLED,
        TaskState.EXPIRED,
    }
)


@dataclass(slots=True)
class TaskPhase:
    """A named step in a multi-phase task plan.

    Phases are optional — a task with an empty ``phases`` list is a single-shot
    unit of work that goes straight from ``created`` to ``completed`` (or
    ``failed``). For multi-phase tasks, the owner emits ``ag2.task.phase_entered``
    to advance ``TaskMetadata.current_phase`` and ``ag2.task.phase_completed``
    to stamp ``completed_at`` — phase ordering is enforced by the owner (the
    hub only validates that every referenced phase id exists on the task).
    """

    id: str
    description: str = ""
    started_at: str | None = None
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskPhase:
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )


@dataclass(slots=True)
class TaskSpec:
    """User-provided task description handed to the owner's task handler.

    ``spec_type`` is the key the owner's :class:`ActorClient` uses to pick a
    task handler via :meth:`ActorClient.on_task` — an empty string routes to
    the default ``"*"`` handler. ``payload`` is an arbitrary JSON-serializable
    dict the handler interprets any way it wants.
    """

    title: str
    description: str = ""
    phases: list[TaskPhase] = field(default_factory=list)
    spec_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "phases": [p.to_dict() for p in self.phases],
            "spec_type": self.spec_type,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSpec:
        return cls(
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            phases=[TaskPhase.from_dict(p) for p in data.get("phases", [])],
            spec_type=str(data.get("spec_type", "")),
            payload=dict(data.get("payload", {})),
        )

    def phase_ids(self) -> list[str]:
        return [p.id for p in self.phases]


@dataclass(slots=True)
class TaskMetadata:
    """The hub's durable record for a single task.

    Rewritten on every state change at ``hub/tasks/{task_id}/metadata.json``.
    ``result`` holds the terminal value for ``completed`` tasks; ``error``
    holds the terminal message for ``failed`` tasks. ``progress`` is a
    free-form dict merged on every ``ag2.task.progress`` event.
    """

    task_id: str
    session_id: str
    owner_id: str
    requester_id: str
    spec: TaskSpec
    state: TaskState
    created_at: str
    expires_at: str
    current_phase: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    last_progress_at: str | None = None
    result: Any | None = None
    error: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_TASK_STATES

    def copy(self) -> TaskMetadata:
        return TaskMetadata.from_dict(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "owner_id": self.owner_id,
            "requester_id": self.requester_id,
            "spec": self.spec.to_dict(),
            "state": self.state.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "current_phase": self.current_phase,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "last_progress_at": self.last_progress_at,
            "result": self.result,
            "error": self.error,
            "progress": dict(self.progress),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskMetadata:
        return cls(
            task_id=data["task_id"],
            session_id=data["session_id"],
            owner_id=data["owner_id"],
            requester_id=data["requester_id"],
            spec=TaskSpec.from_dict(data.get("spec", {})),
            state=TaskState(data.get("state", TaskState.CREATED.value)),
            created_at=data["created_at"],
            expires_at=data["expires_at"],
            current_phase=data.get("current_phase"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            last_progress_at=data.get("last_progress_at"),
            result=data.get("result"),
            error=data.get("error"),
            progress=dict(data.get("progress", {})),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> TaskMetadata:
        return cls.from_dict(json.loads(payload))
