# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Task — framework-core lifecycle primitive.

A ``Task`` is a wrapper any ``Agent`` (or other actor) can use to give a
unit of work a trackable lifecycle. The Task emits ``TaskStarted``,
``TaskProgress``, ``TaskCompleted``, ``TaskFailed``, ``TaskExpired``,
and ``TaskCancelled`` events on a bound stream so any observer (network
mirror, watcher, UI, test harness) can follow along without
participating in execution.

Tasks are agent-owned. The framework does not assign or schedule them.
Standalone usage requires no hub or network — events fly past harmlessly
if no observer subscribes. Network observation is layered on top via
``ag2.network``.

Tasks can checkpoint owner-defined resume state via
:meth:`Task.checkpoint`. When ``Task`` is constructed with
``checkpoint_store=`` (typically the network's
``HubBackedCheckpointStore``), the state is persisted under
``task_id``. Construction with ``resume_from=prior_task_id`` reads
back the prior checkpoint and exposes it via :attr:`resumed_state` so
the owner can pick up mid-flow after a restart.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated, Any, Protocol, runtime_checkable
from uuid import uuid4

from .annotations import Inject
from .context import ConversationContext
from .events import (
    TaskCancelled,
    TaskCompleted,
    TaskExpired,
    TaskFailed,
    TaskProgress,
    TaskStarted,
)
from .stream import MemoryStream

__all__ = (
    "TERMINAL_TASK_STATES",
    "CheckpointStore",
    "Task",
    "TaskInject",
    "TaskMetadata",
    "TaskSpec",
    "TaskState",
)


_TASK_DEP_KEY = "ag2.task"


class TaskState(str, Enum):
    """Lifecycle state of a ``Task``. Terminal states are immutable."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


TERMINAL_TASK_STATES: frozenset[TaskState] = frozenset({
    TaskState.COMPLETED,
    TaskState.FAILED,
    TaskState.EXPIRED,
    TaskState.CANCELLED,
})


@runtime_checkable
class CheckpointStore(Protocol):
    """Persistence Protocol for owner-supplied task resume state.

    Implementations write opaque JSON-serialisable dicts keyed by
    ``task_id`` and read them back. The framework never inspects the
    payload — the owner picks what to checkpoint, when, and how to
    interpret it on resume. ``read`` returns ``None`` if no checkpoint
    has been written for the given id.

    Networked agents typically use ``HubBackedCheckpointStore`` (in
    ``ag2.network.client.checkpoint``) which delegates to the
    hub. Standalone agents may supply any store satisfying this
    Protocol or omit checkpointing entirely.
    """

    async def write(self, task_id: str, state: dict[str, Any]) -> None: ...

    async def read(self, task_id: str) -> dict[str, Any] | None: ...


@dataclass(slots=True)
class TaskSpec:
    """What a ``Task`` is doing — title plus optional description and payload.

    ``capability`` tags the task with a capability name from the
    owning agent's ``Resume.claimed_capabilities``. When set, the
    network's ``TaskMirror`` calls ``Hub.record_observation`` on the
    terminal event so the matching ``Resume.observed[capability]``
    track record is updated. Untagged tasks are still observed but
    don't update any ``observed`` stat.
    """

    title: str
    description: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    capability: str | None = None


@dataclass(slots=True)
class TaskMetadata:
    """Mutable lifecycle record for a Task. Updated on state transitions."""

    task_id: str
    owner_id: str
    spec: TaskSpec
    state: TaskState
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    expires_at: str | None = None
    last_progress_at: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str = ""
    # Optional network association — set when an AgentClient mirrors this task.
    channel_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        ``state`` collapses to its enum value; ``spec`` flattens via
        ``dataclasses.asdict``. ``result`` must itself be
        JSON-serialisable (the owner's responsibility — the same
        contract a checkpoint payload carries)."""
        data = asdict(self)
        data["state"] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskMetadata":
        """Rebuild from :meth:`to_dict` output.

        Defensive against partial or legacy payloads (missing optional
        keys, a malformed ``spec``) so reloading a persisted task that
        predates a field addition does not raise."""
        spec_data = data.get("spec")
        if isinstance(spec_data, dict):
            capability_raw = spec_data.get("capability")
            spec = TaskSpec(
                title=str(spec_data.get("title", "")),
                description=str(spec_data.get("description", "")),
                payload=dict(spec_data.get("payload") or {}),
                capability=capability_raw if isinstance(capability_raw, str) else None,
            )
        else:
            spec = TaskSpec(title="")
        state_raw = data.get("state", TaskState.CREATED.value)
        state = TaskState(state_raw) if isinstance(state_raw, str) else state_raw
        return cls(
            task_id=str(data["task_id"]),
            owner_id=str(data["owner_id"]),
            spec=spec,
            state=state,
            created_at=str(data.get("created_at", "")),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            expires_at=data.get("expires_at"),
            last_progress_at=data.get("last_progress_at"),
            progress=dict(data.get("progress") or {}),
            result=data.get("result"),
            error=str(data.get("error", "")),
            channel_id=data.get("channel_id"),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expires_iso(ttl_seconds: int | None, base: datetime) -> str | None:
    if ttl_seconds is None:
        return None
    return (base + timedelta(seconds=ttl_seconds)).isoformat()


class Task:
    """Lifecycle handle for a unit of work.

    Created via ``Agent.task(...)`` or directly. Use as an async context
    manager::

        async with agent.task("research framework X") as task:
            await task.progress({"step": "search"})
            findings = await search(...)
            await task.complete(findings)

    On clean exit, auto-completes with ``result=None`` if the user did not
    call ``complete()`` or ``fail()``. On exception, auto-fails with the
    raised exception (the exception still propagates).

    If a ``ConversationContext`` is passed, events flow on that stream and
    ``ag2.task`` is stamped into ``context.dependencies`` for the duration
    of the ``async with`` block (so ``TaskInject`` resolves to the active
    task). With no context, the Task creates a private ``MemoryStream`` —
    standalone use; events still fire but only observers attached to that
    private stream see them.
    """

    def __init__(
        self,
        *,
        owner_id: str,
        spec: TaskSpec,
        context: ConversationContext | None = None,
        ttl_seconds: int | None = None,
        checkpoint_store: CheckpointStore | None = None,
        resume_from: str | None = None,
    ) -> None:
        # __init__ stores params; side effects happen in __aenter__.
        self._owner_id = owner_id
        self._spec = spec
        self._ttl_seconds = ttl_seconds
        self._context = context
        self._owns_context = False
        self._metadata: TaskMetadata | None = None
        self._had_previous_dep = False
        self._previous_dep: Any = None
        self._checkpoint_store = checkpoint_store
        self._resume_from = resume_from
        # Populated in ``__aenter__`` by reading from ``checkpoint_store``
        # when ``resume_from`` is set; ``None`` otherwise (including the
        # case where the resume target has no checkpoint on file).
        self._resumed_state: dict[str, Any] | None = None

    @property
    def task_id(self) -> str:
        if self._metadata is None:
            raise RuntimeError("Task.task_id is not available before __aenter__")
        return self._metadata.task_id

    @property
    def state(self) -> TaskState:
        if self._metadata is None:
            return TaskState.CREATED
        return self._metadata.state

    @property
    def metadata(self) -> TaskMetadata:
        if self._metadata is None:
            raise RuntimeError("Task.metadata is not available before __aenter__")
        return self._metadata

    @property
    def context(self) -> ConversationContext:
        if self._context is None:
            raise RuntimeError("Task has no bound context (use 'async with task: ...')")
        return self._context

    @property
    def resumed_state(self) -> dict[str, Any] | None:
        """The checkpoint loaded for a ``resume_from`` task, if any.

        ``None`` when no ``resume_from`` was supplied, when no
        ``checkpoint_store`` was wired, or when the target task had no
        checkpoint on file. The value is the exact dict the prior
        owner passed to :meth:`Task.checkpoint`; callers interpret it.
        """
        return self._resumed_state

    async def progress(self, payload: dict[str, Any]) -> None:
        """Emit ``TaskProgress``; merges payload into ``metadata.progress``.

        No-op if the task is already in a terminal state.
        """
        if self._metadata is None:
            raise RuntimeError("Task.progress() called before __aenter__")
        if self._metadata.state in TERMINAL_TASK_STATES:
            return
        self._metadata.progress.update(payload)
        self._metadata.last_progress_at = _now_iso()
        await self.context.send(
            TaskProgress(
                task_id=self._metadata.task_id,
                agent_name=self._owner_id,
                objective=self._spec.title,
                content="",
                payload=dict(payload),
            )
        )

    async def complete(self, result: Any = None) -> None:
        """Terminal: emit ``TaskCompleted``; state ← COMPLETED.

        No-op if already terminal.
        """
        if self._metadata is None:
            raise RuntimeError("Task.complete() called before __aenter__")
        if self._metadata.state in TERMINAL_TASK_STATES:
            return
        self._metadata.state = TaskState.COMPLETED
        self._metadata.result = result
        self._metadata.completed_at = _now_iso()
        await self.context.send(
            TaskCompleted(
                task_id=self._metadata.task_id,
                agent_name=self._owner_id,
                objective=self._spec.title,
                result=result,
                task_stream=self.context.stream.id,
            )
        )

    async def fail(self, error: str | BaseException) -> None:
        """Terminal: emit ``TaskFailed``; state ← FAILED.

        Accepts a string (wrapped in ``RuntimeError``) or any
        ``BaseException``. No-op if already terminal.
        """
        if self._metadata is None:
            raise RuntimeError("Task.fail() called before __aenter__")
        if self._metadata.state in TERMINAL_TASK_STATES:
            return
        if isinstance(error, str):
            exc: BaseException = RuntimeError(error)
        else:
            exc = error
        self._metadata.state = TaskState.FAILED
        self._metadata.error = str(exc)
        self._metadata.completed_at = _now_iso()
        await self.context.send(
            TaskFailed(
                task_id=self._metadata.task_id,
                agent_name=self._owner_id,
                objective=self._spec.title,
                error=exc,
            )
        )

    async def expire(self) -> None:
        """Terminal: emit ``TaskExpired``; state ← EXPIRED.

        Called by an external TTL observer (e.g. the network hub's TTL
        sweeper, mirrored back to the agent's stream). No-op if already
        terminal.
        """
        if self._metadata is None:
            raise RuntimeError("Task.expire() called before __aenter__")
        if self._metadata.state in TERMINAL_TASK_STATES:
            return
        self._metadata.state = TaskState.EXPIRED
        self._metadata.completed_at = _now_iso()
        await self.context.send(
            TaskExpired(
                task_id=self._metadata.task_id,
                agent_name=self._owner_id,
                objective=self._spec.title,
            )
        )

    async def cancel(self, reason: str = "") -> None:
        """Terminal: emit ``TaskCancelled``; state ← CANCELLED.

        Owner-driven cancellation — distinct from ``fail`` (the work
        could not complete) and ``expire`` (TTL elapsed). Peers may
        request cancellation via the ``ag2.task.cancel_request``
        envelope; the owner decides whether to honour by calling this
        method. ``reason`` is a free-form diagnostic surfaced on the
        emitted ``TaskCancelled`` event. No-op if already terminal.
        """
        if self._metadata is None:
            raise RuntimeError("Task.cancel() called before __aenter__")
        if self._metadata.state in TERMINAL_TASK_STATES:
            return
        self._metadata.state = TaskState.CANCELLED
        self._metadata.completed_at = _now_iso()
        if reason:
            self._metadata.error = reason
        await self.context.send(
            TaskCancelled(
                task_id=self._metadata.task_id,
                agent_name=self._owner_id,
                objective=self._spec.title,
                reason=reason,
            )
        )

    async def checkpoint(self, state: dict[str, Any]) -> None:
        """Persist owner-supplied resume state via the configured store.

        Distinct from :meth:`progress` — progress is observable
        bookkeeping (emitted as ``TaskProgress``); a checkpoint is a
        crash-recovery snapshot the owner reads back on the next run
        via ``Task(..., resume_from=task_id)``. Silent no-op when no
        ``checkpoint_store`` was supplied (standalone agents that
        haven't opted in pay no cost) or when the task is already
        terminal.
        """
        if self._metadata is None:
            raise RuntimeError("Task.checkpoint() called before __aenter__")
        if self._metadata.state in TERMINAL_TASK_STATES:
            return
        if self._checkpoint_store is None:
            return
        await self._checkpoint_store.write(self._metadata.task_id, dict(state))

    async def __aenter__(self) -> "Task":
        if self._metadata is not None:
            raise RuntimeError(f"Task already entered (state={self._metadata.state})")

        if self._context is None:
            self._context = ConversationContext(stream=MemoryStream())
            self._owns_context = True

        if self._resume_from is not None and self._checkpoint_store is not None:
            self._resumed_state = await self._checkpoint_store.read(self._resume_from)

        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        self._metadata = TaskMetadata(
            task_id=uuid4().hex,
            owner_id=self._owner_id,
            spec=self._spec,
            state=TaskState.RUNNING,
            created_at=now_iso,
            started_at=now_iso,
            expires_at=_expires_iso(self._ttl_seconds, now),
        )

        existing = self._context.dependencies.get(_TASK_DEP_KEY)
        if existing is not None:
            self._had_previous_dep = True
            self._previous_dep = existing
        self._context.dependencies[_TASK_DEP_KEY] = self

        await self._context.send(
            TaskStarted(
                task_id=self._metadata.task_id,
                agent_name=self._owner_id,
                objective=self._spec.title,
                spec=self._spec,
            )
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        try:
            if self._metadata is None:
                return
            if exc is not None and self._metadata.state not in TERMINAL_TASK_STATES:
                await self.fail(exc)
            elif self._metadata.state == TaskState.RUNNING:
                await self.complete()
        finally:
            if self._context is not None:
                if self._had_previous_dep:
                    self._context.dependencies[_TASK_DEP_KEY] = self._previous_dep
                else:
                    self._context.dependencies.pop(_TASK_DEP_KEY, None)
            self._had_previous_dep = False
            self._previous_dep = None


TaskInject = Annotated[Task | None, Inject(_TASK_DEP_KEY, default=None)]
