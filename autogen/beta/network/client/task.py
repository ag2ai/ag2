# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Task — client-side handle for a live network task.

Thin wrapper around the owning :class:`Session` that stamps ``task_id`` on
every envelope it emits and tracks the current :class:`TaskMetadata`. The
hub is the source of truth for state transitions; :meth:`Task.refresh` pulls
the latest metadata from the hub cache so callers see transitions the sweeper
or the peer just applied.

Task handles are produced in two places:

* :meth:`Session.create_task` — the requester's handle. Usually the requester
  only *waits* on the task; emitting phase / progress / result events on the
  requester side is disallowed by the hub's sender-authority check.
* The default notify handler in ``autogen/beta/network/client/handlers.py``
  when an ``ag2.task.assigned`` envelope arrives — the owner's handle.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ..envelope import (
    EV_TASK_ERROR,
    EV_TASK_PHASE_COMPLETED,
    EV_TASK_PHASE_ENTERED,
    EV_TASK_PROGRESS,
    EV_TASK_RESULT,
    TASK_TERMINAL_EVENT_TYPES,
    Envelope,
)
from ..errors import (
    TaskCancelledError,
    TaskExpiredError,
    TaskFailedError,
)
from ..errors import TimeoutError as NetTimeoutError
from ..task import TaskMetadata, TaskState

if TYPE_CHECKING:
    from .session import Session


class Task:
    """Live handle for a hub-tracked network task.

    Every mutation method routes through the owning session's
    :meth:`ActorClient._send_envelope` path, which means:

    * The envelope picks up the sender's pre/post-send transform pipeline.
    * The hub enforces the task state machine on every event.
    * The session WAL is the durable audit trail for the entire task.

    Methods either return the updated :class:`TaskMetadata` (for terminal
    transitions) or ``None`` (for intermediate transitions like ``progress``).
    """

    def __init__(self, *, session: Session, metadata: TaskMetadata) -> None:
        self._session = session
        self._metadata = metadata

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def task_id(self) -> str:
        return self._metadata.task_id

    @property
    def session(self) -> Session:
        return self._session

    @property
    def metadata(self) -> TaskMetadata:
        return self._metadata

    @property
    def state(self) -> TaskState:
        return self._metadata.state

    def is_terminal(self) -> bool:
        return self._metadata.is_terminal()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> TaskMetadata:
        """Re-read the task from the hub's in-memory cache.

        The hub caches every task in ``_tasks`` and rewrites
        ``hub/tasks/{task_id}/metadata.json`` on every transition, so
        a peek through :meth:`Hub.peek_task` is authoritative as long
        as the hub is in-process. Cross-process deployments will route
        this through the HTTP ``GET /v1/tasks/{id}`` endpoint Phase 3b
        exposes; the public signature does not change.
        """

        latest = self._session._client._hub.peek_task(self.task_id)
        if latest is not None:
            self._metadata = latest
        return self._metadata

    # ------------------------------------------------------------------
    # Owner-side transitions
    # ------------------------------------------------------------------

    async def phase_entered(self, phase_id: str, *, description: str = "") -> None:
        """Mark ``phase_id`` as entered. Advances the task to ``running``."""

        envelope = self._build_envelope(
            event_type=EV_TASK_PHASE_ENTERED,
            event_data={"phase_id": phase_id, "description": description},
        )
        await self._post(envelope)

    async def phase_completed(self, phase_id: str) -> None:
        """Mark ``phase_id`` as completed. Does not advance overall state."""

        envelope = self._build_envelope(
            event_type=EV_TASK_PHASE_COMPLETED,
            event_data={"phase_id": phase_id},
        )
        await self._post(envelope)

    async def progress(self, **fields: Any) -> None:
        """Merge ``fields`` into the task's progress dict.

        The first progress event moves the task from ``created`` to
        ``running`` on the hub side if the owner skipped phase events.
        """

        envelope = self._build_envelope(
            event_type=EV_TASK_PROGRESS,
            event_data={"update": dict(fields)},
        )
        await self._post(envelope)

    async def result(self, value: Any) -> TaskMetadata:
        """Terminal. Record ``value`` and transition state to ``completed``."""

        envelope = self._build_envelope(
            event_type=EV_TASK_RESULT,
            event_data={"value": value},
        )
        await self._post(envelope)
        return self.refresh()

    async def fail(self, error: str) -> TaskMetadata:
        """Terminal. Record ``error`` and transition state to ``failed``."""

        envelope = self._build_envelope(
            event_type=EV_TASK_ERROR,
            event_data={"error": error},
        )
        await self._post(envelope)
        return self.refresh()

    # ------------------------------------------------------------------
    # Requester-side control
    # ------------------------------------------------------------------

    async def cancel(self, *, reason: str = "") -> TaskMetadata:
        """Cancel the task through the hub API (direct call, not an envelope).

        Cancellation is driven by the hub's direct :meth:`Hub.cancel_task`
        method — symmetric with :meth:`Hub.close_session` — because
        cancellation may originate from either the requester or the
        owner and needs hub-authoritative state transition even if the
        session is mid-send. The hub emits the broadcast
        ``ag2.task.cancelled`` envelope.
        """

        client = self._session._client
        updated = await client._hub.cancel_task(
            self.task_id, requested_by=client.actor_id, reason=reason
        )
        self._metadata = updated
        return updated

    # ------------------------------------------------------------------
    # Waiting for terminal state
    # ------------------------------------------------------------------

    async def wait(
        self, *, timeout: float | None = None
    ) -> TaskMetadata:
        """Block until the task reaches a terminal state.

        Opens a session subscription starting at the task's current
        WAL cursor and resolves on the first envelope whose ``task_id``
        matches ``self.task_id`` and whose event type is terminal
        (``ag2.task.result`` / ``error`` / ``cancelled`` / ``expired``).

        If the task is already terminal when ``wait`` is called the
        metadata is returned synchronously without opening a
        subscription.

        Raises:
            :class:`TaskFailedError`     — task ended in ``failed`` state.
            :class:`TaskCancelledError`  — task ended in ``cancelled`` state.
            :class:`TaskExpiredError`    — task ended in ``expired`` state.
            :class:`NetTimeoutError`     — ``timeout`` elapsed before terminal.
        """

        latest = self.refresh()
        if latest.is_terminal():
            return self._resolve_terminal(latest)

        client = self._session._client
        queue = await client._open_subscription(
            session_id=self._session.session_id, since=0
        )
        try:
            while True:
                try:
                    envelope = await asyncio.wait_for(queue.get(), timeout=timeout)
                except asyncio.TimeoutError as exc:
                    raise NetTimeoutError(
                        f"task {self.task_id} wait timed out after {timeout}s"
                    ) from exc
                if envelope.task_id != self.task_id:
                    continue
                if envelope.event_type not in TASK_TERMINAL_EVENT_TYPES:
                    continue
                # The hub has already applied the transition by the
                # time the subscriber sees the envelope, so a peek
                # returns the authoritative terminal metadata.
                terminal = client._hub.peek_task(self.task_id)
                if terminal is None:
                    terminal = latest  # defensive fallback
                self._metadata = terminal
                return self._resolve_terminal(terminal)
        finally:
            await client._close_subscription(queue)

    def _resolve_terminal(self, metadata: TaskMetadata) -> TaskMetadata:
        if metadata.state is TaskState.COMPLETED:
            return metadata
        if metadata.state is TaskState.FAILED:
            raise TaskFailedError(metadata.error or "task failed", metadata)
        if metadata.state is TaskState.CANCELLED:
            raise TaskCancelledError(metadata.error or "", metadata)
        if metadata.state is TaskState.EXPIRED:
            raise TaskExpiredError(metadata)
        return metadata

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_envelope(
        self, *, event_type: str, event_data: dict[str, Any]
    ) -> Envelope:
        client = self._session._client
        # Task events broadcast across the session by default — the hub
        # fans them out to every non-sender participant, which is what
        # lets a non-owner requester see them without subscribing.
        return Envelope(
            session_id=self._session.session_id,
            sender_id=client.actor_id,
            recipient_id=None,
            event_type=event_type,
            event_data=dict(event_data),
            task_id=self.task_id,
        )

    async def _post(self, envelope: Envelope) -> None:
        await self._session._client._send_envelope(envelope)
        self.refresh()
