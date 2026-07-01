# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``TaskMirror`` — bridges agent ``Task*`` stream events to ``Hub.observe_task``.

The network is *one observer* of agent-owned tasks: tasks are created
and driven by the agent itself, and the mirror simply subscribes to
``TaskStarted`` / ``TaskProgress`` / ``TaskCompleted`` / ``TaskFailed``
/ ``TaskExpired`` on the agent's stream and forwards the corresponding
``TaskMetadata`` updates to the hub.
"""

import contextlib
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ag2.events import (
    TaskCancelled,
    TaskCompleted,
    TaskExpired,
    TaskFailed,
    TaskProgress,
    TaskStarted,
)
from ag2.task import TERMINAL_TASK_STATES, TaskMetadata, TaskSpec, TaskState

from .errors import NotFoundError

if TYPE_CHECKING:
    from ag2.context import Stream

    from .client.hub_client import HubClient
    from .hub import Hub

__all__ = ("TaskMirror",)


logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskMirror:
    """Forwards an Agent's ``Task*`` events to the hub.

    Construct one per ``AgentClient`` (the owner is the Agent's
    ``agent_id``). Attach to a stream for the duration of a notify
    handler / Agent.ask call, then detach.

    The mirror routes through a :class:`HubClient` so task-event
    forwarding goes through the same surface as the rest of the client.
    Tests that hold a bare ``Hub`` can still pass it directly via the
    ``hub=`` keyword for convenience.

    Failures forwarding to the hub are swallowed — the mirror must
    never crash the agent's turn.
    """

    def __init__(
        self,
        *,
        hub_client: "HubClient | None" = None,
        hub: "Hub | None" = None,
        owner_id: str,
        channel_id: str | None = None,
    ) -> None:
        # __init__ stores params; subscription happens in attach().
        if hub_client is None and hub is None:
            raise TypeError("TaskMirror requires either hub_client= or hub= (legacy)")
        self._hub_client = hub_client
        self._hub = hub if hub is not None else (hub_client._hub if hub_client is not None else None)
        self._owner_id = owner_id
        self._channel_id = channel_id

    async def _observe(self, metadata: TaskMetadata) -> None:
        if self._hub_client is not None:
            await self._hub_client.observe_task(metadata)
        elif self._hub is not None:
            await self._hub.observe_task(metadata)

    async def _update(
        self,
        task_id: str,
        *,
        state: TaskState | None = None,
        progress: dict[str, object] | None = None,
        result: object | None = None,
        error: str | None = None,
    ) -> None:
        if self._hub_client is not None:
            await self._hub_client.update_task(
                task_id,
                state=state,
                progress=progress,
                result=result,
                error=error,
            )
        elif self._hub is not None:
            await self._hub.update_task(
                task_id,
                state=state,
                progress=progress,
                result=result,
                error=error,
            )

    async def _record(
        self,
        *,
        owner_id: str,
        capability: str,
        outcome: TaskState,
        latency_ms: int | None,
        task_id: str,
    ) -> None:
        if self._hub_client is not None:
            await self._hub_client.record_observation(
                owner_id=owner_id,
                capability=capability,
                outcome=outcome,
                latency_ms=latency_ms,
                task_id=task_id,
            )
        elif self._hub is not None:
            await self._hub.record_observation(
                owner_id=owner_id,
                capability=capability,
                outcome=outcome,
                latency_ms=latency_ms,
                task_id=task_id,
            )

    def attach(self, stream: "Stream") -> list[object]:
        """Subscribe to ``Task*`` events; returns sub ids for ``detach``."""
        return [
            stream.where(TaskStarted).subscribe(self._on_started, sync_to_thread=False),
            stream.where(TaskProgress).subscribe(self._on_progress, sync_to_thread=False),
            stream.where(TaskCompleted).subscribe(self._on_completed, sync_to_thread=False),
            stream.where(TaskFailed).subscribe(self._on_failed, sync_to_thread=False),
            stream.where(TaskExpired).subscribe(self._on_expired, sync_to_thread=False),
            stream.where(TaskCancelled).subscribe(self._on_cancelled, sync_to_thread=False),
        ]

    def detach(self, stream: "Stream", sub_ids: list[object]) -> None:
        """Unsubscribe the previously-attached subscriptions."""
        for sid in sub_ids:
            with contextlib.suppress(Exception):
                stream.unsubscribe(sid)  # type: ignore[arg-type]

    async def _escalate(self, task_id: str, op: str, exc: BaseException) -> None:
        """Report a mirror-side failure without crashing the agent turn.

        Logs at ``ERROR`` and fires ``on_task_event(task_id,
        "mirror_failed", payload)`` through the hub's listeners so the
        failure surfaces to operators / dashboards. Routing goes
        through :meth:`HubClient.fire_task_event` (preferred) or
        :meth:`Hub.fire_task_event` (direct construction in tests) —
        both are public surfaces.
        """
        logger.error(
            "task mirror failed: op=%s task_id=%s exc=%r",
            op,
            task_id,
            exc,
        )
        payload = {
            "op": op,
            "owner_id": self._owner_id,
            "channel_id": self._channel_id,
            "exc_type": type(exc).__name__,
            "exc_message": str(exc),
        }
        with contextlib.suppress(Exception):
            if self._hub_client is not None:
                await self._hub_client.fire_task_event(task_id, "mirror_failed", payload)
            elif self._hub is not None:
                await self._hub.fire_task_event(task_id, "mirror_failed", payload)

    async def _on_started(self, event: TaskStarted) -> None:
        spec = event.spec if event.spec is not None else TaskSpec(title=event.objective or "")
        now = _now_iso()
        metadata = TaskMetadata(
            task_id=event.task_id,
            owner_id=self._owner_id,
            spec=spec,
            state=TaskState.RUNNING,
            created_at=now,
            started_at=now,
            channel_id=self._channel_id,
        )
        try:
            await self._observe(metadata)
        except Exception as exc:
            await self._escalate(event.task_id, "started", exc)

    async def _on_progress(self, event: TaskProgress) -> None:
        try:
            await self._update(
                event.task_id,
                progress=dict(event.payload) if event.payload else None,
            )
        except NotFoundError:
            pass
        except Exception as exc:
            await self._escalate(event.task_id, "progress", exc)

    async def _on_completed(self, event: TaskCompleted) -> None:
        try:
            await self._update(
                event.task_id,
                state=TaskState.COMPLETED,
                result=event.result,
            )
        except NotFoundError:
            pass
        except Exception as exc:
            await self._escalate(event.task_id, "completed", exc)
        await self._record_observation_if_tagged(event.task_id, TaskState.COMPLETED)

    async def _on_failed(self, event: TaskFailed) -> None:
        try:
            await self._update(
                event.task_id,
                state=TaskState.FAILED,
                error=str(event.error),
            )
        except NotFoundError:
            pass
        except Exception as exc:
            await self._escalate(event.task_id, "failed", exc)
        await self._record_observation_if_tagged(event.task_id, TaskState.FAILED)

    async def _on_expired(self, event: TaskExpired) -> None:
        try:
            await self._update(event.task_id, state=TaskState.EXPIRED)
        except NotFoundError:
            pass
        except Exception as exc:
            await self._escalate(event.task_id, "expired", exc)
        await self._record_observation_if_tagged(event.task_id, TaskState.EXPIRED)

    async def _on_cancelled(self, event: TaskCancelled) -> None:
        try:
            await self._update(
                event.task_id,
                state=TaskState.CANCELLED,
                error=event.reason or None,
            )
        except NotFoundError:
            pass
        except Exception as exc:
            await self._escalate(event.task_id, "cancelled", exc)
        await self._record_observation_if_tagged(event.task_id, TaskState.CANCELLED)

    async def _record_observation_if_tagged(self, task_id: str, outcome: TaskState) -> None:
        """If the task's spec carried a ``capability`` tag, push the
        observation through so the owner's ``Resume.observed`` updates.

        Reads the task's ``capability`` + ``started_at`` from the
        in-process hub cache when one is present (using the hub clock
        for latency); cross-process it fetches the observed
        ``TaskMetadata`` over the wire via ``HubClient.get_task`` and
        uses the local clock. Either way the observation is pushed
        through ``_record`` (hub-client-routed, so RPC cross-process).
        """
        if outcome not in TERMINAL_TASK_STATES:
            return
        task_meta: TaskMetadata | None = None
        now_iso: str | None = None
        if self._hub is not None:
            task_meta = self._hub._tasks.get(task_id)
            now_iso = self._hub._clock()
        elif self._hub_client is not None:
            try:
                task_meta = await self._hub_client.get_task(task_id)
            except Exception:
                task_meta = None
            now_iso = datetime.now(timezone.utc).isoformat()
        if task_meta is None or task_meta.spec.capability is None:
            return
        latency_ms: int | None = None
        if task_meta.started_at and now_iso is not None:
            try:
                started = datetime.fromisoformat(task_meta.started_at).timestamp()
                now = datetime.fromisoformat(now_iso).timestamp()
                latency_ms = int(max(0.0, now - started) * 1000)
            except Exception:
                latency_ms = None
        with contextlib.suppress(Exception):
            await self._record(
                owner_id=task_meta.owner_id,
                capability=task_meta.spec.capability,
                outcome=outcome,
                latency_ms=latency_ms,
                task_id=task_id,
            )
