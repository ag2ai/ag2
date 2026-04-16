# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Task verbs — ``run_task`` and ``track_task``.

``run_task`` is the merged form (replaces the design's pre-Phase-6
split between ``run_task`` and ``start_task``) — set ``blocking=True``
(default) to await the terminal state, or ``blocking=False`` to get
the ``task_id`` immediately and poll later via ``track_task``.

Both verbs route through :meth:`Session.create_task` /
:meth:`Hub.peek_task`, so every access check, rule enforcement, and
hub-side state-machine guard applies the same way it would to a
direct API caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from autogen.beta.annotations import Context
from autogen.beta.tools.final.function_tool import tool

from ...errors import (
    AccessDeniedError,
    LimitExceededError,
    NetworkError,
    TaskCancelledError,
    TaskExpiredError,
    TaskFailedError,
)
from ...errors import TimeoutError as NetTimeoutError
from ...task import TaskMetadata, TaskSpec
from .session import _resolve_session

if TYPE_CHECKING:
    from ..actor_client import ActorClient


def _summarise_task(metadata: TaskMetadata) -> dict[str, Any]:
    """LLM-friendly view of a :class:`TaskMetadata` snapshot."""

    state = metadata.state
    state_value = state.value if hasattr(state, "value") else str(state)

    return {
        "task_id": metadata.task_id,
        "session_id": metadata.session_id,
        "owner_id": metadata.owner_id,
        "requester_id": metadata.requester_id,
        "state": state_value,
        "current_phase": metadata.current_phase,
        "progress": dict(metadata.progress or {}),
        "result": metadata.result,
        "error": metadata.error,
        "created_at": metadata.created_at,
        "started_at": metadata.started_at,
        "completed_at": metadata.completed_at,
        "expires_at": metadata.expires_at,
    }


def build_task_verbs(client: ActorClient) -> list[Any]:
    """Return the ``run_task`` and ``track_task`` verb tools."""

    @tool
    async def run_task(
        ctx: Context,
        title: str,
        description: str,
        spec_type: str = "",
        owner: str | None = None,
        blocking: bool = True,
        session_id: str | None = None,
        timeout: float | None = None,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Create a network task inside a session and (optionally) wait for it.

        Args:
            title: Short human-readable title for the task.
            description: Full prompt the task owner's handler will run.
            spec_type: Optional handler routing key. The owner's
                ``ActorClient.on_task(spec_type)`` handler picks tasks
                up by this string; an empty string routes to the
                default ``"*"`` handler.
            owner: The participant name or id that should execute the
                task. Defaults to the single non-initiator participant
                in two-party sessions; required in multi-party sessions.
            blocking: When ``True`` (default) the verb awaits the
                task's terminal state and returns the result dict.
                When ``False`` the verb returns immediately with the
                task id and the LLM can poll later via
                :func:`track_task`.
            session_id: Optional explicit session id. Defaults to the
                current notify-handler session.
            timeout: Optional wait timeout (seconds) for the blocking
                path. ``None`` = wait until terminal or until the
                hub's TTL sweeper expires the task.
            ttl_seconds: Optional task TTL (seconds). ``None`` =
                inherit from the owner's
                ``rule.limits.task_ttl_default``.

        Returns one of:

        * ``{task_id, state, ...}`` for the non-blocking path
        * ``{task_id, state="completed", result, ...}`` on success
        * ``{task_id, state, error, ...}`` on failure / cancel /
          expiry / timeout

        Errors do not raise — the LLM reads the dict and decides
        whether to retry, give up, or hand off.
        """

        try:
            session = _resolve_session(ctx, client, session_id)
        except (ValueError, Exception) as exc:
            return {"error": str(exc)}

        spec = TaskSpec(
            title=title,
            description=description,
            spec_type=spec_type or "",
        )

        try:
            task = await session.create_task(
                spec,
                owner=owner,
                blocking=False,
                ttl_seconds=ttl_seconds,
            )
        except ValueError as exc:
            return {"error": str(exc)}
        except (AccessDeniedError, LimitExceededError, NetworkError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        if not blocking:
            # ``session.create_task(blocking=False)`` returns a Task handle
            # whose ``metadata`` is the freshly-created snapshot.
            return _summarise_task(task.metadata)

        try:
            terminal = await task.wait(timeout=timeout)
        except TaskFailedError as exc:
            summary = _summarise_task(exc.metadata)
            summary["error"] = exc.metadata.error or "task failed"
            return summary
        except TaskCancelledError as exc:
            summary = _summarise_task(exc.metadata)
            summary["error"] = exc.metadata.error or "task cancelled"
            return summary
        except TaskExpiredError as exc:
            summary = _summarise_task(exc.metadata)
            summary["error"] = "task expired"
            return summary
        except NetTimeoutError as exc:
            return {
                "task_id": task.task_id,
                "state": task.state.value
                if hasattr(task.state, "value")
                else str(task.state),
                "error": f"timeout: {exc}",
            }

        return _summarise_task(terminal)

    @tool
    async def track_task(task_id: str) -> dict[str, Any]:
        """Look up a task's current state by id.

        Args:
            task_id: The hub-stamped task id returned by an earlier
                :func:`run_task` call.

        Returns the task summary dict, or ``{error}`` if the hub has
        never heard of this id (already-archived terminal tasks may
        return ``unknown``).
        """

        metadata = client._hub.peek_task(task_id)
        if metadata is None:
            return {"error": f"unknown task {task_id!r}"}
        return _summarise_task(metadata)

    return [run_task, track_task]


__all__ = ("build_task_verbs",)
