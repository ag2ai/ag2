# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``tasks(action)`` — task lifecycle for the LLM.

Active task actions (resolved via ``TaskInject`` to the agent's
currently-running ``agent.task(...)`` block):

* ``progress`` — emit a ``TaskProgress`` payload on the active task.
* ``complete`` — terminal: emit ``TaskCompleted``. Usually unnecessary
  (the ``async with`` exits auto-completes).

Discovery / observation:

* ``list``     — tasks the agent owns or is waiting on.
* ``status``   — refresh ``TaskMetadata`` for a task by id.
* ``wait``     — block until a peer's task reaches a terminal state.
* ``cancel``   — post an ``ag2.task.cancel_request`` envelope to the
  task owner. The owner is free to honour by calling ``Task.cancel``
  or to ignore the request.

``start`` is intentionally **not** a tool — calling it from the LLM
would bypass the ``async with`` lifecycle that scopes
``TaskInject`` correctly. Owners start tasks via ``agent.task(...)``
in their own code; the LLM uses ``progress`` and ``complete`` once a
task is active, and ``delegate`` for one-shot remote work.
"""

import asyncio
from typing import TYPE_CHECKING, Any, Literal

from ag2.tools import tool

from ...envelope import EV_TASK_CANCEL_REQUEST, Envelope
from ..inject import AgentClientInject, TaskInject

if TYPE_CHECKING:
    from ..agent_client import AgentClient


__all__ = ("make_tasks_tool",)


def _task_summary(meta: Any) -> dict[str, Any]:
    return {
        "task_id": meta.task_id,
        "owner_id": meta.owner_id,
        "title": meta.spec.title,
        "capability": meta.spec.capability,
        "state": meta.state.value,
        "created_at": meta.created_at,
        "started_at": meta.started_at,
        "completed_at": meta.completed_at,
        "expires_at": meta.expires_at,
        "last_progress_at": meta.last_progress_at,
        "result": meta.result,
        "error": meta.error,
    }


def make_tasks_tool(agent_client: "AgentClient") -> object:
    """Return a closure-bound ``tasks`` tool."""

    @tool
    async def tasks(
        action: Literal["progress", "complete", "list", "status", "wait", "cancel"],
        *,
        payload: dict | None = None,
        result: Any | None = None,
        task_id: str | None = None,
        reason: str = "",
        scope: Literal["own", "all"] = "own",
        state: Literal["active", "all"] = "active",
        timeout: float = 300.0,
        poll_interval: float = 0.1,
        limit: int = 20,
        client: AgentClientInject = None,
        active_task: TaskInject = None,
    ) -> dict | list[dict] | str:
        """Task lifecycle.

        Active-task actions (require an open ``agent.task(...)`` block):
            ``progress``  args payload (dict)
            ``complete``  args result?

        Observation actions:
            ``list``    args scope="own"|"all", state="active"|"all", limit
            ``status``  args task_id
            ``wait``    args task_id, timeout=300, poll_interval=0.1
            ``cancel``  args task_id, reason?  — posts an
                        ``ag2.task.cancel_request`` envelope to the
                        owner; the owner decides whether to honour it.
        """
        actual = client if client is not None else agent_client
        hub = actual._hub_client

        if action == "progress":
            if active_task is None:
                return "Error: progress requires an active `agent.task(...)` block"
            try:
                await active_task.progress(payload or {})
            except Exception as exc:
                return f"Error: progress failed: {exc}"
            return f"progress emitted on {active_task.task_id}"

        if action == "complete":
            if active_task is None:
                return "Error: complete requires an active `agent.task(...)` block"
            try:
                await active_task.complete(result=result)
            except Exception as exc:
                return f"Error: complete failed: {exc}"
            return f"completed {active_task.task_id}"

        if action == "list":
            owner_filter = actual.agent_id if scope == "own" else None
            metas = await hub.list_tasks(agent_id=owner_filter, limit=limit * 4)
            terminal = {"completed", "failed", "expired", "cancelled"}
            results: list[dict] = []
            for meta in metas:
                if state == "active" and meta.state.value in terminal:
                    continue
                results.append(_task_summary(meta))
                if len(results) >= limit:
                    break
            return results

        if action == "status":
            if not task_id:
                return "Error: status requires `task_id`"
            try:
                meta = await hub.get_task(task_id)
            except Exception:
                return f"Error: task {task_id!r} not found"
            return _task_summary(meta)

        if action == "wait":
            if not task_id:
                return "Error: wait requires `task_id`"
            deadline = asyncio.get_event_loop().time() + timeout
            terminal = {"completed", "failed", "expired", "cancelled"}
            while asyncio.get_event_loop().time() < deadline:
                try:
                    meta = await hub.get_task(task_id)
                except Exception:
                    return f"Error: task {task_id!r} not found"
                if meta.state.value in terminal:
                    return _task_summary(meta)
                await asyncio.sleep(poll_interval)
            return f"Error: task {task_id!r} did not complete within {timeout}s"

        if action == "cancel":
            if not task_id:
                return "Error: cancel requires `task_id`"
            try:
                meta = await hub.get_task(task_id)
            except Exception:
                return f"Error: task {task_id!r} not found"
            if meta.state.value in {"completed", "failed", "expired", "cancelled"}:
                return f"task {task_id!r} is already {meta.state.value}"
            if not meta.channel_id:
                return (
                    f"Error: task {task_id!r} has no associated channel; "
                    "cancel_request envelopes need a channel to ride on"
                )
            envelope = Envelope(
                channel_id=meta.channel_id,
                sender_id=actual.agent_id,
                audience=[meta.owner_id],
                event_type=EV_TASK_CANCEL_REQUEST,
                event_data={"task_id": task_id, "reason": reason},
                task_id=task_id,
            )
            try:
                await hub.post_envelope(envelope)
            except Exception as exc:
                return f"Error: cancel_request post failed: {exc}"
            return f"cancel_request posted to {meta.owner_id} for task {task_id}"

        return f"Error: unknown action {action!r}; choose from progress, complete, list, status, wait, cancel"

    return tasks
