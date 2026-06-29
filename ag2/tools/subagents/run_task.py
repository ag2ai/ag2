# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from ag2.annotations import Context
from ag2.events import (
    HumanInputRequest,
    TaskCompleted,
    TaskFailed,
    TaskStarted,
    Usage,
    UsageEvent,
)
from ag2.stream import MemoryStream, Stream

if TYPE_CHECKING:
    from ag2.agent import Agent


@dataclass
class TaskResult:
    task_id: str
    objective: str
    result: str | None
    completed: bool
    stream: "Stream"
    usage: Usage
    error: Exception | None = None


def _make_hitl_bridge(parent_context: Context):
    """Forward ``HumanInputRequest`` events from the child stream to the parent.

    Defined at module level so it isn't re-created per ``run_task`` call (per
    AGENTS.md: no nested functions in runtime execution paths). The closure
    over ``parent_context`` is captured here, at definition time of the
    bridge, not inside any hot loop.
    """

    async def _bridge_hitl(event: HumanInputRequest, ctx: Context) -> None:
        await parent_context.stream.send(event, ctx)

    return _bridge_hitl


async def run_task(
    agent: "Agent",
    objective: str,
    *,
    parent_context: Context,
    context: str = "",
    stream: "Stream | None" = None,
    emit_events: bool = True,
    task_id: str | None = None,
) -> TaskResult:
    """Run ``agent`` as a sub-task and return its ``TaskResult``.

    ``emit_events`` controls whether ``TaskStarted`` / ``TaskCompleted`` /
    ``TaskFailed`` events are emitted onto ``parent_context.stream``.
    Keep it at the default (``True``) unless the caller is itself going to
    emit its own task lifecycle events.

    ``task_id`` lets callers pre-assign the lifecycle id, which is useful for
    background tools that must return the id before the task completes.
    """
    task_id = task_id or uuid4().hex
    task_stream = stream or MemoryStream(
        storage=parent_context.stream.history.storage,
    )
    prompt = objective
    if context:
        prompt = f"{objective}\n\n## Context\n{context}"

    if emit_events:
        await parent_context.send(TaskStarted(task_id=task_id, agent_name=agent.name, objective=objective))

    # Bridge HITL events to the parent stream so the parent's hook can handle
    # them. If the subagent has its own HITL hook, it is registered as an
    # interrupter and swallows the event first.
    sub_id: str | None = None
    if not agent._hitl_hook:
        sub_id = task_stream.where(HumanInputRequest).subscribe(
            _make_hitl_bridge(parent_context),
            interrupt=True,
        )

    try:
        reply = await agent.ask(
            prompt,
            stream=task_stream,
            dependencies=parent_context.dependencies.copy(),
            # Copy variables so concurrent sibling tasks don't interfere.
            # Mutations made by the child are intentionally not synced back —
            # with concurrent siblings via asyncio.gather, last-writer-wins
            # would silently clobber values, so we keep child mutations
            # scoped to the child run by design.
            variables=parent_context.variables.copy(),
        )

        usage = (await reply.usage()).total

        result = TaskResult(
            task_id=task_id,
            objective=objective,
            result=reply.body,
            completed=True,
            stream=task_stream,
            usage=usage,
        )

        if emit_events:
            # The sub-agent's per-call UsageEvents live on its private stream;
            # emit a single rollup onto the parent so the parent's usage report
            # accounts for the sub-task without seeing its individual calls.
            if usage:
                await parent_context.send(
                    UsageEvent(usage, kind="subtask", label=agent.name),
                )
            await parent_context.send(
                TaskCompleted(
                    task_id=task_id,
                    agent_name=agent.name,
                    objective=objective,
                    result=reply.body,
                    task_stream=task_stream.id,
                    usage=usage,
                )
            )

        return result

    except Exception as e:
        if emit_events:
            await parent_context.send(
                TaskFailed(
                    task_id=task_id,
                    agent_name=agent.name,
                    objective=objective,
                    error=e,
                )
            )
        return TaskResult(
            task_id=task_id,
            objective=objective,
            result=None,
            completed=False,
            stream=task_stream,
            error=e,
            usage=Usage(),
        )

    finally:
        if sub_id:
            task_stream.unsubscribe(sub_id)
