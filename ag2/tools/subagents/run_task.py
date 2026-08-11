# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable
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
from ag2.usage import UsageReport

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
    """Cumulative token usage of the sub-task's stream, like ``AgentReply.usage``.

    On a stream reused across delegations (``persistent_stream()``) this is the
    worker's running total, not the cost of this call — the ``"subtask"`` rollup
    emitted onto the parent carries that.
    """
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


def _make_usage_accumulator(incurred: list[Usage]) -> Callable[[UsageEvent], Awaitable[None]]:
    """Collect what this invocation spends on the sub-task's stream.

    History can't answer "this invocation" once the stream is reused — it would
    re-report everything spent before — so the events are collected as they are
    sent. ``incurred`` must therefore be a fresh list per call.
    """

    async def _accumulate(event: UsageEvent) -> None:
        incurred.append(event.usage)

    return _accumulate


async def _emit_rollup(parent_context: Context, agent_name: str, incurred: list[Usage]) -> None:
    """Emit one ``"subtask"`` rollup for what this invocation spent.

    The sub-agent's per-call ``UsageEvent`` events stay on its private stream;
    the parent gets this single rollup instead, so ``UsageReport`` — which
    aggregates additively — sees each delegation once. Nothing spent, no rollup.
    Shared by the success and the failure path so the two can't drift apart.
    """
    usage = sum(incurred, Usage())
    if not usage:
        return

    await parent_context.send(UsageEvent(usage, kind="subtask", label=agent_name))


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

    # Scope the rollup to this invocation. Registered here and removed in the
    # same ``finally`` as the bridge above, so it cannot outlive the call.
    # Sequential calls that rebuild the stream object (``persistent_stream()``)
    # are therefore accounted per call; concurrent delegations handed the *same*
    # ``Stream`` instance still cross-capture, since the events they emit are
    # indistinguishable on the one stream they share.
    incurred: list[Usage] = []
    usage_sub_id = task_stream.where(UsageEvent).subscribe(_make_usage_accumulator(incurred))

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
            await _emit_rollup(parent_context, agent.name, incurred)
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
        # The sub-task may already have made billable model calls before it
        # failed. Those UsageEvents are on its own stream, so read them the same
        # way the success path does — via ``AgentReply.usage`` — instead of
        # reporting a failed delegation as free.
        try:
            usage = UsageReport.from_events(await task_stream.history.get_events()).total
        except Exception:
            # A storage backend that is itself the reason the sub-task died would
            # raise here too, masking the failure being handled and taking the
            # TaskFailed event with it. Surfacing the sub-task's own error matters
            # more than the cumulative reading, so degrade to this invocation's
            # spend — already in hand, and equal to the cumulative value on
            # anything but a reused stream.
            usage = sum(incurred, Usage())

        if emit_events:
            await _emit_rollup(parent_context, agent.name, incurred)
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
            usage=usage,
        )

    finally:
        task_stream.unsubscribe(usage_sub_id)
        if sub_id:
            task_stream.unsubscribe(sub_id)
