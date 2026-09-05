# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from dataclasses import dataclass, replace
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
from ag2.exceptions import HumanInputError
from ag2.stream import MemoryStream, Stream
from ag2.usage import UsageReport, collect_usage_events

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


async def _emit_rollup(parent_context: Context, agent_name: str, incurred: list[UsageEvent]) -> None:
    """Emit one ``"subtask"`` rollup for what this invocation spent.

    The sub-agent's per-call ``UsageEvent`` events stay on its private stream;
    the parent gets this single rollup instead, so ``UsageReport`` — which
    aggregates additively — sees each delegation once. Nothing spent, no rollup.
    Shared by the success and the failure path so the two can't drift apart.

    One rollup, always — that is an asserted invariant, not an accident, and
    emitting one per pair instead would change ``UsageReport``'s by-model and
    by-provider breakdown for every existing caller. It does carry the
    ``(provider, model)`` pair when the delegated run used exactly one, so
    per-model attribution survives a single-configuration delegation; the two
    fields are declared ``compare=False`` on the event, so this disturbs no
    existing equality assertion.
    """
    usage = _rollup_usage(incurred)
    if not usage:
        return

    provider, model = _sole_pair(incurred)
    await parent_context.send(UsageEvent(usage, kind="subtask", label=agent_name, provider=provider, model=model))


def _rollup_usage(incurred: Iterable[UsageEvent]) -> Usage:
    """One ``Usage`` covering everything the delegation spent.

    Counts are summed. ``total_tokens`` survives only when every call reported
    one, since a partial sum falls below the counts it is meant to cover.
    """
    spent = [event.usage for event in incurred if event.usage]
    usage = sum(spent, Usage())
    if any(one.total_tokens is None for one in spent):
        usage = replace(usage, total_tokens=None)
    return usage


def _sole_pair(incurred: Iterable[UsageEvent]) -> tuple[str | None, str | None]:
    """The one ``(provider, model)`` behind this spend, or absence when several.

    A mixed-model delegation has no honest label, and guessing one — the
    parent's config, or the first call's — would show an attribution that is not
    true. Absence says "this spend is real but unattributable", which a client
    can render; a wrong label it cannot detect. Records that spent nothing are
    ignored: they contribute no tokens, so they must not decide whose the
    tokens are. A sub-agent that itself delegates carries an unlabelled rollup
    of its own, which counts as a pair and correctly makes the parent's rollup
    unlabelled too.
    """
    pairs = {(event.provider, event.model) for event in incurred if event.usage}
    if len(pairs) != 1:
        return None, None
    return pairs.pop()


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
    incurred: list[UsageEvent] = []
    usage_sub_id = task_stream.where(UsageEvent).subscribe(collect_usage_events(incurred))

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
            usage = sum((event.usage for event in incurred), Usage())

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

        # A sub-task killed by the human-input channel is the same failure one
        # level down: reported as a TaskResult it becomes the delegating tool's
        # output, and the parent's model is told the delegation failed and is
        # free to try another route to the same effect. The accounting and the
        # TaskFailed event above still happen — only the swallowing does not.
        if isinstance(e, HumanInputError):
            raise

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
