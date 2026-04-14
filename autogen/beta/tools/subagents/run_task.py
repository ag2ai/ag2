# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from autogen.beta.annotations import Context
from autogen.beta.events import (
    HumanInputRequest,
    TaskCompleted,
    TaskFailed,
    TaskStarted,
    Usage,
)
from autogen.beta.stream import MemoryStream, Stream

if TYPE_CHECKING:
    from autogen.beta.actor import Actor

_DEPTH_KEY = "ag:task_depth"


@dataclass
class TaskResult:
    task_id: str
    objective: str
    result: str | None
    completed: bool
    stream: "Stream"
    usage: Usage
    error: Exception | None = None


def _reply_usage(reply) -> Usage:
    """Pull the typed Usage from an AgentReply, defaulting to an empty Usage."""
    if reply and reply.response and reply.response.usage:
        return reply.response.usage
    return Usage()


async def run_task(
    agent: "Actor",
    objective: str,
    *,
    parent_context: Context,
    context: str = "",
    stream: "Stream | None" = None,
    emit_events: bool = True,
) -> TaskResult:
    """Run ``agent`` as a sub-task and return its ``TaskResult``.

    ``emit_events`` controls whether ``TaskStarted`` / ``TaskCompleted`` /
    ``TaskFailed`` events are emitted onto ``parent_context.stream``.
    Keep it at the default (``True``) unless the caller is itself going to
    emit its own task lifecycle events.
    """
    task_id = uuid4().hex
    task_stream = stream or MemoryStream(
        storage=parent_context.stream.history.storage,
    )
    prompt = objective
    if context:
        prompt = f"{objective}\n\n## Context\n{context}"

    if emit_events:
        await parent_context.send(
            TaskStarted(task_id=task_id, agent_name=agent.name, objective=objective)
        )

    # Bridge HITL events to the parent stream so the parent's hook
    # can handle them. If the subagent has its own HITL hook, it is
    # registered as an interrupter and swallows the event first.
    if not agent._hitl_hook:

        async def _bridge_hitl(event: HumanInputRequest, ctx: Context) -> None:
            await parent_context.stream.send(event, ctx)

        sub_id = task_stream.where(HumanInputRequest).subscribe(_bridge_hitl, interrupt=True)
    else:
        sub_id = None

    try:
        reply = await agent.ask(
            prompt,
            stream=task_stream,
            dependencies=parent_context.dependencies.copy(),
            # Copy variables so concurrent sibling tasks don't interfere,
            # and increment the task depth counter for the child.
            variables={
                **parent_context.variables,
                _DEPTH_KEY: parent_context.variables.get(_DEPTH_KEY, 0) + 1,
            },
        )

        # Sync variable mutations back to the parent context,
        # excluding the depth counter (internal bookkeeping).
        reply.context.variables.pop(_DEPTH_KEY, None)
        parent_context.variables.update(reply.context.variables)

        usage = _reply_usage(reply)

        result = TaskResult(
            task_id=task_id,
            objective=objective,
            result=reply.body,
            completed=True,
            stream=task_stream,
            usage=usage,
        )

        if emit_events:
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
