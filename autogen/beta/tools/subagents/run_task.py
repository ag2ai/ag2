# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import TYPE_CHECKING

from autogen.beta.annotations import Context
from autogen.beta.events import HumanInputRequest, Usage
from autogen.beta.stream import MemoryStream, Stream

if TYPE_CHECKING:
    from autogen.beta.agent import Agent

_DEPTH_KEY = "ag:task_depth"


@dataclass
class TaskResult:
    objective: str
    result: str | None
    completed: bool
    stream: "Stream"
    usage: Usage
    error: Exception | None = None


async def run_task(
    agent: "Agent",
    objective: str,
    *,
    parent_context: Context,
    context: str = "",
    stream: "Stream | None" = None,
) -> TaskResult:
    task_stream = stream or MemoryStream(
        storage=parent_context.stream.history.storage,
    )
    prompt = objective
    if context:
        prompt = f"{objective}\n\n## Context\n{context}"

    # Bridge HITL events to the parent stream so the parent's hook
    # can handle them. If the subagent has its own HITL hook, it is
    # registered as an interrupter and swallows the event first.
    if not agent._hitl_hook:

        async def _bridge_hitl(event: HumanInputRequest, context: Context) -> None:
            await parent_context.stream.send(event, context)

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

        return TaskResult(
            objective=objective,
            result=reply.body,
            completed=True,
            stream=task_stream,
            usage=reply.response.usage,
        )

    except Exception as e:
        return TaskResult(
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
