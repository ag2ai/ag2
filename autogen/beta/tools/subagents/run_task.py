# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import TYPE_CHECKING

from autogen.beta.annotations import Context
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


async def run_task(
    agent: "Agent",
    objective: str,
    *,
    parent_context: Context,
    context: str = "",
    stream: "Stream | None" = None,
) -> TaskResult:
    task_stream = stream or MemoryStream()
    prompt = objective
    if context:
        prompt = f"{objective}\n\n## Context\n{context}"

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
        )

    except Exception as e:
        return TaskResult(
            objective=objective,
            result=str(e),
            completed=False,
            stream=task_stream,
        )
