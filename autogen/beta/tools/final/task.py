# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias
from uuid import uuid4

from autogen.beta.annotations import Context
from autogen.beta.events import TaskCompleted, TaskFailed, TaskStarted
from autogen.beta.middleware.base import ToolMiddleware
from autogen.beta.stream import MemoryStream, Stream

from .function_tool import FunctionTool, tool

if TYPE_CHECKING:
    from autogen.beta.agent import Agent

StreamFactory: TypeAlias = Callable[["Agent", Context], Stream]


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
            variables=parent_context.variables,
        )

        # Sync variable mutations back to the parent context
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


def subagent_tool(
    agent: "Agent",
    *,
    description: str,
    name: str | None = None,
    stream: StreamFactory | None = None,
    middleware: Iterable[ToolMiddleware] = (),
) -> FunctionTool:
    tool_name = name or f"task_{agent.name}"

    @tool(
        name=tool_name,
        description=description,
        middleware=middleware,
    )
    async def delegate(
        ctx: Context,
        objective: str,
        context: str = "",
    ) -> str:
        task_id = str(uuid4())
        task_stream = stream(agent, ctx) if stream else None

        await ctx.send(
            TaskStarted(
                task_id=task_id,
                agent_name=agent.name,
                objective=objective,
            )
        )

        result = await run_task(
            agent,
            objective,
            context=context,
            parent_context=ctx,
            stream=task_stream,
        )

        if result.completed:
            await ctx.send(
                TaskCompleted(
                    task_id=task_id,
                    agent_name=agent.name,
                    objective=objective,
                    result=result.result or "",
                    task_stream=result.stream,
                )
            )

        else:
            await ctx.send(
                TaskFailed(
                    task_id=task_id,
                    agent_name=agent.name,
                    objective=objective,
                    error=result.result or "",
                )
            )

        return result.result or ""

    return delegate


def persistent_stream() -> StreamFactory:
    def stream_factory(agent: "Agent", ctx: "Context") -> MemoryStream:
        key = f"ag:{agent.name}:stream"
        if not (stream_id := ctx.dependencies.get(key)):
            stream_id = ctx.dependencies[key] = uuid4()

        return MemoryStream(
            storage=ctx.stream.history.storage,
            id=stream_id,
        )

    return stream_factory
