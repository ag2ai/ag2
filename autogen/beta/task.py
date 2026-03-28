# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import contextvars
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .annotations import Context
from .events.task_events import TaskCompleted, TaskFailed, TaskStarted
from .middleware import BaseMiddleware
from .stream import MemoryStream
from .tools.final import FunctionTool, tool
from .tools.schemas import ToolSchema
from .tools.tool import Tool

if TYPE_CHECKING:
    from .agent import Agent
    from .context import Stream

_task_depth: contextvars.ContextVar[int] = contextvars.ContextVar("task_depth", default=0)

DEFAULT_MAX_TASK_DEPTH = 1


@dataclass
class TaskResult:
    objective: str
    result: str | None
    completed: bool
    stream: "Stream"


async def _run_task(
    agent: "Agent",
    objective: str,
    *,
    context: str = "",
    dependencies: dict[Any, Any] | None = None,
    stream: "Stream | None" = None,
) -> TaskResult:
    task_stream = stream or MemoryStream()
    prompt = objective
    if context:
        prompt = f"{objective}\n\n## Context\n{context}"

    try:
        reply = await agent.ask(prompt, stream=task_stream, dependencies=dependencies)
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


class _TaskTool(Tool):
    """Wraps a FunctionTool with depth-aware schema visibility."""

    def __init__(self, inner: FunctionTool, max_depth: int) -> None:
        self._inner = inner
        self._max_depth = max_depth

    async def schemas(self, context: "Context") -> Iterable[ToolSchema]:
        if _task_depth.get() >= self._max_depth:
            return []
        return await self._inner.schemas(context)

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        self._inner.register(stack, context, middleware=middleware)


def _make_task_tool(
    agent: "Agent",
    *,
    description: str,
    name: str | None = None,
    max_depth: int = DEFAULT_MAX_TASK_DEPTH,
    stream: "Callable[[], Stream] | None" = None,
) -> _TaskTool:
    tool_name = name or f"task_{agent.name}"
    agent_ref = agent
    stream_factory = stream

    @tool(name=tool_name, description=description)
    async def delegate(objective: str, context: str = "", ctx: Context = ...) -> str:  # type: ignore[assignment]
        depth = _task_depth.get()
        if depth >= max_depth:
            return f"Error: maximum task depth ({max_depth}) reached. Cannot delegate further."

        depth_token = _task_depth.set(depth + 1)
        try:
            parent_deps = dict(ctx.dependencies)
            task_id = str(uuid4())
            task_stream = stream_factory() if stream_factory else None

            await ctx.send(TaskStarted(task_id=task_id, agent_name=agent_ref.name, objective=objective))

            result = await _run_task(
                agent_ref, objective, context=context, dependencies=parent_deps, stream=task_stream
            )

            if result.completed:
                await ctx.send(
                    TaskCompleted(
                        task_id=task_id,
                        agent_name=agent_ref.name,
                        objective=objective,
                        result=result.result or "",
                        task_stream=result.stream,
                    )
                )
            else:
                await ctx.send(
                    TaskFailed(
                        task_id=task_id,
                        agent_name=agent_ref.name,
                        objective=objective,
                        error=result.result or "",
                    )
                )

            return result.result or ""
        finally:
            _task_depth.reset(depth_token)

    return _TaskTool(delegate, max_depth)
