# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING
from uuid import uuid4

from autogen.beta.annotations import Context
from autogen.beta.middleware.base import ToolMiddleware
from autogen.beta.pending import PendingMessagePriority
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.final import FunctionTool, tool

from .run_task import TaskResult, run_task
from .subagent_tool import StreamFactory

if TYPE_CHECKING:
    from autogen.beta.agent import Agent


def background_agent_tool(
    agent: "Agent",
    *,
    description: str,
    name: str | None = None,
    stream: StreamFactory | None = None,
    middleware: Iterable[ToolMiddleware] = (),
    deliver_result: bool = False,
    deliver_errors: bool = True,
    priority: PendingMessagePriority = "asap",
    result_formatter: Callable[[TaskResult], str] | None = None,
) -> FunctionTool:
    """Expose ``agent`` as a fire-and-forget background subagent tool.

    The returned tool starts the subagent and immediately returns a task id.
    Completion is always reported asynchronously through ``TaskCompleted`` /
    ``TaskFailed`` events on the parent stream. Set ``deliver_result=True`` to
    also enqueue a model-visible follow-up message when the task completes
    while the parent ``Agent.ask`` loop is still running. Failed tasks are
    enqueued too when ``deliver_errors`` is true.
    """

    tool_name = name or f"background_task_{agent.name}"

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
        task_id = uuid4().hex
        task_stream = (
            stream(agent, ctx)
            if stream
            else MemoryStream(
                storage=ctx.stream.history.storage,
            )
        )

        asyncio.create_task(
            _run_background_task_and_enqueue(
                agent,
                objective,
                context=context,
                parent_context=ctx,
                stream=task_stream,
                task_id=task_id,
                deliver_result=deliver_result,
                deliver_errors=deliver_errors,
                priority=priority,
                result_formatter=result_formatter,
            )
        )

        return f"Background task started: {task_id}"

    return delegate


async def _run_background_task_and_enqueue(
    agent: "Agent",
    objective: str,
    *,
    context: str,
    parent_context: Context,
    stream: MemoryStream,
    task_id: str,
    deliver_result: bool,
    deliver_errors: bool,
    priority: PendingMessagePriority,
    result_formatter: Callable[[TaskResult], str] | None,
) -> None:
    result = await run_task(
        agent,
        objective,
        context=context,
        parent_context=parent_context,
        stream=stream,
        task_id=task_id,
    )

    if not deliver_result:
        return

    if result.completed or deliver_errors:
        parent_context.enqueue(_format_background_result(result, result_formatter), priority=priority)


def _format_background_result(result: TaskResult, formatter: Callable[[TaskResult], str] | None) -> str:
    if formatter is not None:
        return formatter(result)

    if result.completed:
        return f"Background task {result.task_id} completed for {result.objective}:\n{result.result or ''}"

    return f"Background task {result.task_id} failed for {result.objective}:\n{result.error}"
