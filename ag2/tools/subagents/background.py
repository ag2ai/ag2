# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import TYPE_CHECKING
from uuid import uuid4

from ag2.annotations import Context
from ag2.middleware.base import ToolMiddleware
from ag2.stream import MemoryStream, Stream
from ag2.tools.final import FunctionTool, tool

from .run_task import run_task
from .subagent_tool import StreamFactory, StreamOrFactory, _resolve_stream_argument

if TYPE_CHECKING:
    from ag2.agent import Agent


def background_agent_tool(
    agent: "Agent",
    *,
    description: str,
    name: str | None = None,
    stream: StreamOrFactory | None = None,
    middleware: Iterable[ToolMiddleware] = (),
) -> FunctionTool:
    """Expose ``agent`` as a fire-and-forget background subagent tool.

    The returned tool starts the subagent and immediately returns a task id.
    The parent ``Agent.ask`` loop keeps running and will not return until the
    background task finishes — once it does, its result is delivered to the
    parent LLM as a follow-up turn via ``context.enqueue``.

    ``stream=`` accepts the same shapes as ``subagent_tool``: ``None``, a
    ``Stream`` instance, or a ``StreamFactory``. A ``Stream`` instance is
    reused by every delegation, which makes the sub-agent stateful.
    """

    tool_name = name or f"background_task_{agent.name}"

    # Normalize `stream=` into a factory (or None) once, at construction time.
    stream_factory: StreamFactory | None = _resolve_stream_argument(stream)

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
        task_stream = stream_factory(agent, ctx) if stream_factory else MemoryStream(storage=ctx.stream.history.storage)

        ctx.spawn_background(
            _run_and_deliver(
                agent,
                objective,
                context=context,
                parent_context=ctx,
                stream=task_stream,
                task_id=task_id,
            )
        )

        return f"Background task started: {task_id}"

    return delegate


async def _run_and_deliver(
    agent: "Agent",
    objective: str,
    *,
    context: str,
    parent_context: Context,
    stream: Stream,
    task_id: str,
) -> None:
    result = await run_task(
        agent,
        objective,
        context=context,
        parent_context=parent_context,
        stream=stream,
        task_id=task_id,
    )

    if result.completed:
        message = f"Background task {task_id} ({result.objective}) completed:\n{result.result or ''}"
    else:
        message = f"Background task {task_id} ({result.objective}) failed:\n{result.error}"

    parent_context.enqueue(message)
