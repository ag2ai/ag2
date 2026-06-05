# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, TypeAlias

from ag2.annotations import Context
from ag2.middleware.base import ToolMiddleware
from ag2.stream import Stream
from ag2.tools.final import FunctionTool, tool

from .run_task import run_task

if TYPE_CHECKING:
    from ag2.agent import Agent

StreamFactory: TypeAlias = Callable[["Agent", Context], Stream]


def _as_stream_factory(stream: Stream | StreamFactory | None) -> StreamFactory | None:
    """Normalize a stream instance or factory to a factory."""
    if stream is None or callable(stream):
        return stream

    if not isinstance(stream, Stream):
        raise TypeError(
            f"stream must be a Stream instance, a StreamFactory callable, or None. Got {type(stream).__name__}."
        )

    captured_stream = stream
    return lambda _agent, _ctx: captured_stream


def subagent_tool(
    agent: "Agent",
    *,
    description: str,
    name: str | None = None,
    stream: Stream | StreamFactory | None = None,
    middleware: Iterable[ToolMiddleware] = (),
) -> FunctionTool:
    """Wrap an agent as a callable :class:`FunctionTool`.

    Args:
        agent: The agent to wrap.
        description: Tool description shown to the calling LLM.
        name: Optional custom tool name. Defaults to ``task_{agent.name}``.
        stream: A :class:`Stream` instance, a :class:`StreamFactory` callable,
            or ``None``. If a Stream instance is passed, it is automatically
            wrapped in a factory that returns that instance on every call.
        middleware: Optional tool middleware.

    Returns:
        A :class:`FunctionTool` that runs the agent as a sub-task.

    Raises:
        TypeError: If ``stream`` is not a Stream, StreamFactory, or None.
    """
    stream_factory = _as_stream_factory(stream)

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
        task_stream = stream_factory(agent, ctx) if stream_factory else None

        result = await run_task(
            agent,
            objective,
            context=context,
            parent_context=ctx,
            stream=task_stream,
        )

        return result.result or ""

    return delegate
