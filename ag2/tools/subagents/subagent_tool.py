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
StreamOrFactory: TypeAlias = Stream | StreamFactory


def subagent_tool(
    agent: "Agent",
    *,
    description: str,
    name: str | None = None,
    stream: StreamOrFactory | None = None,
    middleware: Iterable[ToolMiddleware] = (),
) -> FunctionTool:
    """Expose ``agent`` as a delegation tool that runs it as a sub-task.

    ``stream=`` accepts three shapes:

    - ``None`` (default) — each delegation runs on a fresh ``MemoryStream``.
    - a ``Stream`` instance — every delegation runs on that same stream, which
      the caller can read or subscribe to.
    - a ``StreamFactory`` — called once per delegation to build the stream.

    !!! warning
        A ``Stream`` instance also makes the sub-agent stateful: each
        delegation sees every earlier delegation's turns, and token cost grows
        with them — the same behaviour as ``persistent_stream()``. For events
        without accumulated history, use a factory returning a fresh stream.
    """
    tool_name = name or f"task_{agent.name}"

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
        task_stream = stream_factory(agent, ctx) if stream_factory else None

        result = await run_task(
            agent,
            objective,
            context=context,
            parent_context=ctx,
            stream=task_stream,
        )

        if not result.completed:
            return f"Sub-task '{agent.name}' failed: {result.error}"
        return result.result or ""

    return delegate


def _resolve_stream_argument(
    stream: StreamOrFactory | None,
) -> StreamFactory | None:
    """Normalize the ``stream=`` argument into a ``StreamFactory`` or ``None``.

    - ``None``               -> ``None``; each call builds a fresh stream
    - a ``Stream`` instance  -> a factory returning that same instance
    - a ``Stream`` class     -> ``TypeError``
    - a callable             -> returned unchanged
    - anything else          -> ``TypeError``
    """
    if stream is None:
        return None
    # Reject a Stream class passed where an instance is expected. Runs before
    # the isinstance branch, which a Stream class also satisfies.
    if isinstance(stream, type) and all(hasattr(stream, attr) for attr in ("send", "enqueue", "spawn_background")):
        raise TypeError(
            "`stream=` for as_tool / subagent_tool got the Stream class "
            f"`{stream.__name__}` itself, not an instance; "
            f"did you mean `{stream.__name__}()`?"
        )
    if isinstance(stream, Stream):
        instance = stream
        return lambda _agent, _ctx: instance
    if callable(stream):
        return stream
    raise TypeError(
        "`stream=` for as_tool / subagent_tool must be a Stream instance, "
        "a StreamFactory (Callable[[Agent, Context], Stream]), or None; "
        f"got {type(stream).__name__}."
    )
