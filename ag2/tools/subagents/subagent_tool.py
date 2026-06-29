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
    tool_name = name or f"task_{agent.name}"

    # Resolve `stream=` once at construction time so that misuse (passing a
    # non-Stream / non-callable) fails loudly here instead of silently
    # producing an empty per-call MemoryStream every time the tool runs.
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

        return result.result or ""

    return delegate


def _resolve_stream_argument(
    stream: StreamOrFactory | None,
) -> StreamFactory | None:
    """Normalize the ``stream=`` argument into a ``StreamFactory`` or ``None``.

    ``as_tool(stream=...)`` historically only accepted a ``StreamFactory``
    (``Callable[[Agent, Context], Stream]``); passing a bare ``Stream``
    instance silently fell through, because the call site
    ``stream(agent, ctx)`` either misinterpreted the instance as a factory
    (and crashed) or, with the old guard, just returned ``None`` — so the
    sub-agent ran with a fresh ephemeral stream and the caller's stream
    stayed empty. See #2888.

    The new contract:

    - ``None``                  -> ``None`` (use a fresh per-call stream)
    - a ``Stream`` instance     -> wrapped in a factory returning that
                                   exact instance, so callers can hold a
                                   handle to capture sub-agent events
    - a callable                -> treated as a ``StreamFactory`` as before
    - anything else             -> ``TypeError`` raised eagerly
    """
    if stream is None:
        return None
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
