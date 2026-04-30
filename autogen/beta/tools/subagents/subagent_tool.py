# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, TypeAlias
from uuid import uuid4

from autogen.beta.annotations import Context
from autogen.beta.middleware.base import ToolMiddleware
from autogen.beta.stream import Stream
from autogen.beta.tools.final import FunctionTool, tool

from .run_task import run_task

if TYPE_CHECKING:
    from autogen.beta.agent import Agent

StreamFactory: TypeAlias = Callable[["Agent", Context], Stream]


def subagent_tool(
    agent: "Agent",
    *,
    description: str,
    name: str | None = None,
    stream: StreamFactory | None = None,
    middleware: Iterable[ToolMiddleware] = (),
) -> FunctionTool:
    tool_name = name or f"task_{agent.name}"
    propagate_keys: tuple[str, ...] = tuple(getattr(agent.config, "_subtask_propagate_keys", ()))

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
        # Pre-seed protocol-level conversation keys (e.g. A2A `context_id`)
        # so that parallel and serial calls to this tool share one server-side
        # conversation. setdefault is atomic in CPython, so concurrent first
        # calls converge on the value written by whichever ran first.
        for key in propagate_keys:
            ctx.variables.setdefault(key, str(uuid4()))

        task_stream = stream(agent, ctx) if stream else None

        result = await run_task(
            agent,
            objective,
            context=context,
            parent_context=ctx,
            stream=task_stream,
        )

        return result.result or ""

    return delegate
