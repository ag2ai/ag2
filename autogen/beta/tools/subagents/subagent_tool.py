# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, TypeAlias

from autogen.beta.annotations import Context
from autogen.beta.config.seeders import SubtaskContextSeeder
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
    seeder = agent.config if isinstance(agent.config, SubtaskContextSeeder) else None

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
        # Let the agent's config (e.g. A2AConfig) pre-seed protocol-level
        # conversation state so that parallel and serial sub-task calls share
        # one server-side conversation. ``setdefault`` is atomic in CPython, so
        # concurrent first-time calls converge on the same value.
        if seeder is not None:
            seeder.seed_subtask_variables(ctx.variables)

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
