# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Iterable
from typing import TYPE_CHECKING
from uuid import uuid4

from autogen.beta.annotations import Context
from autogen.beta.middleware.base import ToolMiddleware
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.final import FunctionTool, tool

from .run_task import run_task
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
) -> FunctionTool:
    """Expose ``agent`` as a fire-and-forget background subagent tool.

    The returned tool starts the subagent and immediately returns a task id.
    Completion is reported asynchronously through ``TaskCompleted`` /
    ``TaskFailed`` events on the parent stream.
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
            run_task(
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
