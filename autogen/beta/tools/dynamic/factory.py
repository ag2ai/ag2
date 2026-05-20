# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from typing import Any

from autogen.beta.annotations import Context
from autogen.beta.config.config import ModelConfig
from autogen.beta.middleware.base import ToolMiddleware
from autogen.beta.spec import AgentSpec
from autogen.beta.tools.final import FunctionTool, tool
from autogen.beta.tools.tool import Tool

from .handler import resolve_and_run

_SINGLE_DESCRIPTION = (
    "Create an ephemeral dynamic agent from a spec (name, system prompt, "
    "subset of available tools, optional response schema) and immediately "
    "run it on the given objective. Returns the agent's final reply as a "
    "string. The spawned agent has a fresh conversation history and cannot "
    "recursively spawn other dynamic agents. Call this when a focused "
    "sub-task benefits from a tailored system prompt and a narrow tool set."
)


def dynamic_agent(
    *,
    available_tools: Iterable[Tool | Callable[..., Any]],
    config: ModelConfig,
    middleware: Iterable[ToolMiddleware] = (),
) -> FunctionTool:
    """Tool factory that lets a parent Agent dynamically build & run sub-agents.

    Drop the returned :class:`FunctionTool` into ``Agent(tools=[...])`` and the
    parent LLM gains one tool: ``create_and_run_agent(spec, objective)``.

    The parent LLM constructs each spec at runtime (name, system prompt,
    a subset of tool names from ``available_tools``, optional response
    schema) and the framework instantiates an ephemeral :class:`Agent`,
    runs the objective via :func:`run_task` (fresh stream, shallow-copied
    deps, copied vars), and returns the reply string.

    Parameters:
        available_tools: Pool of tools the dynamic agent may pick from
            by name.
        config: Model configuration used to run every dynamic agent
            spawned through this factory.
        middleware: Tool middleware applied to the tool.
    """
    pool = list(available_tools)

    @tool(
        name="create_and_run_agent",
        description=_SINGLE_DESCRIPTION,
        middleware=middleware,
    )
    async def create_and_run_agent(
        spec: AgentSpec,
        objective: str,
        ctx: Context,
    ) -> str:
        return await resolve_and_run(
            spec,
            objective,
            ctx,
            pool=pool,
            config=config,
        )

    return create_and_run_agent
