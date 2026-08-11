# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from typing import Any

from ag2.annotations import Context
from ag2.config.config import ModelConfig
from ag2.middleware.base import ToolMiddleware
from ag2.spec import AgentSpec
from ag2.tools.final import FunctionTool, tool
from ag2.tools.tool import Tool

from .handler import resolve_and_run

_BASE_DESCRIPTION = (
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

    The pool's names + descriptions are appended to the returned tool's
    ``description`` so the parent LLM learns them from the tool schema and
    does not need them spelled out in its system prompt.

    Parameters:
        available_tools: Pool of tools the dynamic agent may pick from
            by name.
        config: Model configuration used to run every dynamic agent
            spawned through this factory.
        middleware: Tool middleware applied to the tool.
    """
    pool = list(available_tools)

    description = _BASE_DESCRIPTION
    if pool:
        menu_lines: list[str] = []
        for raw in pool:
            t = FunctionTool.ensure_tool(raw)
            if isinstance(t, FunctionTool):
                desc = (t.schema.function.description or "").strip()
                menu_lines.append(f"- {t.name}: {desc}" if desc else f"- {t.name}")
            else:
                menu_lines.append(f"- {t.name}")
        menu = "\n".join(menu_lines)
        description = f"{_BASE_DESCRIPTION}\n\nAvailable tools:\n{menu}"

    @tool(
        name="create_and_run_agent",
        description=description,
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
