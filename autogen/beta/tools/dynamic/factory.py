# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from typing import Any

from autogen.beta.annotations import Context
from autogen.beta.config.config import ModelConfig
from autogen.beta.middleware.base import ToolMiddleware
from autogen.beta.tools.final import FunctionTool, Toolkit, tool
from autogen.beta.tools.tool import Tool

from .handler import resolve_and_run
from .spec_input import DynamicAgentSpec

_SINGLE_DESCRIPTION = (
    "Create an ephemeral dynamic agent from a spec (name, system prompt, "
    "subset of available tools) and immediately run it on the given objective. "
    "Returns the agent's final reply as a string. The spawned agent has a "
    "fresh conversation history and cannot recursively spawn other dynamic "
    "agents. Call this when a focused sub-task benefits from a tailored "
    "system prompt and a narrow tool set."
)


def _normalize_pool(
    available_tools: Iterable[Tool | Callable[..., Any]],
) -> list[Tool]:
    """Normalize and dedupe the tool pool once at factory time.

    Conversion via :meth:`FunctionTool.ensure_tool` is done here so the
    runtime path doesn't redo it on every tool call.
    """
    pool: list[Tool] = []
    seen: set[str] = set()
    for t in available_tools:
        normalized = FunctionTool.ensure_tool(t)
        if normalized.name in seen:
            continue
        seen.add(normalized.name)
        pool.append(normalized)
    return pool


def dynamic_agent(
    *,
    available_tools: Iterable[Tool | Callable[..., Any]],
    config: ModelConfig,
    name: str = "dynamic_agent",
    middleware: Iterable[ToolMiddleware] = (),
) -> Toolkit:
    """Tool factory that lets a parent Agent dynamically build & run sub-agents.

    Drop the returned :class:`Toolkit` into ``Agent(tools=[...])`` and the
    parent LLM gains one tool: ``create_and_run_agent(spec, objective)``.

    The parent LLM constructs each spec at runtime (name, system prompt,
    a subset of tool names from ``available_tools``) and the framework
    instantiates an ephemeral :class:`Agent`, runs the objective via
    :func:`run_task` (fresh stream, shallow-copied deps, copied vars),
    and returns the reply string.

    Parameters:
        available_tools: Pool of tools the dynamic agent may pick from
            by name. Normalized and deduped once at factory time.
        config: Model configuration used to run every dynamic agent
            spawned through this factory.
        name: Name of the returned :class:`Toolkit` (the LLM-visible tool
            name is fixed: ``create_and_run_agent``).
        middleware: Tool middleware applied to the tool.
    """
    pool = _normalize_pool(available_tools)

    @tool(
        name="create_and_run_agent",
        description=_SINGLE_DESCRIPTION,
    )
    async def create_and_run_agent(
        spec: DynamicAgentSpec,
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

    return Toolkit(
        create_and_run_agent,
        name=name,
        middleware=middleware,
    )
