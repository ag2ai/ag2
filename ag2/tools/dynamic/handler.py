# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

from ag2.annotations import Context
from ag2.config.config import ModelConfig
from ag2.exceptions import ToolResolutionError
from ag2.spec import AgentSpec
from ag2.tools.subagents.run_task import run_task
from ag2.tools.tool import Tool


async def resolve_and_run(
    spec: AgentSpec,
    objective: str,
    ctx: Context,
    *,
    pool: list[Tool | Callable[..., Any]],
    config: ModelConfig,
) -> str:
    """Build a dynamic agent from ``spec`` and execute ``objective`` via ``run_task``.

    Returns the dynamic agent's reply body. On a failure that the parent
    LLM should be able to recover from (unknown tool names, child-agent
    exception) returns a human-readable ``Error: ...`` string instead of
    raising — so the parent can retry with a corrected spec.
    """
    try:
        agent = spec.to_agent(
            available_tools=pool,
            config=config,
        )
    except ToolResolutionError as e:
        return f"Error: unknown tools {sorted(e.missing)}. Available: {sorted(e.available)}"

    result = await run_task(
        agent,
        objective,
        parent_context=ctx,
    )

    if not result.completed:
        return f"Error: {result.error}"
    return result.result or ""
