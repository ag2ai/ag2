# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.annotations import Context
from autogen.beta.events import ToolCallEvent, ToolResultEvent
from autogen.beta.middleware.base import ToolExecution, ToolMiddleware, ToolResultType

from .run_task import _DEPTH_KEY

DEFAULT_MAX_DEPTH = 3


def depth_limiter(
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> ToolMiddleware:
    """Tool middleware that limits recursive subagent tool calls.

    Depth is tracked via ``context.variables`` and incremented automatically
    by :func:`run_task` for each nesting level.  This middleware only
    **reads** the current depth — it never mutates shared state, which
    makes it safe for concurrent tool execution (``asyncio.gather``).

    When the current depth reaches *max_depth*, the tool returns an error
    message instead of executing, preventing infinite recursion.

    Args:
        max_depth: Maximum allowed nesting depth. ``3`` (default) means the
            top-level agent can delegate up to three levels deep.

    Returns:
        A tool middleware hook that can be passed to the ``middleware``
        parameter of :func:`subagent_tool` or
        :meth:`~autogen.beta.agent.Agent.as_tool`.
    """

    async def hook(
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        if context.variables.get(_DEPTH_KEY, 0) >= max_depth:
            return ToolResultEvent.from_call(
                event,
                result=f"Error: maximum task depth ({max_depth}) reached. Cannot delegate further.",
            )

        return await call_next(event, context)

    return hook
