"""Shared helpers for governance middleware."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def send_tool_error(ctx: Any, event: Any, error: Exception) -> None:
    """Emit a ToolError to the context stream.

    Centralizes ToolError construction to avoid duplication across
    budget and policy middleware.

    The content field uses a generic message to avoid leaking internal
    exception details (class names, stack state, tool names) to the LLM
    or external consumers. The full error repr is logged at WARNING level
    for operator diagnostics only.
    """
    from autogen.beta.events import ToolError

    logger.warning(
        "[Governance] Tool call denied: %r",
        error,
    )

    # Use generic name to avoid leaking internal tool names to downstream
    # consumers (attacker could enumerate protected tool interfaces).
    # The real tool name is available in the WARNING log above.
    await ctx.send(
        ToolError(
            parent_id=getattr(event, "id", None),
            name="<denied>",
            content="Tool call denied by governance policy",
            error=error,
        )
    )
