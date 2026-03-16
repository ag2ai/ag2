"""Policy deny middleware -- blocks tool calls matching a deny list or predicates."""

import json
import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from .base import BaseMiddleware, CallNext
from ._helpers import send_tool_error

logger = logging.getLogger(__name__)


class PolicyViolationError(Exception):
    """Raised when a tool call is denied by governance policy."""


class PolicyDenyMiddleware(BaseMiddleware):
    """
    Denies tool calls that appear in a deny list or match predicate functions.

    LLM calls are passed through unchanged. Only on_tool_call() is active.

    Parameters
    ----------
    denied_tools:
        Set of tool names that are unconditionally denied. A copy is taken
        at construction time to prevent external mutation of the deny list.
    denied_predicates:
        List of callables with signature (tool_name: str, args: dict) -> bool.
        If any predicate returns True, the tool call is denied. Predicate
        exceptions are treated as DENY (fail-closed) with a warning logged.
    """

    def __init__(
        self,
        denied_tools: set[str] | None = None,
        denied_predicates: list[Callable[[str, dict[str, Any]], bool]] | None = None,
    ) -> None:
        # Copy to prevent external mutation of the deny list after construction.
        self._denied_tools: set[str] = set(denied_tools) if denied_tools else set()
        self._denied_predicates: list[Callable[[str, dict[str, Any]], bool]] = (
            list(denied_predicates) if denied_predicates else []
        )

    async def on_llm_call(
        self,
        call_next: CallNext,
        *messages: Any,
        ctx: Any,
        tools: Iterable[Any],
    ) -> None:
        """Pass-through -- policy only applies to tool calls."""
        await call_next(*messages, ctx=ctx, tools=tools)

    async def on_tool_call(
        self,
        call_next: Callable[[Any, Any], Awaitable[None]],
        event: Any,
        ctx: Any,
    ) -> None:
        """
        Evaluate deny list and predicates. Emit ToolError on denial without
        invoking call_next.
        """
        tool_name = getattr(event, "name", None)
        if not isinstance(tool_name, str) or not tool_name:
            logger.warning(
                "[Policy] Tool event has non-str name %r -- treating as DENY",
                tool_name,
            )
            err = PolicyViolationError(
                "Tool event has invalid name type -- denied by governance policy"
            )
            await send_tool_error(ctx, event, err)
            return

        denied = tool_name in self._denied_tools

        if not denied and self._denied_predicates:
            args: dict[str, Any] = {}  # Default: empty args (safe fallback)
            raw_args = getattr(event, "arguments", None)
            # Guard against DoS via oversized or deeply nested JSON.
            # 64 KB for args (JSON parse cost); redaction uses 100 KB for
            # content (regex cost). Different thresholds are intentional.
            _MAX_ARGS_BYTES = 65_536  # 64 KB
            if raw_args is not None and not isinstance(raw_args, str):
                # Non-string arguments (bytes, bytearray, etc.) cannot be
                # safely parsed as JSON -- fail-closed.
                logger.warning(
                    "[Policy] Tool '%s' arguments are %s, not str"
                    " -- treating as DENY (fail-closed)",
                    tool_name,
                    type(raw_args).__name__,
                )
                denied = True
            elif isinstance(raw_args, str) and len(raw_args) > _MAX_ARGS_BYTES:
                logger.warning(
                    "[Policy] Tool '%s' arguments exceed size limit (%d bytes)"
                    " -- treating as DENY (fail-closed)",
                    tool_name,
                    len(raw_args),
                )
                denied = True
            else:
                try:
                    parsed = json.loads(raw_args) if raw_args is not None else {}
                    # Non-dict JSON (array, string, null, etc.) is treated as
                    # suspicious -- fail-closed to avoid predicate bypass.
                    if not isinstance(parsed, dict):
                        logger.warning(
                            "[Policy] Tool '%s' arguments parsed to non-dict type %s"
                            " -- treating as DENY (fail-closed)",
                            tool_name,
                            type(parsed).__name__,
                        )
                        denied = True
                    else:
                        args = parsed
                except (json.JSONDecodeError, TypeError, ValueError, RecursionError, MemoryError):
                    # Fail-closed: unparseable arguments -> DENY.
                    # Using empty args would let predicates see {} and return
                    # ALLOW, silently bypassing argument-based policies.
                    logger.warning(
                        "[Policy] Failed to parse arguments for tool '%s'"
                        " -- treating as DENY (fail-closed)",
                        tool_name,
                    )
                    denied = True

            if not denied:
                for predicate in self._denied_predicates:
                    try:
                        if predicate(tool_name, args):
                            denied = True
                            break
                    except Exception as exc:
                        logger.warning(
                            "[Policy] Predicate %s raised for tool '%s': %r"
                            " -- treating as DENY",
                            predicate,
                            tool_name,
                            exc,
                        )
                        denied = True
                        break

        if denied:
            # Log the specific tool name internally for operator diagnostics.
            logger.warning("[Policy] Tool '%s' DENIED by policy", tool_name)
            # Generic message -- do not include tool name in user-visible error.
            err = PolicyViolationError(
                "Tool call denied by governance policy"
            )
            await send_tool_error(ctx, event, err)
            return

        await call_next(event, ctx)
