"""Secret redaction middleware -- scrubs sensitive patterns before LLM sees them."""

import copy
import logging
import re
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from .base import BaseMiddleware, CallNext

logger = logging.getLogger(__name__)

# Maximum content length to redact. Content exceeding this threshold is
# not redacted (skipped with a warning) to prevent ReDoS on unbounded input.
_MAX_REDACT_BYTES = 100_000  # 100 KB


class SecretRedactionMiddleware(BaseMiddleware):
    """
    Redacts secret patterns from text events before they reach the LLM.

    This middleware should be placed outermost (first in the middleware list)
    so that raw secrets never appear in downstream logs or processing.

    PoC limitation -- shallow only: only the top-level content field of
    ModelRequest events is scanned, plus tool call arguments in on_tool_call.
    Nested structures and binary payloads are not processed. Production
    use requires recursive traversal of all event fields.

    New event objects are created on redaction rather than mutating originals.

    Parameters
    ----------
    patterns:
        List of compiled regex patterns. Each match is replaced with
        the replacement string.
    replacement:
        Replacement text. Default: "[REDACTED]".
    max_content_bytes:
        Maximum content length (in characters) to apply redaction to.
        Content exceeding this limit is passed through unchanged with a
        warning logged. Default: 100_000 (100 KB). Guards against ReDoS
        via unbounded input to catastrophic backtracking patterns.
    """

    def __init__(
        self,
        patterns: list[re.Pattern[str]],
        replacement: str = "[REDACTED]",
        max_content_bytes: int = _MAX_REDACT_BYTES,
    ) -> None:
        self._patterns = list(patterns)
        self._replacement = replacement
        if not isinstance(max_content_bytes, int) or isinstance(max_content_bytes, bool):
            raise TypeError(
                f"max_content_bytes must be int, got {type(max_content_bytes).__name__}"
            )
        if max_content_bytes < 0:
            raise ValueError(
                f"max_content_bytes must be non-negative, got {max_content_bytes}"
            )
        self._max_content_bytes = max_content_bytes

    def _redact_text(self, text: str) -> tuple[str, int]:
        """Apply all patterns to text. Return (redacted_text, replacement_count).

        Returns the original text unchanged if it exceeds max_content_bytes,
        logging a warning to alert operators.
        """
        if len(text) > self._max_content_bytes:
            logger.warning(
                "[Redaction] Content length %d exceeds max_content_bytes=%d"
                " -- skipping redaction to prevent ReDoS",
                len(text),
                self._max_content_bytes,
            )
            return text, 0

        count = 0
        for pattern in self._patterns:
            text, n = pattern.subn(self._replacement, text)
            count += n
        return text, count

    def _copy_event_with_content(self, event: Any, new_content: str) -> Any:
        """Return a copy of event with content replaced by new_content.

        Tries multiple strategies to preserve other fields:
        1. copy.copy() + object.__setattr__ (works for dataclass/EventMeta).
        2. type(event)(content=...) -- PoC fallback, may lose other fields.

        Logs a warning if only the fallback was possible.
        """
        try:
            new_event = copy.copy(event)
            object.__setattr__(new_event, "content", new_content)
            return new_event
        except (TypeError, AttributeError):
            pass

        # Fallback: reconstruct with only content preserved.
        logger.warning(
            "[Redaction] copy.copy() failed for %s -- reconstructing with "
            "content only (other fields may be lost)",
            type(event).__name__,
        )
        try:
            return type(event)(content=new_content)
        except Exception as exc:
            logger.error(
                "[Redaction] Could not reconstruct %s: %r -- blocking event "
                "(fail-closed: will not forward unredacted content)",
                type(event).__name__,
                exc,
            )
            # Fail-closed: return a sanitized event with only the redacted
            # content rather than forwarding the unredacted original.
            # Use a minimal object that downstream can handle.
            raise RuntimeError(
                "[Redaction] Cannot safely redact event -- blocking to prevent secret leak"
            ) from exc

    async def on_llm_call(
        self,
        call_next: CallNext,
        *messages: Any,
        ctx: Any,
        tools: Iterable[Any],
    ) -> None:
        """
        Redact secrets from ModelRequest events before forwarding.

        Shallow only: only ModelRequest.content (str) is scanned.
        """
        from autogen.beta.events import ModelRequest

        redacted_messages: list[Any] = []
        total_redacted = 0

        for event in messages:
            if isinstance(event, ModelRequest) and isinstance(event.content, str):
                new_content, count = self._redact_text(event.content)
                if count:
                    event = self._copy_event_with_content(event, new_content)
                    total_redacted += count
            redacted_messages.append(event)

        if total_redacted:
            logger.info(
                "[Redaction] Redacted %d secret(s) in events before LLM call",
                total_redacted,
            )

        await call_next(*redacted_messages, ctx=ctx, tools=tools)

    async def on_tool_call(
        self,
        call_next: Callable[[Any, Any], Awaitable[None]],
        event: Any,
        ctx: Any,
    ) -> None:
        """
        Redact secrets from tool call arguments before forwarding.

        Scans the arguments field (str) if present. Creates a copy of the
        event with redacted arguments when matches are found.
        """
        arguments = getattr(event, "arguments", None)
        if isinstance(arguments, str):
            new_args, count = self._redact_text(arguments)
            if count:
                try:
                    new_event = copy.copy(event)
                    object.__setattr__(new_event, "arguments", new_args)
                    event = new_event
                except (TypeError, AttributeError):
                    # Fail-closed: do not forward unredacted tool arguments.
                    # Emit ToolError instead of raising RuntimeError to avoid
                    # crashing the middleware chain.
                    logger.error(
                        "[Redaction] Could not copy tool event %s to redact arguments"
                        " -- emitting ToolError (fail-closed)",
                        type(event).__name__,
                    )
                    from autogen.beta.events import ToolError

                    await ctx.send(
                        ToolError(
                            parent_id=getattr(event, "id", None),
                            name="<redaction-blocked>",
                            content="Tool call blocked: cannot safely redact arguments",
                            error=None,
                        )
                    )
                    return  # do not call call_next
                logger.info(
                    "[Redaction] Redacted %d secret(s) in tool arguments",
                    count,
                )

        await call_next(event, ctx)
