# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from ag2.annotations import Context
from ag2.events import BaseEvent, ModelRequest, ModelResponse, ToolResultsEvent, estimated_tokens
from ag2.middleware.base import BaseMiddleware, LLMCall, MiddlewareFactory


class TokenLimiter(MiddlewareFactory):
    """Truncate message history to fit within a token budget.

    Sizes each event with the shared content estimate (text by
    ``chars_per_token``, non-text by a per-modality budget) — never the
    truncated ``str(event)`` repr.
    """

    def __init__(self, max_tokens: int, chars_per_token: int = 4) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be greater than 0")
        if chars_per_token < 1:
            raise ValueError("chars_per_token must be greater than 0")
        self._max_tokens = max_tokens
        self._chars_per_token = chars_per_token

    def __call__(self, event: "BaseEvent", context: "Context") -> "BaseMiddleware":
        return _TokenLimiter(event, context, self._max_tokens, self._chars_per_token)


class _TokenLimiter(BaseMiddleware):
    def __init__(
        self,
        event: "BaseEvent",
        context: "Context",
        max_tokens: int,
        chars_per_token: int,
    ) -> None:
        super().__init__(event, context)
        self._max_tokens = max_tokens
        self._chars_per_token = chars_per_token

    @staticmethod
    def _skip_leading_tool_results(events: Sequence[BaseEvent], start: int) -> int:
        while start < len(events) and isinstance(events[start], ToolResultsEvent):
            start += 1
        return start

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        event_tokens = [estimated_tokens(event, self._chars_per_token) for event in events]
        if sum(event_tokens) <= self._max_tokens:
            return await call_next(events, context)

        prefix_length = 1 if isinstance(events[0], ModelRequest) else 0
        current_tokens = event_tokens[0] if prefix_length else 0
        retained_start = len(events)

        for idx in range(len(events) - 1, prefix_length - 1, -1):
            event_token_count = event_tokens[idx]
            # Always preserve the most recent event, even if it exceeds the remaining budget.
            if retained_start == len(events) or current_tokens + event_token_count <= self._max_tokens:
                retained_start = idx
                current_tokens += event_token_count
            else:
                break

        retained_start = self._skip_leading_tool_results(events, retained_start)
        trimmed = events[retained_start:]
        if prefix_length:
            trimmed = [events[0], *trimmed]

        return await call_next(trimmed, context)
