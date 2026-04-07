# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, ToolResultsEvent
from autogen.beta.middleware.base import BaseMiddleware, LLMCall, MiddlewareFactory


class TokenLimiter(MiddlewareFactory):
    """Truncate message history to fit within a token budget.

    Uses a simple character-based estimate (``chars_per_token`` chars per token)
    unless a custom tokenizer is provided.
    """

    def __init__(self, max_tokens: int, chars_per_token: int = 4) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be greater than 0")
        if chars_per_token < 1:
            raise ValueError("chars_per_token must be greater than 0")
        self._max_chars = max_tokens // chars_per_token

    def __call__(self, event: "BaseEvent", context: "Context") -> "BaseMiddleware":
        return _TokenLimiter(event, context, self._max_chars)


class _TokenLimiter(BaseMiddleware):
    def __init__(
        self,
        event: "BaseEvent",
        context: "Context",
        max_chars: int,
    ) -> None:
        super().__init__(event, context)
        self._max_chars = max_chars

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
        event_lengths = [len(str(event)) for event in events]
        if sum(event_lengths) <= self._max_chars:
            return await call_next(events, context)

        prefix_length = 1 if isinstance(events[0], ModelRequest) else 0
        current_chars = event_lengths[0] if prefix_length else 0
        retained_start = len(events)

        for idx in range(len(events) - 1, prefix_length - 1, -1):
            event_chars = event_lengths[idx]
            # Always preserve the most recent event, even if it exceeds the remaining budget.
            if retained_start == len(events) or current_chars + event_chars <= self._max_chars:
                retained_start = idx
                current_chars += event_chars
            else:
                break

        retained_start = self._skip_leading_tool_results(events, retained_start)
        trimmed = events[retained_start:]
        if prefix_length:
            trimmed = [events[0], *trimmed]

        return await call_next(trimmed, context)
