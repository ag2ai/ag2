# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, ToolResultsEvent
from autogen.beta.middleware.base import BaseMiddleware, LLMCall, MiddlewareFactory


class HistoryLimiter(MiddlewareFactory):
    def __init__(self, max_events: int) -> None:
        if max_events < 1:
            raise ValueError("max_events must be greater than 0")
        self._max_events = max_events

    def __call__(self, event: "BaseEvent", context: "Context") -> "BaseMiddleware":
        return _HistoryLimiter(event, context, self._max_events)


class _HistoryLimiter(BaseMiddleware):
    """Truncate message history to a maximum number of events."""

    def __init__(self, event: "BaseEvent", context: "Context", max_events: int) -> None:
        super().__init__(event, context)
        self._max_events = max_events

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        if len(events) <= self._max_events:
            return await call_next(events, context)

        first = events[0]
        if isinstance(first, ModelRequest):
            if self._max_events == 1:
                trimmed: Sequence[BaseEvent] = [first]
            else:
                tail_start = len(events) - (self._max_events - 1)
                tail_start = _skip_leading_tool_results(events, tail_start)
                trimmed = [first, *events[tail_start:]]
        else:
            start = _skip_leading_tool_results(events, len(events) - self._max_events)
            trimmed = events[start:]

        return await call_next(trimmed, context)


def _skip_leading_tool_results(events: Sequence[BaseEvent], start: int) -> int:
    while start < len(events) and isinstance(events[start], ToolResultsEvent):
        start += 1
    return start
