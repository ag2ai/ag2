# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, ToolResults
from autogen.beta.middlewares.base import BaseMiddleware, LLMCall, MiddlewareFactory


class HistoryLimiter(MiddlewareFactory):
    def __init__(self, max_events: int) -> None:
        if max_events < 1:
            raise ValueError("max_events must be greater than 0")
        self._max_events = max_events

    def __call__(self, event: "BaseEvent", ctx: "Context") -> "BaseMiddleware":
        return _HistoryLimiter(event, ctx, self._max_events)


class _HistoryLimiter(BaseMiddleware):
    """Truncate message history to a maximum number of events."""

    def __init__(self, event: "BaseEvent", ctx: "Context", max_events: int) -> None:
        super().__init__(event, ctx)
        self._max_events = max_events

    @staticmethod
    def _skip_leading_tool_results(events: Sequence[BaseEvent], start: int) -> int:
        while start < len(events) and isinstance(events[start], ToolResults):
            start += 1
        return start

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        ctx: Context,
    ) -> ModelResponse:
        if len(events) <= self._max_events:
            return await call_next(events, ctx)

        first = events[0]
        if isinstance(first, ModelRequest):
            if self._max_events == 1:
                trimmed = [first]
            else:
                tail_start = len(events) - (self._max_events - 1)
                tail_start = self._skip_leading_tool_results(events, tail_start)
                trimmed = [first, *events[tail_start:]]
        else:
            start = self._skip_leading_tool_results(events, len(events) - self._max_events)
            trimmed = list(events[start:])

        return await call_next(trimmed, ctx)
