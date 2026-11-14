# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from ag2._replay import replayable_span
from ag2.annotations import Context
from ag2.events import BaseEvent, ModelRequest, ModelResponse
from ag2.middleware.base import BaseMiddleware, LLMCall, MiddlewareFactory
from ag2.middleware.describe import MiddlewareDescription


class HistoryLimiter(MiddlewareFactory):
    def __init__(self, max_events: int) -> None:
        if max_events < 1:
            raise ValueError("max_events must be greater than 0")
        self._max_events = max_events

    def describe(self) -> MiddlewareDescription:
        return MiddlewareDescription(
            kind=type(self).__qualname__,
            config={"max_events": self._max_events},
        )

    def __call__(self, event: "BaseEvent", context: "Context") -> "BaseMiddleware":
        return _HistoryLimiter(event, context, self._max_events)


class _HistoryLimiter(BaseMiddleware):
    """Truncate message history to a maximum number of events.

    ``max_events`` is a target, not a guarantee: events the cut orphaned are
    dropped from the tail, and a tail that would reduce to nothing widens instead
    (see :mod:`ag2._replay`).
    """

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
        if self._max_events == 1:
            cut = next(
                (
                    len(events) - offset - 1
                    for offset, request in enumerate(reversed(events))
                    if isinstance(request, ModelRequest) and request.origin == "user"
                ),
                len(events) - 1,
            )
            # Preserve the current question and its tool round when present.
            trimmed: Sequence[BaseEvent] = replayable_span(events, cut)
        elif isinstance(first, ModelRequest):
            # Reduce over the tail so replay can widen without duplicating first.
            tail = events[1:]
            trimmed = [first, *replayable_span(tail, len(tail) - (self._max_events - 1))]
        else:
            trimmed = replayable_span(events, len(events) - self._max_events)

        return await call_next(trimmed, context)
