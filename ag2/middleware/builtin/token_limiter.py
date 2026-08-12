# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from ag2._replay import replayable_span
from ag2.annotations import Context
from ag2.events import BaseEvent, ModelRequest, ModelResponse, estimated_tokens
from ag2.middleware.base import BaseMiddleware, LLMCall, MiddlewareFactory
from ag2.middleware.describe import MiddlewareDescription


class TokenLimiter(MiddlewareFactory):
    """Truncate message history to fit within a token budget.

    Sizes each event with the shared content estimate (text by
    ``chars_per_token``, non-text by a per-modality budget) — never the
    truncated ``str(event)`` repr.

    ``max_tokens`` is a target, not a guarantee: events the cut orphaned are
    dropped from the tail, and a tail that would reduce to nothing widens past the
    budget instead (see :mod:`ag2._replay`).
    """

    def __init__(self, max_tokens: int, chars_per_token: int = 4) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be greater than 0")
        if chars_per_token < 1:
            raise ValueError("chars_per_token must be greater than 0")
        self._max_tokens = max_tokens
        self._chars_per_token = chars_per_token

    def describe(self) -> MiddlewareDescription:
        return MiddlewareDescription(
            kind=type(self).__qualname__,
            config={"max_tokens": self._max_tokens, "chars_per_token": self._chars_per_token},
        )

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

        if prefix_length:
            # Reduced over the tail rather than the whole list: events[0] is
            # re-attached here, and a span widened past index 0 of the full list
            # would duplicate it. Nothing is lost by excluding it — a ModelRequest
            # answers no call and anchors no builtin one.
            tail = events[1:]
            trimmed: Sequence[BaseEvent] = [events[0], *replayable_span(tail, retained_start - 1)]
        else:
            trimmed = replayable_span(events, retained_start)

        return await call_next(trimmed, context)
