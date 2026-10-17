# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""TokenBudgetPolicy — keep events within a token budget."""

from ag2._replay import replayable_span
from ag2.context import ConversationContext as Context
from ag2.events import BaseEvent, estimated_tokens


class TokenBudgetPolicy:
    """Keep events within a token budget.

    Estimates full text content by character count and non-text parts by a
    per-modality budget. Retains most recent events first.

    The budget is a target, not a guarantee: events the cut orphaned are dropped
    from the span, and a span that would reduce to nothing widens past the budget
    instead.
    """

    name = "token_budget"

    def __init__(self, max_tokens: int, chars_per_token: int = 4, transparent: bool = False) -> None:
        if chars_per_token < 1:
            raise ValueError("chars_per_token must be greater than 0")
        self._max_tokens = max_tokens
        self._chars_per_token = chars_per_token
        self._transparent = transparent

    async def apply(
        self,
        prompts: list[str],
        events: list[BaseEvent],
        context: Context,
    ) -> tuple[list[str], list[BaseEvent]]:
        event_tokens = [estimated_tokens(event, self._chars_per_token) for event in events]
        if sum(event_tokens) <= self._max_tokens:
            return prompts, events

        # Retain from the end, fitting within budget
        retained: list[BaseEvent] = []
        budget = self._max_tokens
        for event, cost in zip(reversed(events), reversed(event_tokens)):
            if budget - cost < 0 and retained:
                break
            retained.append(event)
            budget -= cost
        retained.reverse()
        # The loop breaks on the first event that does not fit, so `retained` is a
        # contiguous suffix and its length names the cut. Pruning that span only
        # removes events, so the budget holds — except where the span would prune
        # to nothing and widens instead, which overshoots the budget deliberately:
        # an empty request is rejected outright, an oversized one is not.
        retained = replayable_span(events, len(events) - len(retained))

        if self._transparent:
            prompts = prompts + [f"[{self.name}] Showing {len(retained)} of {len(events)} events (token budget)."]
        return prompts, retained
