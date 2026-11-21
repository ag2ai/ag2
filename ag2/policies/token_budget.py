# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""TokenBudgetPolicy — keep events within a token budget."""

from ag2._replay import replayable_span
from ag2.assembly import AssemblyPolicy
from ag2.context import ConversationContext as Context
from ag2.events import BaseEvent, estimated_tokens


class TokenBudgetPolicy(AssemblyPolicy):
    """Keep events within a token budget.

    Sizes each event from its full content with the shared estimate (text by
    ``chars_per_token``, non-text by a per-modality budget), never from the
    truncated ``str(event)`` repr. Retains most recent events first.

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
        costs = [estimated_tokens(e, self._chars_per_token) for e in events]
        if sum(costs) <= self._max_tokens:
            return prompts, events

        # Retain from the end, fitting within budget
        retained: list[BaseEvent] = []
        budget = self._max_tokens
        for event, cost in zip(reversed(events), reversed(costs), strict=True):
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
