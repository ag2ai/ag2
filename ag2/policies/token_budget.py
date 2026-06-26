# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""TokenBudgetPolicy — keep events within a token budget."""

from ag2.context import ConversationContext as Context
from ag2.events import BaseEvent

from ._pairing import ensure_tool_pairing


class TokenBudgetPolicy:
    """Keep events within a token budget.

    Estimates tokens by character count. Retains most recent events first.
    """

    name = "token_budget"

    def __init__(self, max_tokens: int, chars_per_token: int = 4, transparent: bool = False) -> None:
        self._max_chars = max_tokens * chars_per_token
        self._transparent = transparent

    async def apply(
        self,
        prompts: list[str],
        events: list[BaseEvent],
        context: Context,
    ) -> tuple[list[str], list[BaseEvent]]:
        total_chars = sum(len(str(e)) for e in events)
        if total_chars <= self._max_chars:
            return prompts, events

        # Retain from the end, fitting within budget
        retained: list[BaseEvent] = []
        budget = self._max_chars
        for event in reversed(events):
            cost = len(str(event))
            if budget - cost < 0 and retained:
                break
            retained.append(event)
            budget -= cost
        retained.reverse()
        retained = ensure_tool_pairing(retained)

        if self._transparent:
            prompts = prompts + [f"[{self.name}] Showing {len(retained)} of {len(events)} events (token budget)."]
        return prompts, retained
