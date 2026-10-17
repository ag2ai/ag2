# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""SlidingWindowPolicy — keep the last N events."""

from ag2._replay import replayable_span
from ag2.context import ConversationContext as Context
from ag2.events import BaseEvent


class SlidingWindowPolicy:
    """Keep the last N events. Drop older events.

    The window is a target, not a guarantee: events the cut orphaned are dropped
    from it, and a window that would reduce to nothing widens instead.

    Optional transparency: injects a note about how many events were omitted.
    """

    name = "sliding_window"

    def __init__(self, max_events: int, transparent: bool = False) -> None:
        self._max = max_events
        self._transparent = transparent

    async def apply(
        self,
        prompts: list[str],
        events: list[BaseEvent],
        context: Context,
    ) -> tuple[list[str], list[BaseEvent]]:
        total = len(events)
        if total <= self._max:
            return prompts, events
        trimmed = replayable_span(events, total - self._max)
        if self._transparent:
            prompts = prompts + [f"[{self.name}] Showing last {len(trimmed)} of {total} events."]
        return prompts, trimmed
