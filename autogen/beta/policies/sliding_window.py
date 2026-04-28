# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""SlidingWindowPolicy — keep the last N events."""

from autogen.beta.context import ConversationContext as Context
from autogen.beta.events import BaseEvent, ToolResultsEvent


class SlidingWindowPolicy:
    """Keep the last N events. Drop older events.

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
        trimmed = events[-self._max :]
        # Skip leading ToolResultsEvents whose matching tool_use was trimmed away.
        while trimmed and isinstance(trimmed[0], ToolResultsEvent):
            trimmed = trimmed[1:]
        if self._transparent:
            prompts = prompts + [f"[{self.name}] Showing last {len(trimmed)} of {total} events."]
        return prompts, trimmed
