# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""SlidingWindowPolicy — keep the last N events."""

from autogen.beta.context import ConversationContext as Context
from autogen.beta.events import BaseEvent, ModelResponse, ToolResultsEvent


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
        if total > self._max:
            window = events[-self._max :]
        else:
            window = events
        repaired = _drop_orphan_tool_results(window)
        if self._transparent and total > self._max:
            prompts = prompts + [f"[{self.name}] Showing last {len(repaired)} of {total} events."]
        return prompts, repaired


def _drop_orphan_tool_results(events: list[BaseEvent]) -> list[BaseEvent]:
    """Remove ToolResultEvents whose matching ToolCallEvent is not in the window.

    OpenAI rejects the assembled messages with HTTP 400 when a ``tool``-role
    message references a ``tool_call_id`` that is not present in any preceding
    ``assistant`` message's ``tool_calls`` (see issue #2793). The window can
    contain such orphans for two reasons:

    1. The matching ModelResponse holding the ToolCallsEvent was older than
       the window and got trimmed away. The orphan can sit at the head of
       the window or, if a complete tool round-trip survived in front of it,
       in the middle.
    2. Sequential agents share context across separate streams, so the
       assembled history can carry tool results emitted on one stream
       alongside no matching call from another stream — and this can occur
       even when no trimming has happened.

    For each ToolResultsEvent we keep only the individual ToolResultEvents
    whose ``parent_id`` matches a ToolCallEvent ``id`` already seen earlier
    in the window (extracted from ModelResponse.tool_calls). If every result
    in a container is orphaned, the whole container is dropped.
    """
    seen_call_ids: set[str] = set()
    kept: list[BaseEvent] = []
    for event in events:
        if isinstance(event, ModelResponse):
            seen_call_ids.update(call.id for call in event.tool_calls.calls)
            kept.append(event)
            continue
        if isinstance(event, ToolResultsEvent):
            paired = [r for r in event.results if r.parent_id in seen_call_ids]
            if not paired:
                continue
            if len(paired) == len(event.results):
                kept.append(event)
            else:
                kept.append(ToolResultsEvent(results=paired))
            continue
        kept.append(event)
    return kept
