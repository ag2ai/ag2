# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Internal helpers: enforce event pairing after history trimming."""

from ag2.events import (
    BaseEvent,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    ModelReasoning,
    ModelResponse,
    ToolResultsEvent,
    is_conversational,
)

# Events a provider emits as one group behind a single reasoning item.
_REASONING_GROUP = (BuiltinToolCallEvent, BuiltinToolResultEvent, ModelReasoning)


def ensure_tool_pairing(events: list[BaseEvent]) -> list[BaseEvent]:
    """Drop ToolResultsEvents whose matching ToolCallsEvent was trimmed away.

    Scans the full event list (not only the head) and removes any
    ToolResultsEvent that has no surviving ToolCallEvent ancestor. Required by
    providers (e.g. OpenAI) that reject ``tool``-role messages without a
    preceding ``tool_calls`` message.
    """
    call_ids: set[str] = set()
    for event in events:
        if isinstance(event, ModelResponse) and event.tool_calls:
            call_ids.update(call.id for call in event.tool_calls.calls)
    return [
        event
        for event in events
        if not isinstance(event, ToolResultsEvent) or any(result.parent_id in call_ids for result in event.results)
    ]


def _anchors(events: list[BaseEvent]) -> bool:
    """True if any durable reasoning item is present."""
    return any(isinstance(e, ModelReasoning) and is_conversational(e) for e in events)


def safe_cut(events: list[BaseEvent], start: int) -> int:
    """Move a trim point forward so no builtin tool call loses its reasoning item.

    Providers (e.g. OpenAI Responses) reject a replayed server-side tool call
    whose paired reasoning item was trimmed away, so a group split by the cut is
    dropped whole rather than half-kept.
    """
    if start <= 0:
        return 0
    if start >= len(events) or not isinstance(events[start], _REASONING_GROUP):
        return start

    head = start
    while head > 0 and isinstance(events[head - 1], _REASONING_GROUP):
        head -= 1
    if not _anchors(events[head:start]):
        return start

    # Skip the orphaned remainder, stopping at the next group's own anchor.
    while start < len(events) and isinstance(events[start], _REASONING_GROUP):
        if _anchors([events[start]]):
            break
        start += 1
    return start
