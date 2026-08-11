# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Internal helper: enforce tool call/result pairing after history trimming."""

from ag2.events import BaseEvent, ModelResponse, ToolResultsEvent


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
