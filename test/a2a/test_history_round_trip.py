# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.a2a.mappers.history import events_to_payload, payload_to_events
from ag2.events import (
    ToolCallEvent,
    ToolCallsEvent,
    ToolResult,
    ToolResultEvent,
    ToolResultsEvent,
)


def test_tool_calls_event_wrapper_round_trips() -> None:
    original = ToolCallsEvent([
        ToolCallEvent(id="call-1", name="get_local_time", arguments="{}"),
        ToolCallEvent(id="call-2", name="save_local_note", arguments='{"title":"x","body":"y"}'),
    ])

    [restored] = payload_to_events(events_to_payload([original]))

    assert restored == original


def test_tool_results_event_wrapper_round_trips() -> None:
    original = ToolResultsEvent([
        ToolResultEvent(parent_id="call-1", name="get_local_time", result=ToolResult("2026-05-09T16:00:00")),
        ToolResultEvent(parent_id="call-2", name="save_local_note", result=ToolResult("Saved.")),
    ])

    [restored] = payload_to_events(events_to_payload([original]))

    assert restored == original


def test_wrapper_and_leaves_serialize_independently() -> None:
    # If the wrapper gets dropped on the wire the Anthropic mapper falls
    # back to per-leaf handling, which then accesses ``message.content`` —
    # an attribute ToolResultEvent does not have — and crashes.
    leaf = ToolResultEvent(parent_id="call-1", name="t", result=ToolResult("ok"))
    wrapper = ToolResultsEvent([leaf])

    restored = payload_to_events(events_to_payload([wrapper, leaf]))

    assert restored == [wrapper, leaf]
