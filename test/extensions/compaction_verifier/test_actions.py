# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.events import (
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolNotFoundEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from ag2.extensions.compaction_verifier import actions_from_events, call_signature

from .conftest import record_trajectory


class TestCallSignature:
    def test_key_order_and_whitespace_do_not_matter(self) -> None:
        assert call_signature("f", '{"b": 2, "a": 1}') == call_signature("f", '{"a":1,"b":2}')

    def test_values_are_kept_verbatim(self) -> None:
        assert call_signature("f", '{"a": 1}') != call_signature("f", '{"a": "1"}')
        assert call_signature("f", '{"a": 1}') == 'f({"a":1})'

    def test_empty_arguments_are_an_empty_object(self) -> None:
        assert call_signature("ping", "") == call_signature("ping", "{}") == "ping({})"

    def test_malformed_json_keeps_the_stripped_raw_text(self) -> None:
        assert call_signature("f", ' {"a": ') == 'f({"a":)'


class TestActionsFromEvents:
    def test_success_and_failure(self) -> None:
        ok = ToolCallEvent("get", arguments='{"k": "x"}', id="1")
        bad = ToolCallEvent("get", arguments='{"k": "y"}', id="2")
        events = [
            ModelResponse(tool_calls=ToolCallsEvent([ok, bad])),
            ToolResultsEvent([
                ToolResultEvent.from_call(ok, "value"),
                ToolErrorEvent.from_call(bad, KeyError("y")),
            ]),
        ]

        actions = actions_from_events(events)

        assert [(a.call_id, a.blocked, a.error_type) for a in actions] == [
            ("1", False, None),
            ("2", True, "KeyError"),
        ]

    def test_unknown_tool_is_blocked(self) -> None:
        call = ToolCallEvent("nope", id="1")
        events = [ToolCallsEvent([call]), ToolResultsEvent([ToolNotFoundEvent.from_call(call, ValueError("nope"))])]

        (action,) = actions_from_events(events)

        assert action.blocked

    def test_a_call_without_a_result_is_not_an_action(self) -> None:
        call = ToolCallEvent("get", id="1")

        assert actions_from_events([ModelResponse(tool_calls=ToolCallsEvent([call]))]) == []

    def test_containers_and_individual_events_count_once(self) -> None:
        call = ToolCallEvent("get", id="1")
        result = ToolResultEvent.from_call(call, "v")
        events = [
            ModelResponse(tool_calls=ToolCallsEvent([call])),
            ToolCallsEvent([call]),
            call,
            result,
            ToolResultsEvent([result]),
        ]

        assert len(actions_from_events(events)) == 1

    @pytest.mark.asyncio
    async def test_reads_a_recorded_agent_history_in_call_order(self) -> None:
        events = await record_trajectory(fetches=3)

        actions = actions_from_events(events)

        assert [a.name for a in actions] == ["login", "fetch", "fetch", "fetch"]
        assert not any(a.blocked for a in actions)
        assert actions[1].signature == 'fetch({"item":"a","token":"tok-7f3a"})'
