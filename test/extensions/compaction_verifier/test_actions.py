# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.events import (
    BaseEvent,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolNotFoundEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from ag2.extensions.compaction_verifier import actions_from_events, call_signature
from ag2.extensions.compaction_verifier.actions import actions_with_positions

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

    def test_an_id_reused_by_a_later_response_is_a_new_call(self) -> None:
        # AG2's Ollama client names the first call of every response call_0
        events: list[BaseEvent] = []
        for i in range(7):
            call = ToolCallEvent("fetch", arguments=f'{{"i": {i}}}', id="call_0")
            result = ToolResultEvent.from_call(call, "ok")
            events += [ModelResponse(tool_calls=ToolCallsEvent([call])), ToolCallsEvent([call]), call]
            events += [result, ToolResultsEvent([result])]

        actions = actions_from_events(events)

        assert [a.signature for a in actions] == [f'fetch({{"i":{i}}})' for i in range(7)]

    def test_parallel_calls_numbered_per_response(self) -> None:
        events: list[BaseEvent] = []
        for turn in range(2):
            a = ToolCallEvent("get", arguments=f'{{"k": "a{turn}"}}', id="call_0")
            b = ToolCallEvent("get", arguments=f'{{"k": "b{turn}"}}', id="call_1")
            events += [
                ModelResponse(tool_calls=ToolCallsEvent([a, b])),
                ToolResultsEvent([ToolResultEvent.from_call(a, "v"), ToolErrorEvent.from_call(b, KeyError("b"))]),
            ]

        actions = actions_from_events(events)

        assert [(a.signature, a.blocked) for a in actions] == [
            ('get({"k":"a0"})', False),
            ('get({"k":"b0"})', True),
            ('get({"k":"a1"})', False),
            ('get({"k":"b1"})', True),
        ]

    def test_positions_are_the_answering_events(self) -> None:
        a, b = ToolCallEvent("f", id="call_0"), ToolCallEvent("g", id="call_0")
        events = [
            ModelResponse(tool_calls=ToolCallsEvent([a])),
            ToolResultsEvent([ToolResultEvent.from_call(a, "1")]),
            ModelResponse(tool_calls=ToolCallsEvent([b])),
            ToolResultsEvent([ToolResultEvent.from_call(b, "2")]),
        ]

        assert [(x.name, at) for x, at in actions_with_positions(events)] == [("f", 1), ("g", 3)]

    def test_two_calls_sharing_an_id_in_one_response_cannot_be_paired(self) -> None:
        a, b = ToolCallEvent("f", id="x"), ToolCallEvent("g", id="x")

        with pytest.raises(ValueError, match="several calls with one id"):
            actions_from_events([ModelResponse(tool_calls=ToolCallsEvent([a, b]))])

    def test_an_id_reused_before_its_call_is_answered_cannot_be_paired(self) -> None:
        a, b = ToolCallEvent("f", id="x"), ToolCallEvent("g", id="x")

        with pytest.raises(ValueError, match="still awaiting its result"):
            actions_from_events([ToolCallsEvent([a]), ToolCallsEvent([b])])

    @pytest.mark.asyncio
    async def test_a_recorded_history_reads_the_same_with_ollama_ids(self) -> None:
        unique = actions_from_events(await record_trajectory(fetches=6))
        ollama = actions_from_events(await record_trajectory(fetches=6, ollama_ids=True))

        assert len(ollama) == 7
        assert {a.call_id for a in ollama} == {"call_0"}
        assert [(a.signature, a.blocked) for a in ollama] == [(a.signature, a.blocked) for a in unique]

    @pytest.mark.asyncio
    async def test_reads_a_recorded_agent_history_in_call_order(self) -> None:
        events = await record_trajectory(fetches=3)

        actions = actions_from_events(events)

        assert [a.name for a in actions] == ["login", "fetch", "fetch", "fetch"]
        assert not any(a.blocked for a in actions)
        assert actions[1].signature == 'fetch({"item":"a","token":"tok-7f3a"})'
