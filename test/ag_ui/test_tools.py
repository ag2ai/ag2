# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tool calls through `AGUIStream`: the server's own tools, the client's, and what a result looks like to the client."""

import json
import logging
from base64 import b64encode
from typing import Any

import pytest
from ag_ui.core import (
    PROTOCOL_VERSION,
    AssistantMessage,
    FunctionCall,
    RunFinishedEvent,
    RunFinishedSuccessOutcome,
    TextMessageChunkEvent,
    ToolCall,
    ToolCallArgsEvent,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
    ToolMessage,
    UserMessage,
)
from dirty_equals import IsStr

from ag2 import Agent, ToolResult
from ag2.ag_ui import AGUIStream
from ag2.events import (
    AudioInput,
    BinaryInput,
    DataInput,
    DocumentInput,
    FileIdInput,
    ImageInput,
    Input,
    TextInput,
    ToolCallEvent,
    ToolResultsEvent,
    UrlInput,
)
from ag2.testing import TestConfig
from test.ag_ui.harness import dispatch_events, each, kinds_of, recording_history, run_input, sole, weather_tool, wire

pytestmark = pytest.mark.asyncio

PNG = b"\x89PNG\r\n"
PNG_B64 = b64encode(PNG).decode()


class TestBackendTools:
    async def test_a_call_is_announced_closed_and_then_answered(self) -> None:
        agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="get_current_time"), "It is 10:30."))

        @agent.tool
        def get_current_time() -> str:
            return "10:30"

        events = await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="What time is it?")))

        # Closed before it runs, so a call paused mid-run is never left open.
        assert [kind for kind in kinds_of(events) if "ToolCall" in kind.__name__] == [
            ToolCallStartEvent,
            ToolCallArgsEvent,
            ToolCallEndEvent,
            ToolCallResultEvent,
        ]
        assert sole(events, TextMessageChunkEvent).delta == "It is 10:30."

    async def test_the_events_of_one_call_share_its_id(self) -> None:
        agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="my_tool"), "Done"))

        @agent.tool
        def my_tool() -> str:
            return "result"

        events = await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="Call my_tool")))

        call_id = sole(events, ToolCallStartEvent).tool_call_id
        assert wire(sole(events, ToolCallStartEvent)) == {
            "type": "TOOL_CALL_START",
            "toolCallId": call_id,
            "toolCallName": "my_tool",
        }
        assert wire(sole(events, ToolCallArgsEvent)) == {"type": "TOOL_CALL_ARGS", "toolCallId": call_id, "delta": "{}"}
        assert wire(sole(events, ToolCallEndEvent)) == {"type": "TOOL_CALL_END", "toolCallId": call_id}
        assert wire(sole(events, ToolCallResultEvent)) == {
            "type": "TOOL_CALL_RESULT",
            "toolCallId": call_id,
            "messageId": IsStr(),
            "content": "result",
            "role": "tool",
        }

    async def test_arguments_and_result_are_those_of_the_call(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig(ToolCallEvent(name="calculate_sum", arguments='{"a":5,"b":3}'), "The sum is 8."),
        )

        @agent.tool
        def calculate_sum(a: int, b: int) -> int:
            return a + b

        events = await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="What is 5 + 3?")))

        assert json.loads(sole(events, ToolCallArgsEvent).delta) == {"a": 5, "b": 3}
        assert sole(events, ToolCallResultEvent).content == "8"

    async def test_parallel_calls_are_each_answered(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig((ToolCallEvent(name="tool_a"), ToolCallEvent(name="tool_b")), "Both done."),
        )

        @agent.tool
        def tool_a() -> str:
            return "Result A"

        @agent.tool
        def tool_b() -> str:
            return "Result B"

        events = await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="Call both tools")))

        assert sorted(event.tool_call_name for event in each(events, ToolCallStartEvent)) == ["tool_a", "tool_b"]
        assert sorted(event.content for event in each(events, ToolCallResultEvent)) == ["Result A", "Result B"]


class TestClientTools:
    async def test_a_call_to_a_client_tool_is_left_to_the_client(self) -> None:
        agent = Agent(
            "test_agent", config=TestConfig(ToolCallEvent(name="get_weather", arguments='{"location":"Paris"}'))
        )
        incoming = run_input(UserMessage(id="m1", content="Weather in Paris?"), tools=[weather_tool()])

        events = await dispatch_events(AGUIStream(agent), incoming)

        assert wire(sole(events, ToolCallChunkEvent)) == {
            "type": "TOOL_CALL_CHUNK",
            "toolCallId": IsStr(),
            "toolCallName": "get_weather",
            "delta": '{"location":"Paris"}',
        }
        assert each(events, ToolCallResultEvent) == []

    async def test_a_server_tool_of_the_same_name_is_not_replaced(self) -> None:
        runs: list[str] = []

        def get_weather(location: str) -> str:
            runs.append(location)
            return "sunny"

        agent = Agent(
            "test_agent",
            tools=[get_weather],
            config=TestConfig(ToolCallEvent(name="get_weather", arguments='{"location":"Paris"}'), "Sunny in Paris."),
        )
        incoming = run_input(UserMessage(id="m1", content="Weather in Paris?"), tools=[weather_tool()])

        events = await dispatch_events(AGUIStream(agent), incoming)

        assert runs == ["Paris"]
        assert each(events, ToolCallChunkEvent) == []
        assert sole(events, RunFinishedEvent).outcome == RunFinishedSuccessOutcome()

    async def test_the_result_the_client_sends_back_reaches_the_model(self) -> None:
        agent = Agent("test_agent", config=TestConfig("Sunny in Paris."))
        middleware, calls = recording_history()
        incoming = run_input(
            UserMessage(id="m1", content="Weather in Paris?"),
            AssistantMessage(
                id="m2",
                tool_calls=[
                    ToolCall(id="c1", function=FunctionCall(name="get_weather", arguments='{"location": "Paris"}'))
                ],
            ),
            ToolMessage(id="m3", content="Sunny, 22°C", tool_call_id="c1"),
            tools=[weather_tool()],
        )

        events = await dispatch_events(AGUIStream(agent), incoming, middleware=[middleware])

        [call] = calls
        [results] = [event for event in call.events if isinstance(event, ToolResultsEvent)]
        [result] = results.results
        assert (result.parent_id, result.name, result.result.parts) == ("c1", "get_weather", [TextInput("Sunny, 22°C")])
        assert sole(events, TextMessageChunkEvent).delta == "Sunny in Paris."

    async def test_several_calls_to_one_client_tool_are_each_left_to_the_client(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig([
                ToolCallEvent(name="get_weather", arguments='{"location":"Paris"}'),
                ToolCallEvent(name="get_weather", arguments='{"location":"London"}'),
            ]),
        )
        incoming = run_input(UserMessage(id="m1", content="Paris and London?"), tools=[weather_tool()])

        events = await dispatch_events(AGUIStream(agent), incoming)

        # Parallel calls carry no order the client can rely on.
        assert sorted(event.delta for event in each(events, ToolCallChunkEvent)) == [
            '{"location":"London"}',
            '{"location":"Paris"}',
        ]

    async def test_server_and_client_tools_in_one_turn(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig([
                ToolCallEvent(name="get_current_time"),
                ToolCallEvent(name="get_weather", arguments='{"location":"London"}'),
            ]),
        )

        @agent.tool
        def get_current_time() -> str:
            return "10:30"

        incoming = run_input(UserMessage(id="m1", content="Time, and weather?"), tools=[weather_tool()])

        events = await dispatch_events(AGUIStream(agent), incoming)

        assert sole(events, ToolCallStartEvent).tool_call_name == "get_current_time"
        assert sole(events, ToolCallChunkEvent).tool_call_name == "get_weather"


async def _result_content(
    result: ToolResult, *, protocol_version: str | None = PROTOCOL_VERSION
) -> str | list[dict[str, Any]]:
    """What the client is sent for a tool that returns `result`, as it is on the wire."""
    agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="produce"), "done"))

    @agent.tool
    def produce() -> ToolResult:
        return result

    events = await dispatch_events(
        AGUIStream(agent), run_input(UserMessage(id="m1", content="go"), protocol_version=protocol_version)
    )
    return wire(sole(events, ToolCallResultEvent))["content"]


class TestAResultForAOneZeroClient:
    async def test_a_lone_text_is_a_plain_string(self) -> None:
        assert await _result_content(ToolResult("sunny")) == "sunny"

    async def test_text_beside_an_image_arrives_as_both_parts(self) -> None:
        content = await _result_content(ToolResult("the chart", ImageInput(data=PNG, media_type="image/png")))

        assert content == [
            {"type": "text", "text": "the chart"},
            {"type": "image", "source": {"type": "data", "value": PNG_B64, "mimeType": "image/png"}},
        ]

    async def test_structured_output_travels_as_its_text(self) -> None:
        """The protocol has no JSON part."""
        content = await _result_content(ToolResult(DataInput({"temp": 22}), TextInput("ok")))

        assert content == [{"type": "text", "text": '{"temp":22}'}, {"type": "text", "text": "ok"}]

    @pytest.mark.parametrize(
        "part,expected",
        [
            (
                ImageInput("https://x/a.png"),
                {"type": "image", "source": {"type": "url", "value": "https://x/a.png"}},
            ),
            (
                AudioInput(data=b"RIFF", media_type="audio/wav"),
                {
                    "type": "audio",
                    "source": {"type": "data", "value": b64encode(b"RIFF").decode(), "mimeType": "audio/wav"},
                },
            ),
            (
                DocumentInput("https://x/a.pdf"),
                {"type": "document", "source": {"type": "url", "value": "https://x/a.pdf"}},
            ),
            (
                UrlInput("https://x/blob", kind="binary"),
                {"type": "document", "source": {"type": "url", "value": "https://x/blob"}},
            ),
            (
                BinaryInput(b"\x00", media_type="application/octet-stream"),
                {
                    "type": "document",
                    "source": {
                        "type": "data",
                        "value": b64encode(b"\x00").decode(),
                        "mimeType": "application/octet-stream",
                    },
                },
            ),
            (
                FileIdInput("file-abc"),
                {"type": "document", "source": {"type": "file", "value": "file-abc"}},
            ),
        ],
    )
    async def test_each_part_becomes_the_content_part_of_its_kind(self, part: Input, expected: dict[str, Any]) -> None:
        assert await _result_content(ToolResult("see", part)) == [{"type": "text", "text": "see"}, expected]

    async def test_metadata_carries_over(self) -> None:
        image = ImageInput("https://x/a.png")
        image.metadata = {"alt": "a chart"}

        content = await _result_content(ToolResult(image))

        assert content == [
            {"type": "image", "source": {"type": "url", "value": "https://x/a.png"}, "metadata": {"alt": "a chart"}}
        ]

    async def test_a_lone_text_with_metadata_keeps_it_in_a_part(self) -> None:
        """A plain string has nowhere to put metadata, and dropping it would lose it."""
        text = TextInput("sunny")
        text.metadata = {"source": "met office"}

        content = await _result_content(ToolResult(text))

        assert content == [{"type": "text", "text": "sunny", "metadata": {"source": "met office"}}]

    async def test_a_result_of_nothing_is_the_empty_string(self) -> None:
        assert await _result_content(ToolResult()) == ""


class TestAResultForAClientPredatingOneZero:
    """Its schema reads a tool result as a string, so it gets the text and nothing invented."""

    async def test_a_lone_text_is_the_same_string(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            content = await _result_content(ToolResult("sunny"), protocol_version=None)

        assert content == "sunny"
        assert caplog.records == []

    async def test_media_are_dropped_and_the_loss_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        result = ToolResult(
            "the chart",
            ImageInput(data=PNG, media_type="image/png"),
            DataInput({"temp": 22}),
            FileIdInput("file-abc"),
        )

        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            content = await _result_content(result, protocol_version=None)

        assert content == 'the chart\n{"temp":22}'
        [warning] = caplog.records
        assert "document" in warning.getMessage()
        assert "image" in warning.getMessage()
        assert "@ag-ui/* 1.0" in warning.getMessage()

    async def test_a_result_of_media_only_is_the_empty_string(self) -> None:
        content = await _result_content(ToolResult(ImageInput(data=PNG, media_type="image/png")), protocol_version=None)

        assert content == ""
