# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Annotated
from unittest.mock import MagicMock

import pytest
from ag_ui.core import (
    AssistantMessage,
    CustomEvent,
    FunctionCall,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from dirty_equals import IsInt, IsPartialDict, IsStr

from autogen.beta import Agent, Context, Variable
from autogen.beta.ag_ui import AGUIEvent, AGUIStream
from autogen.beta.events import ToolCallEvent
from autogen.beta.testing import TestConfig

from .utils import (
    assert_event_type,
    assert_no_event_type,
    collect_events,
    create_run_input,
    get_events_of_type,
    get_weather_tool,
)

pytestmark = pytest.mark.asyncio


class TestBasicConversation:
    async def test_basic_user_message(self) -> None:
        agent = Agent("test_agent", config=TestConfig("Hello! I'm doing well, thank you for asking."))

        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="msg_1", content="Hello, how are you?"))

        events = await collect_events(stream, run_input)

        run_started = assert_event_type(events, "RUN_STARTED")
        assert run_started == IsPartialDict({
            "threadId": run_input.thread_id,
            "runId": run_input.run_id,
            "timestamp": IsInt(),
        })

        text_message = assert_event_type(events, "TEXT_MESSAGE_CHUNK")
        assert text_message == IsPartialDict({
            "delta": "Hello! I'm doing well, thank you for asking.",
            "timestamp": IsInt(),
        })

        run_finished = assert_event_type(events, "RUN_FINISHED")
        assert run_finished == IsPartialDict({
            "threadId": run_input.thread_id,
            "runId": run_input.run_id,
            "timestamp": IsInt(),
        })

    async def test_multiple_messages_history(self) -> None:
        agent = Agent("test_agent", config=TestConfig("I see you've been talking about weather. It's sunny today!"))

        stream = AGUIStream(agent)

        run_input = create_run_input(
            UserMessage(id="msg_1", content="What's the weather like?"),
            AssistantMessage(id="msg_2", content="I'll check the weather for you."),
            UserMessage(id="msg_3", content="Thanks! And tomorrow?"),
        )

        events = await collect_events(stream, run_input)

        assert_event_type(events, "RUN_STARTED")
        assert_event_type(events, "TEXT_MESSAGE_CHUNK")
        assert_event_type(events, "RUN_FINISHED")


class TestBackendTools:
    async def test_backend_tool_call_and_result(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ToolCallEvent(name="get_current_time"),
                "The current time is 2024-01-15T10:30:00Z",
            ),
        )

        @agent.tool
        def get_current_time() -> str:
            return "2024-01-15T10:30:00Z"

        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="msg_1", content="What time is it?"))

        events = await collect_events(stream, run_input)

        assert_event_type(events, "RUN_STARTED")

        tool_start = assert_event_type(events, "TOOL_CALL_START")
        assert tool_start == IsPartialDict({
            "toolCallName": "get_current_time",
        })

        tool_args = assert_event_type(events, "TOOL_CALL_ARGS")
        assert tool_args == IsPartialDict({
            "delta": "{}",
        })

        tool_result = assert_event_type(events, "TOOL_CALL_RESULT")
        assert tool_result == IsPartialDict({
            "content": IsStr(regex=r".*2024-01-15T10:30:00Z.*"),
        })

        assert_event_type(events, "TOOL_CALL_END")

        text_message = assert_event_type(events, "TEXT_MESSAGE_CHUNK")
        assert text_message == IsPartialDict({
            "delta": IsStr(regex=r".*2024-01-15T10:30:00Z.*"),
        })

        assert_event_type(events, "RUN_FINISHED")

    async def test_backend_tool_with_arguments(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ToolCallEvent(name="calculate_sum", arguments='{"a":5,"b":3}'),
                "The sum of 5 and 3 is 8.",
            ),
        )

        @agent.tool
        def calculate_sum(a: int, b: int) -> int:
            return a + b

        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="msg_1", content="What is 5 + 3?"))

        events = await collect_events(stream, run_input)

        tool_start = assert_event_type(events, "TOOL_CALL_START")
        assert tool_start == IsPartialDict({
            "toolCallName": "calculate_sum",
        })

        tool_args = assert_event_type(events, "TOOL_CALL_ARGS")
        args = json.loads(tool_args["delta"])
        assert args == IsPartialDict({
            "a": 5,
            "b": 3,
        })

        tool_result = assert_event_type(events, "TOOL_CALL_RESULT")
        assert tool_result == IsPartialDict({
            "content": IsStr(regex=r".*8.*"),
        })

    async def test_multiple_backend_tool_calls(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig(
                (
                    ToolCallEvent(name="tool_a"),
                    ToolCallEvent(name="tool_b"),
                ),
                "Both tools executed successfully.",
            ),
        )

        @agent.tool
        def tool_a() -> str:
            return "Result A"

        @agent.tool
        def tool_b() -> str:
            return "Result B"

        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="msg_1", content="Call both tools"))

        events = await collect_events(stream, run_input)

        tool_starts = get_events_of_type(events, "TOOL_CALL_START")
        assert len(tool_starts) == 2
        assert sorted(tool_starts, key=lambda e: e["toolCallName"]) == [
            IsPartialDict({
                "toolCallName": "tool_a",
            }),
            IsPartialDict({
                "toolCallName": "tool_b",
            }),
        ]

        tool_results = get_events_of_type(events, "TOOL_CALL_RESULT")
        assert len(tool_results) == 2


class TestFrontendTools:
    async def test_frontend_tool_call(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ToolCallEvent(name="get_weather", arguments='{"location":"Paris"}'),
            ),
        )

        stream = AGUIStream(agent)
        run_input = create_run_input(
            UserMessage(id="msg_1", content="What's the weather in Paris?"),
            tools=[get_weather_tool()],
        )

        events = await collect_events(stream, run_input)

        tool_calls = get_events_of_type(events, "TOOL_CALL_CHUNK")
        assert len(tool_calls) == 1
        assert tool_calls[0] == IsPartialDict({
            "toolCallName": "get_weather",
            "delta": IsStr(regex=r".*Paris.*"),
        })

        assert_event_type(events, "RUN_FINISHED")

    async def test_frontend_tool_with_result(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig(
                "The weather in Paris is sunny with 22°C.",
            ),
        )

        stream = AGUIStream(agent)

        # Request with tool result already included
        run_input = create_run_input(
            UserMessage(id="msg_1", content="What's the weather in Paris?"),
            AssistantMessage(
                id="msg_2",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(
                            name="get_weather",
                            arguments='{"location": "Paris"}',
                        ),
                    )
                ],
            ),
            ToolMessage(
                id="msg_3",
                content="Sunny, 22°C",
                tool_call_id="call_1",
            ),
            tools=[get_weather_tool()],
        )

        events = await collect_events(stream, run_input)

        text_message = assert_event_type(events, "TEXT_MESSAGE_CHUNK")
        assert text_message == IsPartialDict({
            "delta": IsStr(regex=r"(?i).*sunny.*|.*22.*"),
        })

    async def test_multiple_frontend_tools(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig([
                ToolCallEvent(name="get_weather", arguments='{"location":"Paris"}'),
                ToolCallEvent(name="get_weather", arguments='{"location":"London"}'),
            ]),
        )

        stream = AGUIStream(agent)
        run_input = create_run_input(
            UserMessage(id="msg_1", content="What's the weather in Paris and London?"),
            tools=[get_weather_tool()],
        )

        events = await collect_events(stream, run_input)

        tool_chunks = get_events_of_type(events, "TOOL_CALL_CHUNK")
        assert len(tool_chunks) == 2
        assert sorted(tool_chunks, key=lambda c: c["delta"]) == [
            IsPartialDict({
                "delta": IsStr(regex=r".*London.*"),
            }),
            IsPartialDict({
                "delta": IsStr(regex=r".*Paris.*"),
            }),
        ]


class TestMixedTools:
    async def test_backend_and_frontend_tools(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig([
                # Create a message with both backend and frontend tool calls
                ToolCallEvent(name="get_current_time"),
                ToolCallEvent(name="get_weather", arguments='{"location":"London"}'),
            ]),
        )

        @agent.tool
        def get_current_time() -> str:
            return "2024-01-15T10:30:00Z"

        stream = AGUIStream(agent)
        run_input = create_run_input(
            UserMessage(id="msg_1", content="What time is it and what's the weather in Paris?"),
            tools=[get_weather_tool()],
        )

        events = await collect_events(stream, run_input)

        backend_start = assert_event_type(events, "TOOL_CALL_START")
        assert backend_start == IsPartialDict({
            "toolCallName": "get_current_time",
        })

        frontend_chunk = assert_event_type(events, "TOOL_CALL_CHUNK")
        assert frontend_chunk == IsPartialDict({
            "toolCallName": "get_weather",
        })


class TestEventTypes:
    async def test_text_message_event_structure(self) -> None:
        agent = Agent("test_agent", config=TestConfig("Hello world!"))

        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="msg_1", content="Hi!"))

        events = await collect_events(stream, run_input)

        text_msg = assert_event_type(events, "TEXT_MESSAGE_CHUNK")
        assert text_msg == IsPartialDict({
            "messageId": IsStr(),
            "delta": "Hello world!",
            "timestamp": IsInt(),
        })

    async def test_tool_call_event_structure(self) -> None:
        agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="my_tool"), "Done"))

        @agent.tool
        def my_tool() -> str:
            return "result"

        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="msg_1", content="Call my_tool"))

        events = await collect_events(stream, run_input)

        tool_start = assert_event_type(events, "TOOL_CALL_START")
        assert tool_start == IsPartialDict({
            "toolCallId": IsStr(),
            "toolCallName": "my_tool",
            "timestamp": IsInt(),
        })

        tool_args = assert_event_type(events, "TOOL_CALL_ARGS")
        assert tool_args == IsPartialDict({
            "toolCallId": IsStr(),
            "delta": IsStr(),
            "timestamp": IsInt(),
        })

        tool_result = assert_event_type(events, "TOOL_CALL_RESULT")
        assert tool_result == IsPartialDict({
            "toolCallId": IsStr(),
            "content": IsStr(),
            "messageId": IsStr(),
            "timestamp": IsInt(),
        })

        tool_end = assert_event_type(events, "TOOL_CALL_END")
        assert tool_end == IsPartialDict({
            "toolCallId": IsStr(),
            "timestamp": IsInt(),
        })


class TestStateSnapshotEvent:
    async def test_initial_agent_variables_send_state_event(self, mock: MagicMock) -> None:
        agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="my_tool"), "Done"), variables={"var": "123"})

        @agent.tool
        def my_tool(var: Annotated[str, Variable()]) -> str:
            mock(var)
            return "result"

        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="msg_1", content="Hello!"))

        # Dispatch with context
        events = await collect_events(stream, run_input)

        tool_result = get_events_of_type(events, "STATE_SNAPSHOT")

        assert len(tool_result) == 1
        assert tool_result[0] == IsPartialDict({"timestamp": IsInt(), "snapshot": {"var": "123"}})

        mock.assert_called_once_with("123")

    async def test_agent_turn_variables_send_state_event(self, mock: MagicMock) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig(ToolCallEvent(name="my_tool"), "Done"),
        )

        @agent.tool
        def my_tool(var: Annotated[str, Variable()]) -> str:
            mock(var)
            return "result"

        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="msg_1", content="Hello!"))

        events = await collect_events(stream, run_input, variables={"var": "123"})

        tool_result = get_events_of_type(events, "STATE_SNAPSHOT")

        assert len(tool_result) == 1
        assert tool_result[0] == IsPartialDict({"timestamp": IsInt(), "snapshot": {"var": "123"}})

        mock.assert_called_once_with("123")

    async def test_frontend_variables_usage(self, mock: MagicMock) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig(ToolCallEvent(name="my_tool"), "Done"),
        )

        @agent.tool
        def my_tool(var: Annotated[str, Variable()]) -> str:
            mock(var)
            return "result"

        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="msg_1", content="Hello!"), state={"var": "123"})

        events = await collect_events(stream, run_input)

        assert_no_event_type(events, "STATE_SNAPSHOT")

        mock.assert_called_once_with("123")

    async def test_no_initial_state_snapshot_when_state_matches(self) -> None:
        agent = Agent("test_agent", config=TestConfig("Done"))
        stream = AGUIStream(agent)

        run_input = create_run_input(UserMessage(id="msg_1", content="Hello!"))

        events = await collect_events(stream, run_input)

        assert_no_event_type(events, "STATE_SNAPSHOT")

    async def test_state_snapshot_when_tool_returns_reply_result_with_context(self) -> None:
        agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="my_tool"), "Done"), variables={"var": "123"})

        @agent.tool
        def my_tool(var: Annotated[str, Variable()], ctx: Context) -> str:
            ctx.variables["var2"] = "1"
            ctx.variables["var3"] = "1234"
            return "result"

        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="msg_1", content="Hello!"))

        events = await collect_events(stream, run_input, variables={"var2": "1234"})

        tool_result = get_events_of_type(events, "STATE_SNAPSHOT")

        assert len(tool_result) == 2
        assert tool_result == [
            IsPartialDict({"snapshot": {"var": "123", "var2": "1234"}}),
            IsPartialDict({"snapshot": {"var": "123", "var2": "1", "var3": "1234"}}),
        ]


async def test_custom_event() -> None:
    agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="my_tool"), "Done"))

    @agent.tool
    async def my_tool(ctx: Context) -> None:
        await ctx.send(AGUIEvent(CustomEvent(name="test", value=123)))

    stream = AGUIStream(agent)
    run_input = create_run_input(UserMessage(id="msg_1", content="Hello!"))

    events = await collect_events(stream, run_input)

    tool_result = assert_event_type(events, "CUSTOM")

    assert tool_result == IsPartialDict({
        "name": "test",
        "value": 123,
    })
