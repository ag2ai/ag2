# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
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
    ImageInputContent,
    InputContentUrlSource,
    TextInputContent,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from dirty_equals import IsInt, IsPartialDict, IsStr

from ag2 import Agent, Context, Variable
from ag2.ag_ui import AGUIEvent, AGUIStream
from ag2.events import ModelRequest, TextInput, ToolCallEvent, UrlInput
from ag2.testing import TestConfig, TrackingConfig
from test.ag_ui.harness import dispatch_run, every, only, run_input, weather_tool

pytestmark = pytest.mark.asyncio


class TestBasicConversation:
    async def test_basic_user_message(self) -> None:
        agent = Agent("test_agent", config=TestConfig("Hello! I'm doing well, thank you for asking."))

        stream = AGUIStream(agent)
        incoming = run_input(UserMessage(id="msg_1", content="Hello, how are you?"))

        events = await dispatch_run(stream, incoming)

        run_started = only(events, "RUN_STARTED")
        assert run_started == IsPartialDict({
            "threadId": incoming.thread_id,
            "runId": incoming.run_id,
            "timestamp": IsInt(),
        })

        text_message = only(events, "TEXT_MESSAGE_CHUNK")
        assert text_message == IsPartialDict({
            "delta": "Hello! I'm doing well, thank you for asking.",
            "timestamp": IsInt(),
        })

        run_finished = only(events, "RUN_FINISHED")
        assert run_finished == IsPartialDict({
            "threadId": incoming.thread_id,
            "runId": incoming.run_id,
            "timestamp": IsInt(),
        })

    async def test_multiple_messages_history(self) -> None:
        agent = Agent("test_agent", config=TestConfig("I see you've been talking about weather. It's sunny today!"))

        stream = AGUIStream(agent)

        incoming = run_input(
            UserMessage(id="msg_1", content="What's the weather like?"),
            AssistantMessage(id="msg_2", content="I'll check the weather for you."),
            UserMessage(id="msg_3", content="Thanks! And tomorrow?"),
        )

        events = await dispatch_run(stream, incoming)

        only(events, "RUN_STARTED")
        only(events, "TEXT_MESSAGE_CHUNK")
        only(events, "RUN_FINISHED")


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
        incoming = run_input(UserMessage(id="msg_1", content="What time is it?"))

        events = await dispatch_run(stream, incoming)

        only(events, "RUN_STARTED")

        tool_start = only(events, "TOOL_CALL_START")
        assert tool_start == IsPartialDict({
            "toolCallName": "get_current_time",
        })

        tool_args = only(events, "TOOL_CALL_ARGS")
        assert tool_args == IsPartialDict({
            "delta": "{}",
        })

        tool_result = only(events, "TOOL_CALL_RESULT")
        assert tool_result == IsPartialDict({
            "content": IsStr(regex=r".*2024-01-15T10:30:00Z.*"),
        })

        only(events, "TOOL_CALL_END")

        text_message = only(events, "TEXT_MESSAGE_CHUNK")
        assert text_message == IsPartialDict({
            "delta": IsStr(regex=r".*2024-01-15T10:30:00Z.*"),
        })

        only(events, "RUN_FINISHED")

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
        incoming = run_input(UserMessage(id="msg_1", content="What is 5 + 3?"))

        events = await dispatch_run(stream, incoming)

        tool_start = only(events, "TOOL_CALL_START")
        assert tool_start == IsPartialDict({
            "toolCallName": "calculate_sum",
        })

        tool_args = only(events, "TOOL_CALL_ARGS")
        args = json.loads(tool_args["delta"])
        assert args == IsPartialDict({
            "a": 5,
            "b": 3,
        })

        tool_result = only(events, "TOOL_CALL_RESULT")
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
        incoming = run_input(UserMessage(id="msg_1", content="Call both tools"))

        events = await dispatch_run(stream, incoming)

        tool_starts = every(events, "TOOL_CALL_START")
        assert len(tool_starts) == 2
        assert sorted(tool_starts, key=lambda e: e["toolCallName"]) == [
            IsPartialDict({
                "toolCallName": "tool_a",
            }),
            IsPartialDict({
                "toolCallName": "tool_b",
            }),
        ]

        tool_results = every(events, "TOOL_CALL_RESULT")
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
        incoming = run_input(
            UserMessage(id="msg_1", content="What's the weather in Paris?"),
            tools=[weather_tool()],
        )

        events = await dispatch_run(stream, incoming)

        tool_calls = every(events, "TOOL_CALL_CHUNK")
        assert len(tool_calls) == 1
        assert tool_calls[0] == IsPartialDict({
            "toolCallName": "get_weather",
            "delta": IsStr(regex=r".*Paris.*"),
        })

        only(events, "RUN_FINISHED")

    async def test_frontend_tool_with_result(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig(
                "The weather in Paris is sunny with 22°C.",
            ),
        )

        stream = AGUIStream(agent)

        # Request with tool result already included
        incoming = run_input(
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
            tools=[weather_tool()],
        )

        events = await dispatch_run(stream, incoming)

        text_message = only(events, "TEXT_MESSAGE_CHUNK")
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
        incoming = run_input(
            UserMessage(id="msg_1", content="What's the weather in Paris and London?"),
            tools=[weather_tool()],
        )

        events = await dispatch_run(stream, incoming)

        tool_chunks = every(events, "TOOL_CALL_CHUNK")
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
        incoming = run_input(
            UserMessage(id="msg_1", content="What time is it and what's the weather in Paris?"),
            tools=[weather_tool()],
        )

        events = await dispatch_run(stream, incoming)

        backend_start = only(events, "TOOL_CALL_START")
        assert backend_start == IsPartialDict({
            "toolCallName": "get_current_time",
        })

        frontend_chunk = only(events, "TOOL_CALL_CHUNK")
        assert frontend_chunk == IsPartialDict({
            "toolCallName": "get_weather",
        })


class TestEventTypes:
    async def test_text_message_event_structure(self) -> None:
        agent = Agent("test_agent", config=TestConfig("Hello world!"))

        stream = AGUIStream(agent)
        incoming = run_input(UserMessage(id="msg_1", content="Hi!"))

        events = await dispatch_run(stream, incoming)

        text_msg = only(events, "TEXT_MESSAGE_CHUNK")
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
        incoming = run_input(UserMessage(id="msg_1", content="Call my_tool"))

        events = await dispatch_run(stream, incoming)

        tool_start = only(events, "TOOL_CALL_START")
        assert tool_start == IsPartialDict({
            "toolCallId": IsStr(),
            "toolCallName": "my_tool",
            "timestamp": IsInt(),
        })

        tool_args = only(events, "TOOL_CALL_ARGS")
        assert tool_args == IsPartialDict({
            "toolCallId": IsStr(),
            "delta": IsStr(),
            "timestamp": IsInt(),
        })

        tool_result = only(events, "TOOL_CALL_RESULT")
        assert tool_result == IsPartialDict({
            "toolCallId": IsStr(),
            "content": IsStr(),
            "messageId": IsStr(),
            "timestamp": IsInt(),
        })

        tool_end = only(events, "TOOL_CALL_END")
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
        incoming = run_input(UserMessage(id="msg_1", content="Hello!"))

        # Dispatch with context
        events = await dispatch_run(stream, incoming)

        tool_result = every(events, "STATE_SNAPSHOT")

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
        incoming = run_input(UserMessage(id="msg_1", content="Hello!"))

        events = await dispatch_run(stream, incoming, variables={"var": "123"})

        tool_result = every(events, "STATE_SNAPSHOT")

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
        incoming = run_input(UserMessage(id="msg_1", content="Hello!"), state={"var": "123"})

        events = await dispatch_run(stream, incoming)

        assert every(events, "STATE_SNAPSHOT") == []

        mock.assert_called_once_with("123")

    async def test_no_initial_state_snapshot_when_state_matches(self) -> None:
        agent = Agent("test_agent", config=TestConfig("Done"))
        stream = AGUIStream(agent)

        incoming = run_input(UserMessage(id="msg_1", content="Hello!"))

        events = await dispatch_run(stream, incoming)

        assert every(events, "STATE_SNAPSHOT") == []

    async def test_state_snapshot_when_tool_returns_reply_result_with_context(self) -> None:
        agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="my_tool"), "Done"), variables={"var": "123"})

        @agent.tool
        def my_tool(var: Annotated[str, Variable()], ctx: Context) -> str:
            ctx.variables["var2"] = "1"
            ctx.variables["var3"] = "1234"
            return "result"

        stream = AGUIStream(agent)
        incoming = run_input(UserMessage(id="msg_1", content="Hello!"))

        events = await dispatch_run(stream, incoming, variables={"var2": "1234"})

        tool_result = every(events, "STATE_SNAPSHOT")

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
    incoming = run_input(UserMessage(id="msg_1", content="Hello!"))

    events = await dispatch_run(stream, incoming)

    tool_result = only(events, "CUSTOM")

    assert tool_result == IsPartialDict({
        "name": "test",
        "value": 123,
    })


async def test_the_current_turn_reaches_the_llm_as_the_message_it_was_mapped_to() -> None:
    """The mapping itself is `test_mapper.py`'s; this is that it is wired up at all.

    The current turn is handed to the LLM as `messages[-1]`, so a client's
    multimodal content is what the model is actually asked about rather than
    something the transport decoded and dropped.
    """
    tracking = TrackingConfig(TestConfig("A cat."))
    agent = Agent("test_agent", config=tracking)

    await dispatch_run(
        AGUIStream(agent),
        run_input(
            UserMessage(
                id="msg_1",
                content=[
                    TextInputContent(text="describe this"),
                    ImageInputContent(source=InputContentUrlSource(value="https://x/img.png")),
                ],
            )
        ),
    )

    [(last_message,)] = [call.args for call in tracking.mock.call_args_list]
    assert last_message == ModelRequest([
        TextInput("describe this"),
        UrlInput(kind="image", url="https://x/img.png"),
    ])
