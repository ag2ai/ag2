# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Annotated

import pytest
from ag_ui.core import (
    AssistantMessage,
    FunctionCall,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from dirty_equals import IsInt, IsPartialDict, IsStr

from autogen import ConversableAgent, LLMConfig
from autogen.ag_ui import AGUIStream
from autogen.agentchat import ContextVariables, ReplyResult
from autogen.testing import TestAgent
from autogen.testing import ToolCall as TestToolCall
from test.ag_ui.utils import (
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
        agent = ConversableAgent("test_agent")

        with TestAgent(agent, ["Hello! I'm doing well, thank you for asking."]):
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
        agent = ConversableAgent("test_agent")

        with TestAgent(agent, ["I see you've been talking about weather. It's sunny today!"]):
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

    async def test_system_message_included(self) -> None:
        agent = ConversableAgent("test_agent")

        with TestAgent(agent, ["Response with system context."]):
            stream = AGUIStream(agent)
            run_input = create_run_input(
                SystemMessage(id="msg_1", content="You are a helpful assistant."),
                UserMessage(id="msg_2", content="Hello!"),
            )

            events = await collect_events(stream, run_input)

        assert_event_type(events, "RUN_STARTED")
        text_msg = assert_event_type(events, "TEXT_MESSAGE_CHUNK")
        assert text_msg == IsPartialDict({
            "delta": "Response with system context.",
        })
        assert_event_type(events, "RUN_FINISHED")


class TestBackendTools:
    async def test_backend_tool_call_and_result(self) -> None:
        agent = ConversableAgent(
            "test_agent",
            llm_config=LLMConfig({
                "model": "gpt-4o-mini",
                "api_key": "test-key",
            }),
        )

        # Register a local tool
        @agent.register_for_execution()
        @agent.register_for_llm()
        def get_current_time() -> str:
            return "2024-01-15T10:30:00Z"

        with TestAgent(
            agent,
            [
                TestToolCall("get_current_time").to_message(),
                "The current time is 2024-01-15T10:30:00Z",
            ],
        ):
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
        agent = ConversableAgent(
            "test_agent",
            llm_config=LLMConfig({
                "model": "gpt-4o-mini",
                "api_key": "test-key",
            }),
        )

        @agent.register_for_execution()
        @agent.register_for_llm()
        def calculate_sum(a: Annotated[int, "First number"], b: Annotated[int, "Second number"]) -> int:
            return a + b

        with TestAgent(
            agent,
            [
                TestToolCall("calculate_sum", a=5, b=3).to_message(),
                "The sum of 5 and 3 is 8.",
            ],
        ):
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
        agent = ConversableAgent(
            "test_agent",
            llm_config=LLMConfig({
                "model": "gpt-4o-mini",
                "api_key": "test-key",
            }),
        )

        @agent.register_for_execution()
        @agent.register_for_llm()
        def tool_a() -> str:
            return "Result A"

        @agent.register_for_execution()
        @agent.register_for_llm()
        def tool_b() -> str:
            return "Result B"

        with TestAgent(
            agent,
            [
                TestToolCall("tool_a").to_message(),
                TestToolCall("tool_b").to_message(),
                "Both tools executed successfully.",
            ],
        ):
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
        agent = ConversableAgent(
            "test_agent",
            llm_config=LLMConfig({
                "model": "gpt-4o-mini",
                "api_key": "test-key",
            }),
        )

        with TestAgent(
            agent,
            [
                TestToolCall("get_weather", location="Paris").to_message(),
            ],
        ):
            stream = AGUIStream(agent)
            run_input = create_run_input(
                UserMessage(id="msg_1", content="What's the weather in Paris?"),
                tools=[get_weather_tool()],
            )

            events = await collect_events(stream, run_input)

        assert_event_type(events, "RUN_STARTED")

        tool_chunk = assert_event_type(events, "TOOL_CALL_CHUNK")
        assert tool_chunk == IsPartialDict({
            "toolCallName": "get_weather",
            "delta": IsStr(regex=r".*Paris.*"),
        })

        assert_event_type(events, "RUN_FINISHED")

    async def test_frontend_tool_with_result(self) -> None:
        agent = ConversableAgent(
            "test_agent",
            llm_config=LLMConfig({
                "model": "gpt-4o-mini",
                "api_key": "test-key",
            }),
        )

        with TestAgent(
            agent,
            [
                # Second call after tool result provided
                "The weather in Paris is sunny with 22°C.",
            ],
        ):
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
        agent = ConversableAgent(
            "test_agent",
            llm_config=LLMConfig({
                "model": "gpt-4o-mini",
                "api_key": "test-key",
            }),
        )

        # Create a message with multiple tool calls manually
        multi_tool_message = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "London"}',
                    },
                },
            ],
        }

        with TestAgent(agent, [multi_tool_message]):
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
        agent = ConversableAgent(
            "test_agent",
            llm_config=LLMConfig({
                "model": "gpt-4o-mini",
                "api_key": "test-key",
            }),
        )

        # Register a backend tool
        @agent.register_for_execution()
        @agent.register_for_llm()
        def get_current_time() -> str:
            return "2024-01-15T10:30:00Z"

        # Create a message with both backend and frontend tool calls
        mixed_tool_message = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_current_time",
                        "arguments": "{}",
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                },
            ],
        }

        with TestAgent(agent, [mixed_tool_message]):
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
        agent = ConversableAgent("test_agent")

        with TestAgent(agent, ["Hello world!"]):
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
        agent = ConversableAgent(
            "test_agent",
            llm_config=LLMConfig({
                "model": "gpt-4o-mini",
                "api_key": "test-key",
            }),
        )

        @agent.register_for_execution()
        @agent.register_for_llm()
        def my_tool() -> str:
            return "result"

        with TestAgent(
            agent,
            [
                TestToolCall("my_tool").to_message(),
                "Done",
            ],
        ):
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


class TestContextHandling:
    async def test_context_passed_to_stream(self) -> None:
        agent = ConversableAgent("test_agent")

        with TestAgent(agent, ["Response"]):
            stream = AGUIStream(agent)
            run_input = create_run_input(UserMessage(id="msg_1", content="Hello!"))

            # Dispatch with context
            events = []
            async for event in stream.dispatch(
                run_input,
                context={
                    "user_id": "123",
                    "session": "abc",
                },
            ):
                event_str = event.removeprefix("data: ").strip()
                if event_str:
                    events.append(json.loads(event_str))

        assert_event_type(events, "RUN_STARTED")
        assert_event_type(events, "RUN_FINISHED")


class TestStateSnapshotEvent:
    async def test_initial_state_snapshot_when_agent_has_context_variables(self) -> None:
        agent = ConversableAgent(
            "test_agent",
            context_variables=ContextVariables({"proverbs": ["AG2 the best choice to build your agent."]}),
        )

        with TestAgent(agent, ["Response using context."]):
            stream = AGUIStream(agent)
            run_input = create_run_input(
                UserMessage(id="msg_1", content="Hello!"),
                state=None,
            )

            events = await collect_events(stream, run_input)

        run_started = assert_event_type(events, "RUN_STARTED")
        assert run_started == IsPartialDict({
            "threadId": run_input.thread_id,
            "runId": run_input.run_id,
            "timestamp": IsInt(),
        })

        state_snapshot = assert_event_type(events, "STATE_SNAPSHOT")
        assert state_snapshot == IsPartialDict({
            "snapshot": {"proverbs": ["AG2 the best choice to build your agent."]},
            "timestamp": IsInt(),
        })

        assert_event_type(events, "TEXT_MESSAGE_CHUNK")
        assert_event_type(events, "RUN_FINISHED")

    async def test_no_initial_state_snapshot_when_state_matches(self) -> None:
        agent = ConversableAgent("test_agent")

        with TestAgent(agent, ["Response."]):
            stream = AGUIStream(agent)
            run_input = create_run_input(
                UserMessage(id="msg_1", content="Hello!"),
                state={},
            )

            events = await collect_events(stream, run_input)

        assert_event_type(events, "RUN_STARTED")
        assert_no_event_type(events, "STATE_SNAPSHOT")
        assert_event_type(events, "TEXT_MESSAGE_CHUNK")
        assert_event_type(events, "RUN_FINISHED")

    async def test_state_snapshot_when_tool_returns_reply_result_with_context(self) -> None:
        agent = ConversableAgent(
            "test_agent",
            llm_config=LLMConfig({
                "model": "gpt-4o-mini",
                "api_key": "test-key",
            }),
            context_variables=ContextVariables({"proverbs": ["Initial proverb."]}),
        )

        @agent.register_for_execution()
        @agent.register_for_llm()
        def get_proverbs(context_variables: ContextVariables) -> ReplyResult:
            """Get the current list of proverbs."""
            proverbs = context_variables.get("proverbs", [])
            return ReplyResult(
                message=", ".join(proverbs),
                context_variables=context_variables,
            )

        with TestAgent(
            agent,
            [
                TestToolCall("get_proverbs").to_message(),
                "The proverbs are: Initial proverb.",
            ],
        ):
            stream = AGUIStream(agent)
            run_input = create_run_input(
                UserMessage(id="msg_1", content="What are the proverbs?"),
                state={"proverbs": ["Initial proverb."]},
            )

            events = await collect_events(stream, run_input)

        assert_event_type(events, "RUN_STARTED")

        assert_event_type(events, "TOOL_CALL_START")
        assert_event_type(events, "TOOL_CALL_ARGS")
        assert_event_type(events, "TOOL_CALL_RESULT")
        assert_event_type(events, "TOOL_CALL_END")

        state_snapshots = get_events_of_type(events, "STATE_SNAPSHOT")
        assert len(state_snapshots) >= 1
        snapshot_with_proverbs = next(
            (e for e in state_snapshots if e.get("snapshot", {}).get("proverbs") == ["Initial proverb."]),
            None,
        )
        assert snapshot_with_proverbs is not None
        assert "timestamp" in snapshot_with_proverbs

        assert_event_type(events, "TEXT_MESSAGE_CHUNK")
        assert_event_type(events, "RUN_FINISHED")
