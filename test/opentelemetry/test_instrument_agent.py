# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for autogen.opentelemetry.instrumentators.agent module (instrument_agent)."""

import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from autogen import ConversableAgent
from autogen.opentelemetry import instrument_agent
from autogen.opentelemetry.consts import SpanType
from autogen.testing import TestAgent
from test.opentelemetry.conftest import InMemorySpanExporter


@pytest.fixture()
def otel_setup():
    """Create an in-memory OTEL exporter/provider for capturing spans."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider


# ---------------------------------------------------------------------------
# Basic instrumentation
# ---------------------------------------------------------------------------
class TestInstrumentAgentBasic:
    """Basic tests for instrument_agent."""

    def test_returns_same_agent(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("test_agent", llm_config=False, human_input_mode="NEVER")
        result = instrument_agent(agent, tracer_provider=provider)
        assert result is agent

    def test_wraps_initiate_chat(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("test_agent", llm_config=False, human_input_mode="NEVER")
        instrument_agent(agent, tracer_provider=provider)
        assert hasattr(agent.initiate_chat, "__otel_wrapped__")

    def test_wraps_generate_reply(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("test_agent", llm_config=False, human_input_mode="NEVER")
        instrument_agent(agent, tracer_provider=provider)
        assert hasattr(agent.generate_reply, "__otel_wrapped__")

    def test_wraps_a_generate_reply(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("test_agent", llm_config=False, human_input_mode="NEVER")
        instrument_agent(agent, tracer_provider=provider)
        assert hasattr(agent.a_generate_reply, "__otel_wrapped__")

    def test_wraps_execute_function(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("test_agent", llm_config=False, human_input_mode="NEVER")
        instrument_agent(agent, tracer_provider=provider)
        assert hasattr(agent.execute_function, "__otel_wrapped__")

    def test_wraps_get_human_input(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("test_agent", llm_config=False, human_input_mode="NEVER")
        instrument_agent(agent, tracer_provider=provider)
        assert hasattr(agent.get_human_input, "__otel_wrapped__")


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------
class TestInstrumentAgentIdempotency:
    """Tests that double-instrumenting does not double-wrap."""

    def test_double_instrument_does_not_double_wrap_initiate_chat(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("test_agent", llm_config=False, human_input_mode="NEVER")
        instrument_agent(agent, tracer_provider=provider)
        first_initiate_chat = agent.initiate_chat
        instrument_agent(agent, tracer_provider=provider)
        assert agent.initiate_chat is first_initiate_chat

    def test_double_instrument_does_not_double_wrap_generate_reply(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("test_agent", llm_config=False, human_input_mode="NEVER")
        instrument_agent(agent, tracer_provider=provider)
        first_generate_reply = agent.generate_reply
        instrument_agent(agent, tracer_provider=provider)
        assert agent.generate_reply is first_generate_reply

    def test_double_instrument_does_not_double_wrap_execute_function(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("test_agent", llm_config=False, human_input_mode="NEVER")
        instrument_agent(agent, tracer_provider=provider)
        first_execute_function = agent.execute_function
        instrument_agent(agent, tracer_provider=provider)
        assert agent.execute_function is first_execute_function


# ---------------------------------------------------------------------------
# Conversation span (initiate_chat)
# ---------------------------------------------------------------------------
class TestConversationSpan:
    """Tests that initiate_chat creates proper conversation spans."""

    def test_initiate_chat_creates_conversation_span(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["Hello back!"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        assert len(conversation_spans) >= 1

        conv_span = conversation_spans[0]
        assert conv_span.name == "conversation sender"
        assert conv_span.attributes["gen_ai.operation.name"] == "conversation"
        assert conv_span.attributes["gen_ai.agent.name"] == "sender"

    def test_initiate_chat_records_max_turns(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["response"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        conv_span = conversation_spans[0]
        assert conv_span.attributes.get("gen_ai.conversation.max_turns") == 1

    def test_initiate_chat_records_input_message(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["response"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message="Hello world", max_turns=1)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        conv_span = conversation_spans[0]
        input_messages = json.loads(conv_span.attributes.get("gen_ai.input.messages", "[]"))
        assert len(input_messages) >= 1
        # The input message should contain our "Hello world" text
        found = False
        for msg in input_messages:
            for part in msg.get("parts", []):
                if part.get("content") == "Hello world":
                    found = True
        assert found, f"Expected 'Hello world' in input messages, got: {input_messages}"

    def test_initiate_chat_records_chat_history_output(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["I got your message"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        conv_span = conversation_spans[0]
        output_messages = json.loads(conv_span.attributes.get("gen_ai.output.messages", "[]"))
        assert len(output_messages) >= 1

    def test_initiate_chat_records_conversation_turns(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["response"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        conv_span = conversation_spans[0]
        assert "gen_ai.conversation.turns" in conv_span.attributes
        assert conv_span.attributes["gen_ai.conversation.turns"] >= 1


# ---------------------------------------------------------------------------
# Agent (invoke_agent) spans
# ---------------------------------------------------------------------------
class TestAgentInvokeSpan:
    """Tests that generate_reply creates proper agent spans."""

    def test_generate_reply_creates_agent_span(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["Hello back"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        agent_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.AGENT.value]
        # There should be at least one invoke_agent span (for the recipient generating a reply)
        assert len(agent_spans) >= 1

    def test_agent_span_attributes(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("my_assistant", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["test response"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        agent_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.AGENT.value]
        # Find the recipient's invoke_agent span
        recipient_spans = [s for s in agent_spans if s.attributes.get("gen_ai.agent.name") == "my_assistant"]
        assert len(recipient_spans) >= 1
        span = recipient_spans[0]
        assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
        assert "invoke_agent" in span.name

    def test_agent_span_captures_input_messages(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["reply"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message="Test input", max_turns=1)

        spans = exporter.get_finished_spans()
        agent_spans = [
            s
            for s in spans
            if s.attributes.get("ag2.span.type") == SpanType.AGENT.value
            and s.attributes.get("gen_ai.agent.name") == "recipient"
        ]
        assert len(agent_spans) >= 1
        span = agent_spans[0]
        assert "gen_ai.input.messages" in span.attributes

    def test_agent_span_captures_output_messages(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["my reply"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        agent_spans = [
            s
            for s in spans
            if s.attributes.get("ag2.span.type") == SpanType.AGENT.value
            and s.attributes.get("gen_ai.agent.name") == "recipient"
        ]
        assert len(agent_spans) >= 1
        span = agent_spans[0]
        assert "gen_ai.output.messages" in span.attributes


# ---------------------------------------------------------------------------
# Span hierarchy
# ---------------------------------------------------------------------------
class TestSpanHierarchy:
    """Tests that spans form the correct parent-child hierarchy."""

    def test_agent_spans_are_children_of_conversation(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["response"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        agent_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.AGENT.value]

        assert len(conversation_spans) >= 1
        assert len(agent_spans) >= 1

        conv_span = conversation_spans[0]
        # Agent spans should share the same trace_id as the conversation span
        for a_span in agent_spans:
            assert a_span.context.trace_id == conv_span.context.trace_id

    def test_all_spans_share_same_trace_id(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["response"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        if len(spans) > 1:
            trace_ids = {s.context.trace_id for s in spans}
            assert len(trace_ids) == 1, f"Expected single trace, got {len(trace_ids)} traces"


# ---------------------------------------------------------------------------
# Multiple turn conversation
# ---------------------------------------------------------------------------
class TestMultipleTurns:
    """Tests for multi-turn conversations."""

    def test_two_turn_conversation(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(agent, ["follow-up question"]), TestAgent(recipient, ["first reply", "second reply"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message="Hello", max_turns=2)

        spans = exporter.get_finished_spans()
        agent_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.AGENT.value]
        # With 2 turns, we expect multiple invoke_agent spans
        assert len(agent_spans) >= 2


# ---------------------------------------------------------------------------
# Tool execution span
# ---------------------------------------------------------------------------
class TestToolExecutionSpan:
    """Tests that execute_function creates proper tool spans."""

    def test_execute_function_creates_tool_span(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("tool_agent", llm_config=False, human_input_mode="NEVER")

        def my_tool(x: int) -> str:
            return f"result: {x}"

        agent.register_function({"my_tool": my_tool})
        instrument_agent(agent, tracer_provider=provider)

        func_call = {"name": "my_tool", "arguments": json.dumps({"x": 42})}
        is_success, result = agent.execute_function(func_call)

        assert is_success

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.TOOL.value]
        assert len(tool_spans) == 1

        tool_span = tool_spans[0]
        assert tool_span.attributes["gen_ai.operation.name"] == "execute_tool"
        assert tool_span.attributes["gen_ai.tool.name"] == "my_tool"
        assert tool_span.attributes["gen_ai.tool.type"] == "function"
        assert "execute_tool my_tool" in tool_span.name

    def test_execute_function_records_arguments(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("tool_agent", llm_config=False, human_input_mode="NEVER")

        def my_tool(x: int) -> str:
            return f"result: {x}"

        agent.register_function({"my_tool": my_tool})
        instrument_agent(agent, tracer_provider=provider)

        func_call = {"name": "my_tool", "arguments": json.dumps({"x": 42})}
        agent.execute_function(func_call)

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.TOOL.value]
        tool_span = tool_spans[0]
        assert "gen_ai.tool.call.arguments" in tool_span.attributes

    def test_execute_function_records_call_id(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("tool_agent", llm_config=False, human_input_mode="NEVER")

        def my_tool() -> str:
            return "ok"

        agent.register_function({"my_tool": my_tool})
        instrument_agent(agent, tracer_provider=provider)

        func_call = {"name": "my_tool", "arguments": "{}"}
        agent.execute_function(func_call, call_id="call_abc123")

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.TOOL.value]
        tool_span = tool_spans[0]
        assert tool_span.attributes["gen_ai.tool.call.id"] == "call_abc123"

    def test_execute_function_records_result_on_success(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("tool_agent", llm_config=False, human_input_mode="NEVER")

        def my_tool() -> str:
            return "success_value"

        agent.register_function({"my_tool": my_tool})
        instrument_agent(agent, tracer_provider=provider)

        func_call = {"name": "my_tool", "arguments": "{}"}
        is_success, result = agent.execute_function(func_call)
        assert is_success

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.TOOL.value]
        tool_span = tool_spans[0]
        assert "gen_ai.tool.call.result" in tool_span.attributes

    def test_execute_function_records_error_on_failure(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("tool_agent", llm_config=False, human_input_mode="NEVER")

        def failing_tool() -> str:
            raise ValueError("something went wrong")

        agent.register_function({"failing_tool": failing_tool})
        instrument_agent(agent, tracer_provider=provider)

        func_call = {"name": "failing_tool", "arguments": "{}"}
        is_success, result = agent.execute_function(func_call)
        assert not is_success

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.TOOL.value]
        tool_span = tool_spans[0]
        assert tool_span.attributes.get("error.type") == "ExecutionError"


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------
class TestAsyncInstrumentAgent:
    """Async tests for instrument_agent."""

    @pytest.mark.asyncio
    async def test_a_initiate_chat_creates_conversation_span(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["Hello back!"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            await agent.a_initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        assert len(conversation_spans) >= 1
        conv_span = conversation_spans[0]
        assert conv_span.attributes["gen_ai.operation.name"] == "conversation"
        assert conv_span.attributes["gen_ai.agent.name"] == "sender"

    @pytest.mark.asyncio
    async def test_a_generate_reply_creates_agent_span(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["response"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            await agent.a_initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        agent_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.AGENT.value]
        assert len(agent_spans) >= 1

    @pytest.mark.asyncio
    async def test_async_all_spans_share_same_trace_id(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["response"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            await agent.a_initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        if len(spans) > 1:
            trace_ids = {s.context.trace_id for s in spans}
            assert len(trace_ids) == 1


# ---------------------------------------------------------------------------
# Tool execution with dict arguments (non-string branch)
# ---------------------------------------------------------------------------
class TestToolExecutionDictArguments:
    """Tests that execute_function handles dict arguments (non-string) correctly in the tracing layer."""

    def test_execute_function_with_dict_arguments_records_span(self, otel_setup) -> None:
        """When arguments is a dict (not a JSON string), the tracing layer should serialize via json.dumps.

        Note: The agent's internal execute_function uses json.loads on the arguments,
        which will fail if arguments is a dict. But the tracing wrapper records the
        arguments before delegating, so we verify the span attribute is set correctly
        regardless of the agent's internal handling.
        """
        exporter, provider = otel_setup
        agent = ConversableAgent("tool_agent", llm_config=False, human_input_mode="NEVER")

        def my_tool(x: int) -> str:
            return f"result: {x}"

        agent.register_function({"my_tool": my_tool})
        instrument_agent(agent, tracer_provider=provider)

        # Pass arguments as a dict instead of a JSON string
        func_call = {"name": "my_tool", "arguments": {"x": 42}}
        # The agent may fail internally (json parsing), but the span should still be created
        _is_success, _result = agent.execute_function(func_call)

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.TOOL.value]
        assert len(tool_spans) == 1

        tool_span = tool_spans[0]
        # The dict arguments should have been JSON-serialized by the tracing layer
        recorded_args = tool_span.attributes["gen_ai.tool.call.arguments"]
        assert json.loads(recorded_args) == {"x": 42}


# ---------------------------------------------------------------------------
# Async tool execution span (a_execute_function)
# ---------------------------------------------------------------------------
class TestAsyncToolExecutionSpan:
    """Tests that a_execute_function creates proper tool spans."""

    @pytest.mark.asyncio
    async def test_a_execute_function_creates_tool_span(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("tool_agent", llm_config=False, human_input_mode="NEVER")

        async def my_async_tool(x: int) -> str:
            return f"result: {x}"

        agent.register_function({"my_async_tool": my_async_tool})
        instrument_agent(agent, tracer_provider=provider)

        func_call = {"name": "my_async_tool", "arguments": json.dumps({"x": 42})}
        is_success, result = await agent.a_execute_function(func_call)

        assert is_success

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.TOOL.value]
        assert len(tool_spans) == 1

        tool_span = tool_spans[0]
        assert tool_span.attributes["gen_ai.operation.name"] == "execute_tool"
        assert tool_span.attributes["gen_ai.tool.name"] == "my_async_tool"
        assert tool_span.attributes["gen_ai.tool.type"] == "function"
        assert "execute_tool my_async_tool" in tool_span.name

    @pytest.mark.asyncio
    async def test_a_execute_function_records_arguments(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("tool_agent", llm_config=False, human_input_mode="NEVER")

        async def my_async_tool(x: int, y: str) -> str:
            return f"{y}: {x}"

        agent.register_function({"my_async_tool": my_async_tool})
        instrument_agent(agent, tracer_provider=provider)

        args = {"x": 99, "y": "hello"}
        func_call = {"name": "my_async_tool", "arguments": json.dumps(args)}
        await agent.a_execute_function(func_call)

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.TOOL.value]
        tool_span = tool_spans[0]
        assert "gen_ai.tool.call.arguments" in tool_span.attributes
        recorded_args = json.loads(tool_span.attributes["gen_ai.tool.call.arguments"])
        assert recorded_args == args

    @pytest.mark.asyncio
    async def test_a_execute_function_records_call_id(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("tool_agent", llm_config=False, human_input_mode="NEVER")

        async def my_async_tool() -> str:
            return "ok"

        agent.register_function({"my_async_tool": my_async_tool})
        instrument_agent(agent, tracer_provider=provider)

        func_call = {"name": "my_async_tool", "arguments": "{}"}
        await agent.a_execute_function(func_call, call_id="call_async_123")

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.TOOL.value]
        tool_span = tool_spans[0]
        assert tool_span.attributes["gen_ai.tool.call.id"] == "call_async_123"

    @pytest.mark.asyncio
    async def test_a_execute_function_records_result_on_success(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("tool_agent", llm_config=False, human_input_mode="NEVER")

        async def my_async_tool() -> str:
            return "async_success_value"

        agent.register_function({"my_async_tool": my_async_tool})
        instrument_agent(agent, tracer_provider=provider)

        func_call = {"name": "my_async_tool", "arguments": "{}"}
        is_success, result = await agent.a_execute_function(func_call)
        assert is_success

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.TOOL.value]
        tool_span = tool_spans[0]
        assert "gen_ai.tool.call.result" in tool_span.attributes

    @pytest.mark.asyncio
    async def test_a_execute_function_error(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("tool_agent", llm_config=False, human_input_mode="NEVER")

        async def failing_async_tool() -> str:
            raise ValueError("async error occurred")

        agent.register_function({"failing_async_tool": failing_async_tool})
        instrument_agent(agent, tracer_provider=provider)

        func_call = {"name": "failing_async_tool", "arguments": "{}"}
        is_success, result = await agent.a_execute_function(func_call)
        assert not is_success

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.TOOL.value]
        tool_span = tool_spans[0]
        assert tool_span.attributes.get("error.type") == "ExecutionError"

    @pytest.mark.asyncio
    async def test_a_execute_function_with_dict_arguments(self, otel_setup) -> None:
        """When arguments is a dict (not a JSON string), the tracing layer should serialize via json.dumps.

        The agent's internal a_execute_function may fail parsing dict arguments,
        but the tracing wrapper records them before delegating.
        """
        exporter, provider = otel_setup
        agent = ConversableAgent("tool_agent", llm_config=False, human_input_mode="NEVER")

        async def my_async_tool(x: int) -> str:
            return f"result: {x}"

        agent.register_function({"my_async_tool": my_async_tool})
        instrument_agent(agent, tracer_provider=provider)

        # Pass arguments as a dict instead of a JSON string
        func_call = {"name": "my_async_tool", "arguments": {"x": 7}}
        _is_success, _result = await agent.a_execute_function(func_call)

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.TOOL.value]
        assert len(tool_spans) == 1

        tool_span = tool_spans[0]
        recorded_args = tool_span.attributes["gen_ai.tool.call.arguments"]
        assert json.loads(recorded_args) == {"x": 7}


# ---------------------------------------------------------------------------
# Conversation span: dict message input (sync and async)
# ---------------------------------------------------------------------------
class TestConversationSpanDictMessage:
    """Tests that initiate_chat/a_initiate_chat handles dict message input."""

    def test_initiate_chat_dict_message_records_input(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        dict_message = {"content": "Hello from dict", "role": "user"}
        with TestAgent(recipient, ["reply"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message=dict_message, max_turns=1)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        assert len(conversation_spans) >= 1
        conv_span = conversation_spans[0]
        input_messages = json.loads(conv_span.attributes.get("gen_ai.input.messages", "[]"))
        assert len(input_messages) >= 1
        found = any(part.get("content") == "Hello from dict" for msg in input_messages for part in msg.get("parts", []))
        assert found, f"Expected 'Hello from dict' in input messages, got: {input_messages}"

    @pytest.mark.asyncio
    async def test_a_initiate_chat_dict_message_records_input(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        dict_message = {"content": "Async dict msg", "role": "user"}
        with TestAgent(recipient, ["reply"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            await agent.a_initiate_chat(recipient, message=dict_message, max_turns=1)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        assert len(conversation_spans) >= 1
        conv_span = conversation_spans[0]
        input_messages = json.loads(conv_span.attributes.get("gen_ai.input.messages", "[]"))
        found = any(part.get("content") == "Async dict msg" for msg in input_messages for part in msg.get("parts", []))
        assert found, f"Expected 'Async dict msg' in input messages, got: {input_messages}"


# ---------------------------------------------------------------------------
# Conversation span: cost/usage/chat_id attributes
# ---------------------------------------------------------------------------
class TestConversationSpanCostAndUsage:
    """Tests that initiate_chat records cost, usage, and chat_id attributes."""

    def test_initiate_chat_records_chat_id(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["response"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        conv_span = conversation_spans[0]
        assert "gen_ai.conversation.id" in conv_span.attributes
        assert len(conv_span.attributes["gen_ai.conversation.id"]) > 0

    def test_initiate_chat_records_cost(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["response"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        conv_span = conversation_spans[0]
        assert "gen_ai.usage.cost" in conv_span.attributes

    @pytest.mark.asyncio
    async def test_a_initiate_chat_records_chat_id(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["response"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            await agent.a_initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        conv_span = conversation_spans[0]
        assert "gen_ai.conversation.id" in conv_span.attributes
        assert len(conv_span.attributes["gen_ai.conversation.id"]) > 0

    @pytest.mark.asyncio
    async def test_a_initiate_chat_records_cost(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["response"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            await agent.a_initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        conv_span = conversation_spans[0]
        assert "gen_ai.usage.cost" in conv_span.attributes


# ---------------------------------------------------------------------------
# instrument_resume: wrapping and span attributes
# ---------------------------------------------------------------------------
class TestInstrumentResume:
    """Tests for instrument_resume (a_resume wrapping)."""

    def test_wraps_a_resume(self, otel_setup) -> None:
        """instrument_agent should wrap a_resume on GroupChatManager."""
        from autogen.agentchat.groupchat import GroupChat, GroupChatManager

        exporter, provider = otel_setup
        agent1 = ConversableAgent("agent1", llm_config=False, human_input_mode="NEVER")
        gc = GroupChat(agents=[agent1], messages=[], max_round=2)
        manager = GroupChatManager(groupchat=gc, llm_config=False)
        instrument_agent(manager, tracer_provider=provider)
        assert hasattr(manager.a_resume, "__otel_wrapped__")

    def test_resume_idempotent(self, otel_setup) -> None:
        """Double-instrumenting should not double-wrap a_resume."""
        from autogen.agentchat.groupchat import GroupChat, GroupChatManager

        exporter, provider = otel_setup
        agent1 = ConversableAgent("agent1", llm_config=False, human_input_mode="NEVER")
        gc = GroupChat(agents=[agent1], messages=[], max_round=2)
        manager = GroupChatManager(groupchat=gc, llm_config=False)
        instrument_agent(manager, tracer_provider=provider)
        first_a_resume = manager.a_resume
        instrument_agent(manager, tracer_provider=provider)
        assert manager.a_resume is first_a_resume

    @pytest.mark.asyncio
    async def test_a_resume_creates_conversation_span_with_resumed(self, otel_setup) -> None:
        """a_resume should create a conversation span with gen_ai.conversation.resumed=True."""
        from unittest.mock import AsyncMock

        from autogen.agentchat.groupchat import GroupChat, GroupChatManager

        exporter, provider = otel_setup
        agent1 = ConversableAgent("agent1", llm_config=False, human_input_mode="NEVER")
        gc = GroupChat(agents=[agent1], messages=[], max_round=2)
        manager = GroupChatManager(groupchat=gc, llm_config=False)

        # Mock the original a_resume so we don't actually run the group chat
        mock_result = (agent1, {"content": "resumed reply"})
        manager.a_resume = AsyncMock(return_value=mock_result)

        # Now instrument (wraps the mock)
        instrument_agent(manager, tracer_provider=provider)

        # Call the wrapped a_resume
        messages = [{"content": "test msg", "role": "user", "name": "agent1"}]
        await manager.a_resume(messages)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        assert len(conversation_spans) == 1

        conv_span = conversation_spans[0]
        assert conv_span.attributes["gen_ai.operation.name"] == "conversation"
        assert conv_span.attributes["gen_ai.agent.name"] == manager.name
        assert conv_span.attributes["gen_ai.conversation.resumed"] is True


# ---------------------------------------------------------------------------
# instrument_run_chat: wrapping and span attributes
# ---------------------------------------------------------------------------
class TestInstrumentRunChat:
    """Tests for instrument_run_chat (run_chat and a_run_chat wrapping)."""

    def test_wraps_run_chat(self, otel_setup) -> None:
        from autogen.agentchat.groupchat import GroupChat, GroupChatManager

        exporter, provider = otel_setup
        agent1 = ConversableAgent("agent1", llm_config=False, human_input_mode="NEVER")
        gc = GroupChat(agents=[agent1], messages=[], max_round=2)
        manager = GroupChatManager(groupchat=gc, llm_config=False)
        instrument_agent(manager, tracer_provider=provider)
        assert hasattr(manager.run_chat, "__otel_wrapped__")

    def test_wraps_a_run_chat(self, otel_setup) -> None:
        from autogen.agentchat.groupchat import GroupChat, GroupChatManager

        exporter, provider = otel_setup
        agent1 = ConversableAgent("agent1", llm_config=False, human_input_mode="NEVER")
        gc = GroupChat(agents=[agent1], messages=[], max_round=2)
        manager = GroupChatManager(groupchat=gc, llm_config=False)
        instrument_agent(manager, tracer_provider=provider)
        assert hasattr(manager.a_run_chat, "__otel_wrapped__")

    def test_run_chat_idempotent(self, otel_setup) -> None:
        from autogen.agentchat.groupchat import GroupChat, GroupChatManager

        exporter, provider = otel_setup
        agent1 = ConversableAgent("agent1", llm_config=False, human_input_mode="NEVER")
        gc = GroupChat(agents=[agent1], messages=[], max_round=2)
        manager = GroupChatManager(groupchat=gc, llm_config=False)
        instrument_agent(manager, tracer_provider=provider)
        first_run_chat = manager.run_chat
        instrument_agent(manager, tracer_provider=provider)
        assert manager.run_chat is first_run_chat

    def test_run_chat_creates_conversation_span(self, otel_setup) -> None:
        """run_chat should create a conversation span with input messages."""
        from unittest.mock import MagicMock

        from autogen.agentchat.groupchat import GroupChat, GroupChatManager

        exporter, provider = otel_setup
        agent1 = ConversableAgent("agent1", llm_config=False, human_input_mode="NEVER")
        gc = GroupChat(agents=[agent1], messages=[], max_round=2)
        manager = GroupChatManager(groupchat=gc, llm_config=False)

        # Replace run_chat with a mock before instrumenting
        manager.run_chat = MagicMock(return_value=(True, None))

        # Instrument wraps the mock
        instrument_agent(manager, tracer_provider=provider)

        messages = [{"content": "Hello group", "role": "user", "name": "agent1"}]
        mock_config = MagicMock()
        mock_config.messages = [
            {"content": "Hello group", "role": "user", "name": "agent1"},
            {"content": "Group reply", "role": "assistant", "name": "agent1"},
        ]

        manager.run_chat(messages=messages, sender=agent1, config=mock_config)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        assert len(conversation_spans) == 1

        conv_span = conversation_spans[0]
        assert conv_span.attributes["gen_ai.operation.name"] == "conversation"
        assert conv_span.attributes["gen_ai.agent.name"] == manager.name

        # Check input messages were captured
        input_messages = json.loads(conv_span.attributes.get("gen_ai.input.messages", "[]"))
        assert len(input_messages) >= 1
        found = any(part.get("content") == "Hello group" for msg in input_messages for part in msg.get("parts", []))
        assert found, f"Expected 'Hello group' in input messages, got: {input_messages}"

    def test_run_chat_captures_output_from_config(self, otel_setup) -> None:
        """run_chat should capture output messages from config.messages."""
        from unittest.mock import MagicMock

        from autogen.agentchat.groupchat import GroupChat, GroupChatManager

        exporter, provider = otel_setup
        agent1 = ConversableAgent("agent1", llm_config=False, human_input_mode="NEVER")
        gc = GroupChat(agents=[agent1], messages=[], max_round=2)
        manager = GroupChatManager(groupchat=gc, llm_config=False)

        manager.run_chat = MagicMock(return_value=(True, None))
        instrument_agent(manager, tracer_provider=provider)

        messages = [{"content": "Hi", "role": "user", "name": "agent1"}]
        mock_config = MagicMock()
        mock_config.messages = [
            {"content": "Hi", "role": "user", "name": "agent1"},
            {"content": "Hello back", "role": "assistant", "name": "agent1"},
        ]

        manager.run_chat(messages=messages, sender=agent1, config=mock_config)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        conv_span = conversation_spans[0]

        output_messages = json.loads(conv_span.attributes.get("gen_ai.output.messages", "[]"))
        assert len(output_messages) >= 2

    def test_run_chat_no_messages_no_input_attr(self, otel_setup) -> None:
        """run_chat with no messages should not set gen_ai.input.messages."""
        from unittest.mock import MagicMock

        from autogen.agentchat.groupchat import GroupChat, GroupChatManager

        exporter, provider = otel_setup
        agent1 = ConversableAgent("agent1", llm_config=False, human_input_mode="NEVER")
        gc = GroupChat(agents=[agent1], messages=[], max_round=2)
        manager = GroupChatManager(groupchat=gc, llm_config=False)

        manager.run_chat = MagicMock(return_value=(True, None))
        instrument_agent(manager, tracer_provider=provider)

        mock_config = MagicMock()
        mock_config.messages = []

        manager.run_chat(messages=None, sender=agent1, config=mock_config)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        conv_span = conversation_spans[0]
        assert "gen_ai.input.messages" not in conv_span.attributes

    def test_run_chat_no_config_messages_no_output_attr(self, otel_setup) -> None:
        """run_chat with empty config.messages should not set gen_ai.output.messages."""
        from unittest.mock import MagicMock

        from autogen.agentchat.groupchat import GroupChat, GroupChatManager

        exporter, provider = otel_setup
        agent1 = ConversableAgent("agent1", llm_config=False, human_input_mode="NEVER")
        gc = GroupChat(agents=[agent1], messages=[], max_round=2)
        manager = GroupChatManager(groupchat=gc, llm_config=False)

        manager.run_chat = MagicMock(return_value=(True, None))
        instrument_agent(manager, tracer_provider=provider)

        messages = [{"content": "Hi", "role": "user", "name": "agent1"}]
        mock_config = MagicMock()
        mock_config.messages = []

        manager.run_chat(messages=messages, sender=agent1, config=mock_config)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        conv_span = conversation_spans[0]
        assert "gen_ai.output.messages" not in conv_span.attributes

    @pytest.mark.asyncio
    async def test_a_run_chat_creates_conversation_span(self, otel_setup) -> None:
        """a_run_chat should create a conversation span with input messages."""
        from unittest.mock import AsyncMock, MagicMock

        from autogen.agentchat.groupchat import GroupChat, GroupChatManager

        exporter, provider = otel_setup
        agent1 = ConversableAgent("agent1", llm_config=False, human_input_mode="NEVER")
        gc = GroupChat(agents=[agent1], messages=[], max_round=2)
        manager = GroupChatManager(groupchat=gc, llm_config=False)

        manager.a_run_chat = AsyncMock(return_value=(True, None))
        instrument_agent(manager, tracer_provider=provider)

        messages = [{"content": "Async group hello", "role": "user", "name": "agent1"}]
        mock_config = MagicMock()
        mock_config.messages = [
            {"content": "Async group hello", "role": "user", "name": "agent1"},
            {"content": "Async group reply", "role": "assistant", "name": "agent1"},
        ]

        await manager.a_run_chat(messages=messages, sender=agent1, config=mock_config)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        assert len(conversation_spans) == 1

        conv_span = conversation_spans[0]
        assert conv_span.attributes["gen_ai.operation.name"] == "conversation"
        assert conv_span.attributes["gen_ai.agent.name"] == manager.name

        input_messages = json.loads(conv_span.attributes.get("gen_ai.input.messages", "[]"))
        found = any(
            part.get("content") == "Async group hello" for msg in input_messages for part in msg.get("parts", [])
        )
        assert found

    @pytest.mark.asyncio
    async def test_a_run_chat_captures_output(self, otel_setup) -> None:
        """a_run_chat should capture output messages from config.messages."""
        from unittest.mock import AsyncMock, MagicMock

        from autogen.agentchat.groupchat import GroupChat, GroupChatManager

        exporter, provider = otel_setup
        agent1 = ConversableAgent("agent1", llm_config=False, human_input_mode="NEVER")
        gc = GroupChat(agents=[agent1], messages=[], max_round=2)
        manager = GroupChatManager(groupchat=gc, llm_config=False)

        manager.a_run_chat = AsyncMock(return_value=(True, None))
        instrument_agent(manager, tracer_provider=provider)

        messages = [{"content": "Hi", "role": "user", "name": "agent1"}]
        mock_config = MagicMock()
        mock_config.messages = [
            {"content": "Hi", "role": "user", "name": "agent1"},
            {"content": "Bye", "role": "assistant", "name": "agent1"},
        ]

        await manager.a_run_chat(messages=messages, sender=agent1, config=mock_config)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        conv_span = conversation_spans[0]
        output_messages = json.loads(conv_span.attributes.get("gen_ai.output.messages", "[]"))
        assert len(output_messages) >= 2


# ---------------------------------------------------------------------------
# instrument_initiate_chats: wrapping and span attributes
# ---------------------------------------------------------------------------
class TestInstrumentInitiateChats:
    """Tests for instrument_initiate_chats (initiate_chats and a_initiate_chats)."""

    def test_wraps_initiate_chats(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        instrument_agent(agent, tracer_provider=provider)
        assert hasattr(agent.initiate_chats, "__otel_wrapped__")

    def test_wraps_a_initiate_chats(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        instrument_agent(agent, tracer_provider=provider)
        assert hasattr(agent.a_initiate_chats, "__otel_wrapped__")

    def test_initiate_chats_idempotent(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        instrument_agent(agent, tracer_provider=provider)
        first_initiate_chats = agent.initiate_chats
        instrument_agent(agent, tracer_provider=provider)
        assert agent.initiate_chats is first_initiate_chats

    def test_initiate_chats_creates_multi_conversation_span(self, otel_setup) -> None:
        """initiate_chats should create a multi_conversation span with sequential mode."""
        from unittest.mock import MagicMock

        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient1 = ConversableAgent("r1", llm_config=False, human_input_mode="NEVER")
        recipient2 = ConversableAgent("r2", llm_config=False, human_input_mode="NEVER")

        mock_result1 = MagicMock()
        mock_result1.chat_id = 101
        mock_result1.summary = "Summary 1"
        mock_result2 = MagicMock()
        mock_result2.chat_id = 102
        mock_result2.summary = "Summary 2"

        chat_queue = [
            {"recipient": recipient1, "message": "Hello r1"},
            {"recipient": recipient2, "message": "Hello r2"},
        ]

        agent.initiate_chats = MagicMock(return_value=[mock_result1, mock_result2])
        instrument_agent(agent, tracer_provider=provider)

        agent.initiate_chats(chat_queue)

        spans = exporter.get_finished_spans()
        multi_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.MULTI_CONVERSATION.value]
        assert len(multi_spans) == 1

        span = multi_spans[0]
        assert span.attributes["gen_ai.operation.name"] == "initiate_chats"
        assert span.attributes["gen_ai.agent.name"] == "sender"
        assert span.attributes["ag2.chats.count"] == 2
        assert span.attributes["ag2.chats.mode"] == "sequential"

    def test_initiate_chats_records_recipients(self, otel_setup) -> None:
        """initiate_chats should record recipient names."""
        from unittest.mock import MagicMock

        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient1 = ConversableAgent("alice", llm_config=False, human_input_mode="NEVER")
        recipient2 = ConversableAgent("bob", llm_config=False, human_input_mode="NEVER")

        mock_result = MagicMock()
        mock_result.chat_id = 1
        mock_result.summary = "done"

        chat_queue = [
            {"recipient": recipient1, "message": "Hi alice"},
            {"recipient": recipient2, "message": "Hi bob"},
        ]

        agent.initiate_chats = MagicMock(return_value=[mock_result, mock_result])
        instrument_agent(agent, tracer_provider=provider)

        agent.initiate_chats(chat_queue)

        spans = exporter.get_finished_spans()
        multi_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.MULTI_CONVERSATION.value]
        span = multi_spans[0]

        recipients = json.loads(span.attributes["ag2.chats.recipients"])
        assert "alice" in recipients
        assert "bob" in recipients

    def test_initiate_chats_records_chat_ids_and_summaries(self, otel_setup) -> None:
        """initiate_chats should record chat IDs and summaries from results."""
        from unittest.mock import MagicMock

        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("r1", llm_config=False, human_input_mode="NEVER")

        mock_result = MagicMock()
        mock_result.chat_id = 42
        mock_result.summary = "All done"

        chat_queue = [{"recipient": recipient, "message": "Hi"}]

        agent.initiate_chats = MagicMock(return_value=[mock_result])
        instrument_agent(agent, tracer_provider=provider)

        agent.initiate_chats(chat_queue)

        spans = exporter.get_finished_spans()
        multi_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.MULTI_CONVERSATION.value]
        span = multi_spans[0]

        chat_ids = json.loads(span.attributes["ag2.chats.ids"])
        assert "42" in chat_ids

        summaries = json.loads(span.attributes["ag2.chats.summaries"])
        assert "All done" in summaries

    @pytest.mark.asyncio
    async def test_a_initiate_chats_creates_multi_conversation_span(self, otel_setup) -> None:
        """a_initiate_chats should create a multi_conversation span with parallel mode."""
        from unittest.mock import AsyncMock, MagicMock

        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient1 = ConversableAgent("r1", llm_config=False, human_input_mode="NEVER")
        recipient2 = ConversableAgent("r2", llm_config=False, human_input_mode="NEVER")

        mock_result1 = MagicMock()
        mock_result1.chat_id = 201
        mock_result1.summary = "Async summary 1"
        mock_result2 = MagicMock()
        mock_result2.chat_id = 202
        mock_result2.summary = "Async summary 2"

        mock_results = {0: mock_result1, 1: mock_result2}

        chat_queue = [
            {"recipient": recipient1, "message": "Hello r1"},
            {"recipient": recipient2, "message": "Hello r2"},
        ]

        agent.a_initiate_chats = AsyncMock(return_value=mock_results)
        instrument_agent(agent, tracer_provider=provider)

        await agent.a_initiate_chats(chat_queue)

        spans = exporter.get_finished_spans()
        multi_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.MULTI_CONVERSATION.value]
        assert len(multi_spans) == 1

        span = multi_spans[0]
        assert span.attributes["gen_ai.operation.name"] == "initiate_chats"
        assert span.attributes["ag2.chats.count"] == 2
        assert span.attributes["ag2.chats.mode"] == "parallel"

    @pytest.mark.asyncio
    async def test_a_initiate_chats_records_recipients(self, otel_setup) -> None:
        """a_initiate_chats should record recipient names."""
        from unittest.mock import AsyncMock, MagicMock

        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient1 = ConversableAgent("charlie", llm_config=False, human_input_mode="NEVER")
        recipient2 = ConversableAgent("dana", llm_config=False, human_input_mode="NEVER")

        mock_result = MagicMock()
        mock_result.chat_id = 1
        mock_result.summary = "done"

        chat_queue = [
            {"recipient": recipient1, "message": "Hi"},
            {"recipient": recipient2, "message": "Hi"},
        ]

        agent.a_initiate_chats = AsyncMock(return_value={0: mock_result, 1: mock_result})
        instrument_agent(agent, tracer_provider=provider)

        await agent.a_initiate_chats(chat_queue)

        spans = exporter.get_finished_spans()
        multi_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.MULTI_CONVERSATION.value]
        span = multi_spans[0]

        recipients = json.loads(span.attributes["ag2.chats.recipients"])
        assert "charlie" in recipients
        assert "dana" in recipients

    @pytest.mark.asyncio
    async def test_a_initiate_chats_records_chat_ids_and_summaries(self, otel_setup) -> None:
        """a_initiate_chats should record chat IDs and summaries from results dict."""
        from unittest.mock import AsyncMock, MagicMock

        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("r1", llm_config=False, human_input_mode="NEVER")

        mock_result = MagicMock()
        mock_result.chat_id = 99
        mock_result.summary = "Async done"

        chat_queue = [{"recipient": recipient, "message": "Hi"}]

        agent.a_initiate_chats = AsyncMock(return_value={0: mock_result})
        instrument_agent(agent, tracer_provider=provider)

        await agent.a_initiate_chats(chat_queue)

        spans = exporter.get_finished_spans()
        multi_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.MULTI_CONVERSATION.value]
        span = multi_spans[0]

        chat_ids = json.loads(span.attributes["ag2.chats.ids"])
        assert "99" in chat_ids

        summaries = json.loads(span.attributes["ag2.chats.summaries"])
        assert "Async done" in summaries

    @pytest.mark.asyncio
    async def test_a_initiate_chats_records_prerequisites(self, otel_setup) -> None:
        """a_initiate_chats should record prerequisites when present in chat_queue."""
        from unittest.mock import AsyncMock, MagicMock

        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient1 = ConversableAgent("r1", llm_config=False, human_input_mode="NEVER")
        recipient2 = ConversableAgent("r2", llm_config=False, human_input_mode="NEVER")

        mock_result = MagicMock()
        mock_result.chat_id = 1
        mock_result.summary = "done"

        chat_queue = [
            {"chat_id": 1, "recipient": recipient1, "message": "Hi", "prerequisites": []},
            {"chat_id": 2, "recipient": recipient2, "message": "Hi", "prerequisites": [1]},
        ]

        agent.a_initiate_chats = AsyncMock(return_value={0: mock_result, 1: mock_result})
        instrument_agent(agent, tracer_provider=provider)

        await agent.a_initiate_chats(chat_queue)

        spans = exporter.get_finished_spans()
        multi_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.MULTI_CONVERSATION.value]
        span = multi_spans[0]

        assert "ag2.chats.prerequisites" in span.attributes
        prerequisites = json.loads(span.attributes["ag2.chats.prerequisites"])
        assert prerequisites["2"] == [1]
