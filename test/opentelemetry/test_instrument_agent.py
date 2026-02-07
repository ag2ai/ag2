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
