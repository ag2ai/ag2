# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that content gating (RECORD_CONTENT=False) suppresses sensitive attributes.

When AG2_OTEL_RECORD_CONTENT is not set (the default), spans must still capture
structural metadata but must NOT include message content, tool arguments, or
tool results.
"""

import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

import autogen.opentelemetry.instrumentators.agent_instrumentators._config as _otel_cfg
from autogen import ConversableAgent
from autogen.opentelemetry import instrument_agent
from autogen.opentelemetry.consts import SpanType
from autogen.opentelemetry.instrumentators.agent_instrumentators.human_input import instrument_human_input
from autogen.opentelemetry.setup import get_tracer
from autogen.testing import TestAgent
from test.opentelemetry.conftest import InMemorySpanExporter

# Content attributes that must be absent when RECORD_CONTENT is False
_CONTENT_ATTRS = {
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
    "ag2.human_input.prompt",
    "ag2.human_input.response",
    "ag2.chats.summaries",
}


@pytest.fixture()
def otel_setup():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider


@pytest.fixture(autouse=True)
def _disable_content_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override the conftest autouse fixture to disable content recording."""
    monkeypatch.setattr(_otel_cfg, "RECORD_CONTENT", False)


def _assert_no_content_attrs(span) -> None:
    """Verify that no content-bearing attributes exist on a span."""
    for attr in _CONTENT_ATTRS:
        assert attr not in span.attributes, f"Content attribute {attr!r} should not be set when RECORD_CONTENT=False"


# ---------------------------------------------------------------------------
# Conversation span -- initiate_chat (sync)
# ---------------------------------------------------------------------------
class TestConversationContentGating:
    def test_initiate_chat_omits_content(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["Hello back!"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message="secret message", max_turns=1)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        assert len(conversation_spans) >= 1

        conv_span = conversation_spans[0]
        _assert_no_content_attrs(conv_span)
        # Structural metadata must still be present
        assert conv_span.attributes["gen_ai.operation.name"] == "conversation"
        assert conv_span.attributes["gen_ai.agent.name"] == "sender"

    @pytest.mark.asyncio
    async def test_a_initiate_chat_omits_content(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["Hello back!"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            await agent.a_initiate_chat(recipient, message="secret message", max_turns=1)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        assert len(conversation_spans) >= 1
        _assert_no_content_attrs(conversation_spans[0])

    def test_initiate_chat_with_dict_message_omits_content(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["response"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message={"content": "secret", "role": "user"}, max_turns=1)

        spans = exporter.get_finished_spans()
        conversation_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CONVERSATION.value]
        assert len(conversation_spans) >= 1
        _assert_no_content_attrs(conversation_spans[0])


# ---------------------------------------------------------------------------
# Agent span -- generate_reply
# ---------------------------------------------------------------------------
class TestAgentContentGating:
    def test_generate_reply_omits_content(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["reply"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            agent.initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        agent_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.AGENT.value]
        assert len(agent_spans) >= 1

        for span in agent_spans:
            _assert_no_content_attrs(span)
            # Structural metadata must still be present
            assert span.attributes["gen_ai.operation.name"] == "invoke_agent"

    @pytest.mark.asyncio
    async def test_a_generate_reply_omits_content(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("sender", llm_config=False, human_input_mode="NEVER")
        recipient = ConversableAgent("recipient", llm_config=False, human_input_mode="NEVER")

        with TestAgent(recipient, ["reply"]):
            instrument_agent(agent, tracer_provider=provider)
            instrument_agent(recipient, tracer_provider=provider)
            await agent.a_initiate_chat(recipient, message="Hello", max_turns=1)

        spans = exporter.get_finished_spans()
        agent_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.AGENT.value]
        assert len(agent_spans) >= 1
        for span in agent_spans:
            _assert_no_content_attrs(span)


# ---------------------------------------------------------------------------
# Tool span -- execute_function
# ---------------------------------------------------------------------------
class TestToolContentGating:
    def test_execute_function_omits_arguments_and_result(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("tool_agent", llm_config=False, human_input_mode="NEVER")

        def my_tool(x: int) -> str:
            return f"result: {x}"

        agent.register_function({"my_tool": my_tool})
        instrument_agent(agent, tracer_provider=provider)

        func_call = {"name": "my_tool", "arguments": json.dumps({"x": 42})}
        is_success, _result = agent.execute_function(func_call)
        assert is_success

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.TOOL.value]
        assert len(tool_spans) == 1

        tool_span = tool_spans[0]
        _assert_no_content_attrs(tool_span)
        # Structural metadata must still be present
        assert tool_span.attributes["gen_ai.tool.name"] == "my_tool"
        assert tool_span.attributes["gen_ai.tool.type"] == "function"

    @pytest.mark.asyncio
    async def test_a_execute_function_omits_arguments_and_result(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("tool_agent", llm_config=False, human_input_mode="NEVER")

        async def my_async_tool(x: int) -> str:
            return f"result: {x}"

        agent.register_function({"my_async_tool": my_async_tool})
        instrument_agent(agent, tracer_provider=provider)

        func_call = {"name": "my_async_tool", "arguments": json.dumps({"x": 42})}
        is_success, _result = await agent.a_execute_function(func_call)
        assert is_success

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.TOOL.value]
        assert len(tool_spans) == 1
        _assert_no_content_attrs(tool_spans[0])

    def test_execute_function_failure_still_records_error_type(self, otel_setup) -> None:
        exporter, provider = otel_setup
        agent = ConversableAgent("tool_agent", llm_config=False, human_input_mode="NEVER")

        def failing_tool() -> str:
            raise ValueError("boom")

        agent.register_function({"failing_tool": failing_tool})
        instrument_agent(agent, tracer_provider=provider)

        func_call = {"name": "failing_tool", "arguments": "{}"}
        is_success, _result = agent.execute_function(func_call)
        assert not is_success

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.TOOL.value]
        tool_span = tool_spans[0]
        # Error type is structural, not content -- should still be recorded
        assert tool_span.attributes.get("error.type") == "ExecutionError"
        _assert_no_content_attrs(tool_span)


# ---------------------------------------------------------------------------
# Human input span
# ---------------------------------------------------------------------------
class TestHumanInputContentGating:
    def test_get_human_input_omits_prompt_and_response(self, otel_setup) -> None:
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        agent = ConversableAgent("test_agent2", llm_config=False, human_input_mode="ALWAYS")

        def mock_get_human_input(prompt: str) -> str:
            return "mock_response"

        agent.get_human_input = mock_get_human_input
        instrument_human_input(agent, tracer=tracer)

        result = agent.get_human_input("Enter secret:")
        assert result == "mock_response"

        spans = exporter.get_finished_spans()
        hi_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.HUMAN_INPUT.value]
        assert len(hi_spans) == 1

        hi_span = hi_spans[0]
        _assert_no_content_attrs(hi_span)
        # Structural metadata must still be present
        assert hi_span.attributes["gen_ai.operation.name"] == "await_human_input"
        assert hi_span.attributes["gen_ai.agent.name"] == "test_agent2"

    @pytest.mark.asyncio
    async def test_a_get_human_input_omits_prompt_and_response(self, otel_setup) -> None:
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        agent = ConversableAgent("test_agent3", llm_config=False, human_input_mode="ALWAYS")

        async def mock_a_get_human_input(prompt: str) -> str:
            return "async_mock_response"

        agent.a_get_human_input = mock_a_get_human_input
        instrument_human_input(agent, tracer=tracer)

        result = await agent.a_get_human_input("Enter secret:")
        assert result == "async_mock_response"

        spans = exporter.get_finished_spans()
        hi_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.HUMAN_INPUT.value]
        assert len(hi_spans) == 1
        _assert_no_content_attrs(hi_spans[0])
