# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from collections.abc import Sequence as SequenceType

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult, SpanExporter

from autogen.beta import Agent
from autogen.beta.events import ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent
from autogen.beta.middleware.builtin.telemetry import TelemetryMiddleware
from autogen.beta.testing import TestConfig
from autogen.beta.tools import tool


class _InMemorySpanExporter(SpanExporter):
    """In-memory span exporter for tests."""

    def __init__(self) -> None:
        self._spans: list[ReadableSpan] = []
        self._lock = threading.Lock()

    def export(self, spans: SequenceType[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self) -> list[ReadableSpan]:
        with self._lock:
            return list(self._spans)

    def shutdown(self) -> None:
        with self._lock:
            self._spans.clear()


@pytest.fixture()
def otel_setup():
    exporter = _InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider


@pytest.mark.asyncio()
async def test_turn_span_emitted(otel_setup):
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(ModelResponse(message=ModelMessage(content="Hello!"))),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    await agent.ask("Hi")

    spans = exporter.get_finished_spans()
    turn_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "agent"]
    assert len(turn_spans) == 1
    span = turn_spans[0]
    assert span.name == "invoke_agent assistant"
    assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert span.attributes["gen_ai.agent.name"] == "assistant"


@pytest.mark.asyncio()
async def test_llm_span_with_usage(otel_setup):
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                message=ModelMessage(content="Hi!"),
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            ),
        ),
        middleware=[
            TelemetryMiddleware(
                tracer_provider=provider,
                agent_name="assistant",
                provider_name="openai",
                model_name="gpt-4o-mini",
            )
        ],
    )

    await agent.ask("Hello")

    spans = exporter.get_finished_spans()
    llm_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "llm"]
    assert len(llm_spans) == 1
    span = llm_spans[0]
    assert span.name == "chat gpt-4o-mini"
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4o-mini"
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
    assert span.attributes["gen_ai.usage.output_tokens"] == 5


@pytest.mark.asyncio()
async def test_tool_span(otel_setup):
    exporter, provider = otel_setup

    @tool
    def get_weather(city: str) -> str:
        """Get weather."""
        return f"Sunny in {city}"

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent(
                    calls=[ToolCallEvent(id="call_1", name="get_weather", arguments='{"city": "NYC"}')]
                ),
            ),
            ModelResponse(message=ModelMessage(content="It's sunny in NYC")),
        ),
        tools=[get_weather],
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant", capture_content=False)],
    )

    await agent.ask("Weather?")

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "tool"]
    assert len(tool_spans) == 1
    span = tool_spans[0]
    assert span.name == "execute_tool get_weather"
    assert span.attributes["gen_ai.tool.name"] == "get_weather"
    assert span.attributes["gen_ai.tool.call.id"] == "call_1"
    # capture_content=False explicitly, so no arguments attribute
    assert "gen_ai.tool.call.arguments" not in span.attributes


@pytest.mark.asyncio()
async def test_tool_span_with_content_capture(otel_setup):
    exporter, provider = otel_setup

    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello {name}"

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent(
                    calls=[ToolCallEvent(id="call_1", name="greet", arguments='{"name": "World"}')]
                ),
            ),
            ModelResponse(message=ModelMessage(content="Done")),
        ),
        tools=[greet],
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant", capture_content=True)],
    )

    await agent.ask("Greet")

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "tool"]
    assert len(tool_spans) == 1
    span = tool_spans[0]
    assert span.attributes["gen_ai.tool.call.arguments"] == '{"name": "World"}'
    assert "Hello World" in span.attributes["gen_ai.tool.call.result"]


@pytest.mark.asyncio()
async def test_tool_error_marks_span_error(otel_setup):
    exporter, provider = otel_setup

    @tool
    def fail_tool() -> str:
        """Always fails."""
        raise ValueError("something went wrong")

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent(calls=[ToolCallEvent(id="call_1", name="fail_tool", arguments="{}")]),
            ),
            ModelResponse(message=ModelMessage(content="Error handled")),
        ),
        tools=[fail_tool],
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    # TestClient re-raises ToolError.error on the next LLM call, so we expect ValueError
    with pytest.raises(ValueError, match="something went wrong"):
        await agent.ask("Do it")

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "tool"]
    assert len(tool_spans) == 1
    span = tool_spans[0]
    assert span.status.status_code.name == "ERROR"


@pytest.mark.asyncio()
async def test_span_parent_child_hierarchy(otel_setup):
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(message=ModelMessage(content="Hi!"), usage={"prompt_tokens": 5, "completion_tokens": 3}),
        ),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    await agent.ask("Hello")

    spans = exporter.get_finished_spans()
    turn_span = next(s for s in spans if s.attributes.get("ag2.span.type") == "agent")
    llm_span = next(s for s in spans if s.attributes.get("ag2.span.type") == "llm")

    # LLM span should be a child of the turn span
    assert llm_span.parent is not None
    assert llm_span.parent.span_id == turn_span.context.span_id


@pytest.mark.asyncio()
async def test_capture_content_false_omits_messages(otel_setup):
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(message=ModelMessage(content="Secret response")),
        ),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant", capture_content=False)],
    )

    await agent.ask("Secret question")

    spans = exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.attributes.get("ag2.span.type") == "llm")
    assert "gen_ai.input.messages" not in llm_span.attributes
    assert "gen_ai.output.messages" not in llm_span.attributes


@pytest.mark.asyncio()
async def test_capture_content_true_includes_messages(otel_setup):
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(message=ModelMessage(content="Hello!")),
        ),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant", capture_content=True)],
    )

    await agent.ask("Hi")

    spans = exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.attributes.get("ag2.span.type") == "llm")
    assert "gen_ai.input.messages" in llm_span.attributes
    assert "gen_ai.output.messages" in llm_span.attributes

    input_msgs = json.loads(llm_span.attributes["gen_ai.input.messages"])
    assert any("Hi" in str(m) for m in input_msgs)


@pytest.mark.asyncio()
async def test_auto_detect_model_provider_from_response(otel_setup):
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                message=ModelMessage(content="Hi!"),
                model="gpt-4o-mini-2024-07-18",
                provider="openai",
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            ),
        ),
        middleware=[
            TelemetryMiddleware(
                tracer_provider=provider,
                agent_name="assistant",
                # No provider_name or model_name — should auto-detect from response
            )
        ],
    )

    await agent.ask("Hello")

    spans = exporter.get_finished_spans()
    llm_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "llm"]
    assert len(llm_spans) == 1
    span = llm_spans[0]
    assert span.name == "chat gpt-4o-mini-2024-07-18"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4o-mini-2024-07-18"
    assert span.attributes["gen_ai.response.model"] == "gpt-4o-mini-2024-07-18"
    assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
    assert span.attributes["gen_ai.usage.output_tokens"] == 5


@pytest.mark.asyncio()
async def test_tool_span_has_tool_type(otel_setup):
    exporter, provider = otel_setup

    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello {name}"

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent(
                    calls=[ToolCallEvent(id="call_1", name="greet", arguments='{"name": "World"}')]
                ),
            ),
            ModelResponse(message=ModelMessage(content="Done")),
        ),
        tools=[greet],
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    await agent.ask("Greet")

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "tool"]
    assert len(tool_spans) == 1
    assert tool_spans[0].attributes["gen_ai.tool.type"] == "function"


@pytest.mark.asyncio()
async def test_constructor_params_override_response(otel_setup):
    """When constructor provides model_name/provider_name, those take precedence."""
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                message=ModelMessage(content="Hi!"),
                model="gpt-4o-mini-resolved",
                provider="openai",
                finish_reason="stop",
            ),
        ),
        middleware=[
            TelemetryMiddleware(
                tracer_provider=provider,
                agent_name="assistant",
                provider_name="custom-provider",
                model_name="custom-model",
            )
        ],
    )

    await agent.ask("Hello")

    spans = exporter.get_finished_spans()
    llm_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "llm"]
    span = llm_spans[0]
    # Constructor params win for request attributes
    assert span.attributes["gen_ai.provider.name"] == "custom-provider"
    assert span.attributes["gen_ai.request.model"] == "custom-model"
    # Response model still set
    assert span.attributes["gen_ai.response.model"] == "gpt-4o-mini-resolved"


@pytest.mark.asyncio()
async def test_cache_token_usage_attributes(otel_setup):
    """Cache creation/read token counts appear in LLM span attributes."""
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                message=ModelMessage(content="Hi!"),
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "cache_creation_input_tokens": 80,
                    "cache_read_input_tokens": 0,
                },
                model="claude-sonnet-4-6",
                provider="anthropic",
            ),
        ),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    await agent.ask("Hello")

    spans = exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.attributes.get("ag2.span.type") == "llm")
    assert llm_span.attributes["gen_ai.usage.input_tokens"] == 100
    assert llm_span.attributes["gen_ai.usage.output_tokens"] == 20
    assert llm_span.attributes["gen_ai.usage.cache_creation_input_tokens"] == 80
    # cache_read_input_tokens is 0, so it should NOT be set (guarded by `if usage.get(...)`)
    assert "gen_ai.usage.cache_read_input_tokens" not in llm_span.attributes


@pytest.mark.asyncio()
async def test_cache_read_tokens_when_nonzero(otel_setup):
    """cache_read_input_tokens appears when non-zero (simulates cache hit)."""
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                message=ModelMessage(content="Hi!"),
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 75,
                },
                model="claude-sonnet-4-6",
                provider="anthropic",
            ),
        ),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    await agent.ask("Hello")

    spans = exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.attributes.get("ag2.span.type") == "llm")
    assert llm_span.attributes["gen_ai.usage.cache_read_input_tokens"] == 75
    assert "gen_ai.usage.cache_creation_input_tokens" not in llm_span.attributes
