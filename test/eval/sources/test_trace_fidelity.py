# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Live round-trip fidelity: a Trace reconstructed from spans must match the
Trace captured from the in-memory event stream, on what scorers actually read.

Runs a real (mocked-LLM) agent with ``TelemetryMiddleware`` while capturing
both the live event stream (``EventCapture`` — the runner's substrate) and the
emitted spans. The two Traces are compared on scorer-relevant projections, so
a divergence between what telemetry emits and what scorers expect surfaces here
rather than silently scoring a replayed trace incorrectly.
"""

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ag2 import Agent
from ag2.eval.runtime._capture import EventCapture
from ag2.eval.scorers import no_tool_errors, tool_called
from ag2.eval.sources._otel import readable_spans_to_trace
from ag2.eval.trace import Trace
from ag2.events import ModelMessage, ModelResponse, ToolCallEvent, ToolErrorEvent, ToolResultEvent, Usage
from ag2.middleware.builtin.telemetry import TelemetryMiddleware
from ag2.testing import TestConfig
from ag2.tools import tool


def _live_trace(capture: EventCapture) -> Trace:
    return Trace(events=capture.events, exception=None, duration_ms=0)


@pytest.fixture()
def otel_provider() -> tuple[InMemorySpanExporter, TracerProvider]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider


@pytest.mark.asyncio()
async def test_llm_only_run_tokens_and_response_match(otel_provider) -> None:
    exporter, provider = otel_provider
    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(message=ModelMessage("Hello there!"), usage=Usage(prompt_tokens=11, completion_tokens=4))
        ),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant", model_name="mock")],
    )
    capture = EventCapture()

    await agent.ask("Hi", observers=[capture])

    live = _live_trace(capture)
    spans = readable_spans_to_trace(exporter.get_finished_spans())

    assert len(spans.events_of(ModelResponse)) == len(live.events_of(ModelResponse))
    assert spans.tokens == live.tokens
    assert spans.tokens.input == 11
    assert spans.tokens.output == 4
    assert spans.events_of(ModelResponse)[-1].content == live.events_of(ModelResponse)[-1].content


@pytest.mark.asyncio()
async def test_tool_run_tool_calls_and_results_match(otel_provider) -> None:
    exporter, provider = otel_provider

    @tool
    def get_weather(city: str) -> str:
        return f"Sunny in {city}"

    agent = Agent(
        "weather",
        config=TestConfig([ToolCallEvent(name="get_weather", arguments='{"city": "NYC"}')], "It's sunny in NYC."),
        tools=[get_weather],
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="weather", model_name="mock")],
    )
    capture = EventCapture()

    await agent.ask("Weather in NYC?", observers=[capture])

    live = _live_trace(capture)
    spans = readable_spans_to_trace(exporter.get_finished_spans())

    def calls(trace: Trace) -> list[tuple[str, str]]:
        return [(c.name, c.arguments) for c in trace.events_of(ToolCallEvent)]

    assert calls(spans) == calls(live)
    assert calls(spans) == [("get_weather", '{"city": "NYC"}')]
    assert len(spans.events_of(ToolResultEvent)) == len(live.events_of(ToolResultEvent)) == 1
    assert len(spans.events_of(ToolErrorEvent)) == len(live.events_of(ToolErrorEvent)) == 0

    # Prebuilt scorers reach the same verdict on both Traces.
    assert tool_called("get_weather")._fn(trace=spans) == tool_called("get_weather")._fn(trace=live) is True
    assert no_tool_errors()._fn(trace=spans) == no_tool_errors()._fn(trace=live) is True
