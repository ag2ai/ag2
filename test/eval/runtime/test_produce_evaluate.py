# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end trace-based flow: run an agent, persist its spans to disk, then
evaluate the stored trace through a TraceSource — the produce-then-evaluate
path a developer follows, with execution and grading fully decoupled.
"""

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ag2 import Agent
from ag2.eval import DirectoryTraceSource, Suite, evaluate_traces
from ag2.eval.scorers import final_answer_matches, no_tool_errors, tool_called
from ag2.eval.sources._otel import readable_span_to_data
from ag2.eval.sources.trace_source import save_trace
from ag2.events import ToolCallEvent
from ag2.middleware.builtin.telemetry import TelemetryMiddleware
from ag2.testing import TestConfig
from ag2.tools import tool


@pytest.mark.asyncio()
async def test_produce_to_disk_then_evaluate(tmp_path) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    @tool
    def get_weather(city: str) -> str:
        return f"Sunny in {city}"

    agent = Agent(
        "weather",
        config=TestConfig([ToolCallEvent(name="get_weather", arguments='{"city": "Paris"}')], "It's sunny in Paris."),
        tools=[get_weather],
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="weather", model_name="mock")],
    )

    # Produce: run the agent, persist its spans to disk.
    await agent.ask("weather in Paris?")
    spans = [readable_span_to_data(s) for s in exporter.get_finished_spans()]
    traces_dir = tmp_path / "traces"
    save_trace(traces_dir, "trace-1", spans, task_id="task-1")

    # Evaluate: grade the stored trace — no agent run involved.
    suite = Suite.from_list([
        {"task_id": "task-1", "inputs": {"input": "weather in Paris?"}, "reference_outputs": {"answer": "Paris"}},
    ])
    result = await evaluate_traces(
        DirectoryTraceSource(traces_dir),
        scorers=[
            tool_called("get_weather"),
            no_tool_errors(),
            final_answer_matches(field="answer", matcher="contains"),
        ],
        suite=suite,
        store_dir=tmp_path / "runs",
    )

    assert result.pass_rate("tool_called[get_weather]") == 1.0
    assert result.pass_rate("no_tool_errors") == 1.0
    assert result.pass_rate("final_answer_matches") == 1.0
    assert (tmp_path / "runs" / f"{result.run_id}.json").exists()
