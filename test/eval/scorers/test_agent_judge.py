# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Agent-as-judge scorer (``ag2.eval.scorers.judge``)."""

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ag2._telemetry_consts import ATTR_SPAN_TYPE, SPAN_TYPE_AGENT, SPAN_TYPE_LLM
from ag2.eval import InMemoryTraceSource, Suite, TraceRef, evaluate_traces
from ag2.eval.dataset.task import Task
from ag2.eval.scorers import agent_judge
from ag2.eval.trace import Trace
from ag2.events import ModelMessage, ModelResponse, ToolCallEvent
from ag2.middleware.builtin.telemetry import TelemetryMiddleware
from ag2.testing import TestConfig, TrackingConfig


def _empty_trace() -> Trace:
    return Trace(events=[], exception=None, duration_ms=0)


async def _score(scorer, *, outputs, reference_outputs=None, trace=None, inputs=None) -> list:
    return await scorer(
        inputs=inputs or {},
        outputs=outputs,
        reference_outputs=reference_outputs,
        trace=trace if trace is not None else _empty_trace(),
        task=Task(task_id="t", inputs={}),
    )


@pytest.mark.asyncio()
async def test_verdict_maps_to_single_feedback() -> None:
    judge = agent_judge(
        TestConfig('{"score": 0.9, "reasoning": "looks correct"}'),
        criterion="answer is correct vs reference",
        key="correctness",
    )

    [fb] = await _score(judge, inputs={"input": "q"}, outputs={"body": "a"}, reference_outputs={"answer": "a"})

    assert fb.key == "correctness"
    assert fb.score == 0.9
    assert fb.comment == "looks correct"


@pytest.mark.asyncio()
async def test_verdict_without_label_has_no_value() -> None:
    judge = agent_judge(TestConfig('{"score": 0.5, "reasoning": "meh"}'), criterion="quality", key="quality")

    [fb] = await _score(judge, outputs={"body": "a"})

    assert fb.score == 0.5
    assert fb.value is None
    assert fb.comment == "meh"


@pytest.mark.asyncio()
async def test_multi_dimensional_ensemble_produces_per_dimension_columns(tmp_path) -> None:
    trace = Trace(events=[ModelResponse(message=ModelMessage("Paris"))], exception=None, duration_ms=5)
    source = InMemoryTraceSource([(TraceRef("t1", task_id="task-1"), trace)])
    suite = Suite.from_list([
        {"task_id": "task-1", "inputs": {"input": "capital of France?"}, "reference_outputs": {"answer": "Paris"}},
    ])
    correctness = agent_judge(
        TestConfig('{"score": 1.0, "reasoning": "correct"}'), criterion="correct vs reference", key="correctness"
    )
    faithfulness = agent_judge(
        TestConfig('{"score": 0.5, "reasoning": "partly grounded"}'),
        criterion="grounded in tool results",
        key="faithfulness",
    )

    result = await evaluate_traces(source, scorers=[correctness, faithfulness], suite=suite, store_dir=tmp_path)

    assert result.score_stats("correctness").mean == 1.0
    assert result.score_stats("faithfulness").mean == 0.5


@pytest.mark.asyncio()
async def test_trajectory_judge_runs_with_trace() -> None:
    trace = Trace(
        events=[ToolCallEvent("get_weather", arguments='{"city": "NYC"}')],
        exception=None,
        duration_ms=0,
    )
    judge = agent_judge(
        TestConfig('{"score": 1.0, "reasoning": "used the tool"}'),
        criterion="used the right tool",
        key="tool_use",
        include_trace=True,
    )

    [fb] = await _score(judge, inputs={"input": "weather?"}, outputs={"body": "sunny"}, trace=trace)

    assert fb.key == "tool_use"
    assert fb.score == 1.0


@pytest.mark.asyncio()
async def test_invalid_judge_output_is_captured_not_raised() -> None:
    judge = agent_judge(TestConfig("not valid json"), criterion="x", key="correctness", retries=0)

    [fb] = await _score(judge, outputs={"body": "a"})

    assert fb.key == "correctness"
    assert fb.score is None
    assert "scorer raised" in (fb.comment or "")


@pytest.mark.asyncio()
async def test_score_is_clamped_to_scale() -> None:
    judge = agent_judge(
        TestConfig('{"score": 5.0, "reasoning": "way over"}'),
        criterion="x",
        key="correctness",
        scale=(0.0, 1.0),
    )

    [fb] = await _score(judge, outputs={"body": "a"})

    assert fb.score == 1.0
    assert "clamped" in (fb.comment or "")


@pytest.mark.asyncio()
async def test_reference_rendered_into_prompt_by_default() -> None:
    config = TrackingConfig(TestConfig('{"score": 1.0, "reasoning": "ok"}'))
    judge = agent_judge(config, criterion="correct vs reference", key="correctness")

    await _score(judge, inputs={"input": "q"}, outputs={"body": "a"}, reference_outputs={"answer": "gold"})

    prompt = repr(config.mock.call_args.args[0])
    assert "## Reference" in prompt
    assert "gold" in prompt


@pytest.mark.asyncio()
async def test_include_reference_false_omits_reference_section() -> None:
    config = TrackingConfig(TestConfig('{"score": 1.0, "reasoning": "ok"}'))
    judge = agent_judge(config, criterion="grounded in tool results", key="faithfulness", include_reference=False)

    await _score(judge, inputs={"input": "q"}, outputs={"body": "a"}, reference_outputs={"answer": "gold"})

    prompt = repr(config.mock.call_args.args[0])
    assert "## Reference" not in prompt
    assert "gold" not in prompt


@pytest.mark.asyncio()
async def test_telemetry_middleware_observes_the_judge() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    judge = agent_judge(
        TestConfig('{"score": 1.0, "reasoning": "ok"}'),
        criterion="x",
        key="correctness",
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="judge", model_name="mock")],
    )

    await _score(judge, outputs={"body": "a"})

    span_types = {s.attributes.get(ATTR_SPAN_TYPE) for s in exporter.get_finished_spans()}
    assert SPAN_TYPE_AGENT in span_types  # the judge's own invoke_agent span
    assert SPAN_TYPE_LLM in span_types  # the judge's own LLM call -> token usage observable
