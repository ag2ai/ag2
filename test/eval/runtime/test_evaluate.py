# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluate_traces() — grading traces from a TraceSource."""

import pytest

from ag2.eval import (
    BudgetThresholds,
    InMemoryTraceSource,
    Suite,
    TraceRef,
    evaluate_traces,
    scorer,
)
from ag2.eval.scorers import final_answer_matches, tool_called
from ag2.eval.trace import Trace
from ag2.events import ModelMessage, ModelResponse, ToolCallEvent, Usage


def _trace(answer: str, *, tool_name: str | None = None, in_tok: int = 0, out_tok: int = 0) -> Trace:
    events: list = []
    if tool_name is not None:
        events.append(ToolCallEvent(tool_name, arguments="{}"))
    events.append(
        ModelResponse(message=ModelMessage(answer), usage=Usage(prompt_tokens=in_tok, completion_tokens=out_tok))
    )
    return Trace(events=events, exception=None, duration_ms=10)


@scorer
def has_one_response(trace: Trace) -> bool:
    return len(trace.events_of(ModelResponse)) == 1


@scorer
def answer_is_paris(outputs: dict) -> bool:
    """True iff the parsed ``content`` carries answer == 'Paris'."""
    content = outputs.get("content")
    return isinstance(content, dict) and content.get("answer") == "Paris"


@scorer
def free_text_content_mirrors_body(outputs: dict) -> bool:
    """For a non-JSON answer, ``content`` is the text itself (mirrors reply.content())."""
    return isinstance(outputs.get("content"), str) and outputs["content"] == outputs.get("body")


@pytest.mark.asyncio()
async def test_evaluate_scores_persists_and_joins_reference(tmp_path) -> None:
    source = InMemoryTraceSource([
        (TraceRef("t1", task_id="task-1"), _trace("Paris", tool_name="get_weather", in_tok=5, out_tok=2)),
    ])
    suite = Suite.from_list([
        {"task_id": "task-1", "inputs": {"input": "capital of France?"}, "reference_outputs": {"answer": "Paris"}},
    ])

    result = await evaluate_traces(
        source,
        scorers=[tool_called("get_weather"), final_answer_matches(field="answer", matcher="contains")],
        suite=suite,
        store_dir=tmp_path,
    )

    assert result.pass_rate("tool_called[get_weather]") == 1.0
    assert result.pass_rate("final_answer_matches") == 1.0  # reference joined via task_id
    assert result.aggregates.tokens.total == 7
    assert (tmp_path / f"{result.run_id}.json").exists()


@pytest.mark.asyncio()
async def test_evaluate_reference_free_without_suite(tmp_path) -> None:
    source = InMemoryTraceSource([(TraceRef("only"), _trace("hello"))])

    result = await evaluate_traces(source, scorers=[has_one_response], store_dir=tmp_path)

    assert result.pass_rate("has_one_response") == 1.0
    assert len(result.tasks) == 1
    assert result.tasks[0].task.task_id == "only"


@pytest.mark.asyncio()
async def test_evaluate_records_budget_violation(tmp_path) -> None:
    source = InMemoryTraceSource([(TraceRef("big"), _trace("x", in_tok=100, out_tok=100))])

    result = await evaluate_traces(
        source, scorers=[], store_dir=tmp_path, budgets=BudgetThresholds(max_tokens_per_task=50)
    )

    assert result.aggregates.budget_violations == 1


@pytest.mark.asyncio()
async def test_json_object_answer_projects_structured_content(tmp_path) -> None:
    """A JSON-object final answer is parsed into outputs["content"] (mirrors reply.content())."""
    source = InMemoryTraceSource([
        (TraceRef("t1", task_id="task-1"), _trace('{"answer": "Paris", "confidence": 0.9}')),
    ])
    suite = Suite.from_list([
        {"task_id": "task-1", "inputs": {"input": "capital of France?"}, "reference_outputs": {"answer": "Paris"}},
    ])

    result = await evaluate_traces(
        source,
        # exact match only passes if "Paris" came from the parsed content, not the raw JSON text
        scorers=[answer_is_paris, final_answer_matches(field="answer", matcher="exact")],
        suite=suite,
        store_dir=tmp_path,
    )

    assert result.pass_rate("answer_is_paris") == 1.0
    assert result.pass_rate("final_answer_matches") == 1.0


@pytest.mark.asyncio()
async def test_free_text_answer_content_mirrors_body(tmp_path) -> None:
    """A non-JSON answer leaves content as the text itself (== body)."""
    source = InMemoryTraceSource([(TraceRef("t1", task_id="task-1"), _trace("Paris is the capital."))])

    result = await evaluate_traces(source, scorers=[free_text_content_mirrors_body], store_dir=tmp_path)

    assert result.pass_rate("free_text_content_mirrors_body") == 1.0
