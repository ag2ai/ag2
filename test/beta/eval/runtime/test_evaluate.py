# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluate() — grading traces from a TraceSource."""

import pytest

from autogen.beta.eval import (
    BudgetThresholds,
    InMemoryTraceSource,
    Suite,
    TraceRef,
    evaluate,
    scorer,
)
from autogen.beta.eval.scorers import final_answer_matches, tool_called
from autogen.beta.eval.trace import Trace
from autogen.beta.events import ModelMessage, ModelResponse, ToolCallEvent, Usage


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


@pytest.mark.asyncio()
async def test_evaluate_scores_persists_and_joins_reference(tmp_path) -> None:
    source = InMemoryTraceSource([
        (TraceRef("t1", task_id="task-1"), _trace("Paris", tool_name="get_weather", in_tok=5, out_tok=2)),
    ])
    suite = Suite.from_list([
        {"task_id": "task-1", "inputs": {"input": "capital of France?"}, "reference_outputs": {"answer": "Paris"}},
    ])

    result = await evaluate(
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

    result = await evaluate(source, scorers=[has_one_response], store_dir=tmp_path)

    assert result.pass_rate("has_one_response") == 1.0
    assert len(result.tasks) == 1
    assert result.tasks[0].task.task_id == "only"


@pytest.mark.asyncio()
async def test_evaluate_records_budget_violation(tmp_path) -> None:
    source = InMemoryTraceSource([(TraceRef("big"), _trace("x", in_tok=100, out_tok=100))])

    result = await evaluate(source, scorers=[], store_dir=tmp_path, budgets=BudgetThresholds(max_tokens_per_task=50))

    assert result.aggregates.budget_violations == 1
