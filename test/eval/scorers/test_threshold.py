# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the threshold combinator (``ag2.eval.scorers.threshold``)."""

import pytest
from dirty_equals import IsPartialDict

from ag2.eval import InMemoryTraceSource, Suite, TraceRef, evaluate_traces, scorer
from ag2.eval.dataset.task import Task
from ag2.eval.scorers import agent_judge, threshold
from ag2.eval.trace import Trace
from ag2.events import ModelMessage, ModelResponse
from ag2.testing import TestConfig


def _empty_trace() -> Trace:
    return Trace(events=[], exception=None, duration_ms=0)


async def _run(s, *, outputs, inputs=None, reference_outputs=None, trace=None) -> list:
    return await s(
        inputs=inputs or {},
        outputs=outputs,
        reference_outputs=reference_outputs,
        trace=trace if trace is not None else _empty_trace(),
        task=Task(task_id="t", inputs={}),
    )


@scorer
def raw(outputs) -> float:
    """Toy numeric scorer that echoes a score from outputs."""
    return outputs["score"]


@scorer
def already_bool(outputs) -> bool:
    return outputs["ok"]


@scorer
def category(outputs) -> str:
    return outputs["label"]


@pytest.mark.asyncio()
async def test_pass_at_or_above_lower_bound() -> None:
    gate = threshold(raw, at_least=0.7)

    [fb] = await _run(gate, outputs={"score": 0.8})

    assert fb.key == "raw"
    assert fb.score is True
    assert fb.detail == IsPartialDict({"score": 0.8, "at_least": 0.7})


@pytest.mark.asyncio()
async def test_fail_below_lower_bound_keeps_the_number() -> None:
    gate = threshold(raw, at_least=0.7)

    [fb] = await _run(gate, outputs={"score": 0.6})

    assert fb.score is False
    assert fb.detail == IsPartialDict({"score": 0.6})  # raw number preserved in detail


@pytest.mark.asyncio()
async def test_lower_bound_is_inclusive() -> None:
    gate = threshold(raw, at_least=0.7)

    [fb] = await _run(gate, outputs={"score": 0.7})

    assert fb.score is True


@pytest.mark.asyncio()
async def test_dual_bounds_are_inclusive_on_both_ends() -> None:
    gate = threshold(raw, at_least=0.3, at_most=0.7)

    assert (await _run(gate, outputs={"score": 0.5}))[0].score is True
    assert (await _run(gate, outputs={"score": 0.2}))[0].score is False
    assert (await _run(gate, outputs={"score": 0.9}))[0].score is False
    assert (await _run(gate, outputs={"score": 0.3}))[0].score is True
    assert (await _run(gate, outputs={"score": 0.7}))[0].score is True


@pytest.mark.asyncio()
async def test_custom_key() -> None:
    gate = threshold(raw, at_least=0.7, key="raw_pass")

    [fb] = await _run(gate, outputs={"score": 0.9})

    assert fb.key == "raw_pass"


@pytest.mark.asyncio()
async def test_no_bound_raises_at_construction() -> None:
    with pytest.raises(ValueError):
        threshold(raw)


@pytest.mark.asyncio()
async def test_already_boolean_feedback_passes_through_unchanged() -> None:
    gate = threshold(already_bool, at_least=0.7)

    [fb] = await _run(gate, outputs={"ok": True})

    assert fb.score is True  # not re-thresholded
    assert fb.detail is None  # untouched


@pytest.mark.asyncio()
async def test_categorical_feedback_passes_through_unchanged() -> None:
    gate = threshold(category, at_least=0.7)

    [fb] = await _run(gate, outputs={"label": "timeout"})

    assert fb.value == "timeout"
    assert fb.score is None


@pytest.mark.asyncio()
async def test_ungradeable_is_a_fail() -> None:
    judge = agent_judge(TestConfig("not valid json"), criterion="x", key="quality", retries=0)
    gate = threshold(judge, at_least=0.7)

    [fb] = await _run(gate, outputs={"body": "a"})

    assert fb.key == "quality"
    assert fb.score is False  # judge could not grade -> fail


@pytest.mark.asyncio()
async def test_agent_judge_threshold_param_emits_pass_fail() -> None:
    judge = agent_judge(
        TestConfig('{"score": 0.6, "reasoning": "meh"}'),
        criterion="quality",
        key="quality",
        threshold=0.7,
    )

    [fb] = await _run(judge, inputs={"input": "q"}, outputs={"body": "a"})

    assert fb.key == "quality"
    assert fb.score is False
    assert fb.detail == IsPartialDict({"score": 0.6})


@pytest.mark.asyncio()
async def test_agent_judge_without_threshold_stays_numeric() -> None:
    judge = agent_judge(TestConfig('{"score": 0.6, "reasoning": "meh"}'), criterion="quality", key="quality")

    [fb] = await _run(judge, inputs={"input": "q"}, outputs={"body": "a"})

    assert fb.score == 0.6  # unchanged numeric grade


@pytest.mark.asyncio()
async def test_gate_lands_in_pass_rate_and_records_number(tmp_path) -> None:
    trace = Trace(events=[ModelResponse(message=ModelMessage("ans"))], exception=None, duration_ms=1)
    source = InMemoryTraceSource([(TraceRef("t1", task_id="task-1"), trace)])
    suite = Suite.from_list([{"task_id": "task-1", "inputs": {"input": "q"}}])
    judge = agent_judge(
        TestConfig('{"score": 0.6, "reasoning": "meh"}'),
        criterion="quality",
        key="quality",
        threshold=0.7,
    )

    result = await evaluate_traces(source, scorers=[judge], suite=suite, store_dir=tmp_path)

    assert result.pass_rate("quality") == 0.0  # 0.6 < 0.7 -> fail -> gates into pass_rate
    assert result.tasks[0].feedback[0].detail["score"] == 0.6  # raw number recorded per task
