# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for failure attribution (deterministic detectors + LLM attributor)."""

import pytest

from ag2.eval import InMemoryTraceSource, TraceRef, evaluate_traces
from ag2.eval.dataset.task import Task
from ag2.eval.scorers import failure_attribution
from ag2.eval.trace import Trace
from ag2.events import ModelMessage, ModelResponse, ToolCallEvent, ToolErrorEvent
from ag2.testing import TestConfig


def _trace(events: list, *, exception: BaseException | None = None) -> Trace:
    return Trace(events=events, exception=exception, duration_ms=0)


async def _attribute(scorer, trace: Trace, *, outputs: dict | None = None) -> list:
    return await scorer(
        inputs={"input": "Q?"},
        outputs=outputs or {},
        reference_outputs=None,
        trace=trace,
        task=Task(task_id="t", inputs={}),
    )


@pytest.mark.asyncio()
async def test_crash() -> None:
    [fb] = await _attribute(failure_attribution(), _trace([], exception=ValueError("boom")))
    assert fb.value == "crash"
    assert fb.detail["failed"] is True


@pytest.mark.asyncio()
async def test_no_answer() -> None:
    [fb] = await _attribute(failure_attribution(), _trace([ToolCallEvent("t", arguments="{}")]))
    assert fb.value == "no_answer"


@pytest.mark.asyncio()
async def test_tool_failure_records_decisive_step() -> None:
    call = ToolCallEvent("flaky", arguments="{}")
    err = ToolErrorEvent.from_call(call, RuntimeError("kaboom"))
    [fb] = await _attribute(failure_attribution(), _trace([call, err]))
    assert fb.value == "tool_failure"
    assert fb.detail["decisive_step"] == 1  # index of the error event


@pytest.mark.asyncio()
async def test_clean_run_without_llm_is_none() -> None:
    [fb] = await _attribute(
        failure_attribution(), _trace([ModelResponse(message=ModelMessage("done"))]), outputs={"body": "done"}
    )
    assert fb.value == "none"
    assert fb.detail["failed"] is False


@pytest.mark.asyncio()
async def test_llm_attributor_for_semantic_failure() -> None:
    # no mechanical failure (has an answer) -> defer to the LLM attributor
    config = TestConfig(
        '{"failed": true, "error_mode": "incorrect_answer", "decisive_step": 0, "reasoning": "wrong city"}'
    )
    scorer = failure_attribution(config, key="failure")
    [fb] = await _attribute(scorer, _trace([ModelResponse(message=ModelMessage("Lyon"))]), outputs={"body": "Lyon"})
    assert fb.value == "incorrect_answer"
    assert fb.detail["failed"] is True
    assert fb.detail["decisive_step"] == 0


@pytest.mark.asyncio()
async def test_failure_mode_distribution_via_value_counts(tmp_path) -> None:
    crash = _trace([], exception=ValueError("x"))
    clean = _trace([ModelResponse(message=ModelMessage("ok"))])
    source = InMemoryTraceSource([(TraceRef("t1", task_id="t1"), crash), (TraceRef("t2", task_id="t2"), clean)])

    result = await evaluate_traces(source, scorers=[failure_attribution(key="failure")], store_dir=tmp_path)

    assert result.value_counts("failure") == {"crash": 1, "none": 1}
