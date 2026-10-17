# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from ag2 import Context
from ag2.eval import Feedback, InMemoryTraceSource, Suite, Trace, TraceRef, evaluate_traces, load_run, scorer
from ag2.eval.events import TaskEvaluated
from ag2.events import ModelMessage, ModelResponse
from ag2.stream import MemoryStream


class FailingTraceSource:
    async def list(self) -> AsyncIterator[TraceRef]:
        yield TraceRef("bad", task_id="first")
        yield TraceRef("good", task_id="second")

    async def load(self, ref: TraceRef) -> Trace:
        if ref.trace_id == "bad":
            raise OSError("trace unavailable")
        return Trace(events=[ModelResponse(message=ModelMessage("done"))], exception=None, duration_ms=1)


@scorer
def trace_available(trace: Trace) -> bool:
    return trace.exception is None


@pytest.mark.asyncio
@pytest.mark.parametrize("with_suite", [False, True])
async def test_load_failure_is_graded_and_published(tmp_path: Path, with_suite: bool) -> None:
    suite = Suite.from_list([
        {
            "task_id": "first",
            "inputs": {"input": "First task"},
            "tags": ["critical"],
            "reference_outputs": {"answer": "yes"},
        },
        {"task_id": "second", "inputs": {"input": "Second task"}},
    ])
    stream = MemoryStream()
    published: list[str] = []

    async def record(event: TaskEvaluated, context: Context) -> None:
        published.append(event.task_id)

    with stream.where(TaskEvaluated).sub_scope(record):
        result = await evaluate_traces(
            FailingTraceSource(),
            scorers=[trace_available],
            suite=suite if with_suite else None,
            store_dir=tmp_path,
            stream=stream,
        )

    assert result.pass_rate("trace_available") == 0.5
    assert result.aggregates.errors == 1
    assert result.tasks[0].feedback == (Feedback(key="trace_available", score=False),)
    assert result.tasks[0].trace_ref == TraceRef("bad", task_id="first")
    if with_suite:
        assert result.tasks[0].task == suite.tasks[0]
        assert result.pass_rate("trace_available", tag="critical") == 0.0
    assert sorted(published) == ["first", "second"]
    loaded = load_run(tmp_path / f"{result.run_id}.json")
    assert loaded.tasks[0].task == result.tasks[0].task
    assert loaded.pass_rate("trace_available") == 0.5


@pytest.mark.asyncio
async def test_load_failure_does_not_change_task_identity_for_run_comparison(tmp_path: Path) -> None:
    suite = Suite.from_list([
        {"task_id": "first", "inputs": {"input": "First task"}},
        {"task_id": "second", "inputs": {"input": "Second task"}},
    ])
    source = InMemoryTraceSource([
        (TraceRef(name, task_id=task.task_id), Trace(events=[], exception=None, duration_ms=1))
        for name, task in zip(("bad", "good"), suite, strict=True)
    ])
    baseline = await evaluate_traces(source, scorers=[trace_available], suite=suite, store_dir=tmp_path)
    current = await evaluate_traces(FailingTraceSource(), scorers=[trace_available], suite=suite, store_dir=tmp_path)

    assert current.diff(baseline).regressions == (("trace_available", "first"),)
