# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""evaluate() — grade traces from a :class:`TraceSource`.

The trace-based counterpart to :func:`~autogen.beta.eval.run`. Where ``run``
executes an agent and captures its event stream, ``evaluate`` takes traces that
already exist — from a just-finished run, from disk, or from a cloud backend —
and grades them. The two share everything downstream (``Scorer``, ``Trace``,
``RunResult``); the only difference is that ``evaluate`` never runs an agent.

``outputs`` is projected from the trace (the final model response's content)
rather than read from a live reply, so reference-based scorers like
``final_answer_matches`` work against a reconstructed trace. ``reference_outputs``
come from the paired :class:`~autogen.beta.eval.Suite` task (via
``TraceRef.task_id``); traces with no paired task are graded reference-free.
"""

import asyncio
import os
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from autogen.beta.events import ModelResponse

from .._types import Feedback
from ..dataset import Suite, Task
from ..results import BudgetThresholds, RunResult, TaskResult
from ..scorer import Scorer
from ..sources import TraceRef, TraceSource
from ..trace import Trace

__all__ = ("evaluate",)


async def evaluate(
    source: TraceSource,
    *,
    scorers: Iterable[Scorer],
    store_dir: str | os.PathLike[str],
    suite: Suite | None = None,
    budgets: BudgetThresholds | None = None,
    concurrency: int = 4,
    run_id: str | None = None,
) -> RunResult:
    """Grade every trace from ``source`` and persist a :class:`RunResult`.

    Args:
        source: Where the traces come from (in-memory, disk, or cloud).
        scorers: Scorer instances; each runs once per trace.
        store_dir: Directory the run JSON is written to as ``<run_id>.json``.
        suite: Optional dataset to join traces to by ``TraceRef.task_id`` for
            reference-based scorers. When omitted, a suite is synthesized from
            the traces and scoring is reference-free.
        budgets: Optional observational thresholds; violations are recorded,
            never aborting.
        concurrency: Max traces graded in parallel.
        run_id: Override for the auto-generated run id.
    """
    scorer_list = tuple(scorers)
    refs = [ref async for ref in source.list()]
    tasks_by_id = {task.task_id: task for task in suite} if suite is not None else {}
    semaphore = asyncio.Semaphore(max(1, concurrency))

    actual_run_id = run_id if run_id is not None else uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()

    coros = [
        _evaluate_ref(semaphore, source, ref, scorers=scorer_list, tasks_by_id=tasks_by_id, budgets=budgets)
        for ref in refs
    ]
    task_results = await asyncio.gather(*coros)
    duration_ms = int((time.perf_counter() - started) * 1000)

    result = RunResult(
        run_id=actual_run_id,
        tasks=tuple(task_results),
        suite=suite if suite is not None else _suite_from_refs(refs),
        target_factory_path=f"trace-source:{type(source).__name__}",
        concurrency=max(1, concurrency),
        duration_ms=duration_ms,
        created_at=created_at,
        store_dir=store_dir,
    )
    result.save()
    return result


async def _evaluate_ref(
    semaphore: asyncio.Semaphore,
    source: TraceSource,
    ref: TraceRef,
    *,
    scorers: tuple[Scorer, ...],
    tasks_by_id: dict[str, Task],
    budgets: BudgetThresholds | None,
) -> TaskResult:
    async with semaphore:
        trace = await source.load(ref)
        task = tasks_by_id.get(ref.task_id) if ref.task_id is not None else None
        if task is None:
            task = Task(task_id=ref.task_id or ref.trace_id, inputs={}, reference_outputs=None)

        outputs = _outputs_from_trace(trace)
        feedback: list[Feedback] = []
        for scorer in scorers:
            feedback.extend(
                await scorer(
                    inputs=task.inputs,
                    outputs=outputs,
                    reference_outputs=task.reference_outputs,
                    trace=trace,
                    task=task,
                )
            )

        budget_violation = budgets.exceeded_by(trace) if budgets is not None else False
        return TaskResult(task=task, trace=trace, feedback=tuple(feedback), budget_violation=budget_violation)


def _outputs_from_trace(trace: Trace) -> dict[str, Any]:
    """Project scorer ``outputs`` from a trace: the final model response's text."""
    responses = trace.events_of(ModelResponse)
    if responses and responses[-1].content is not None:
        return {"body": responses[-1].content}
    return {}


def _suite_from_refs(refs: list[TraceRef]) -> Suite:
    """Synthesize a reference-free Suite (one task per trace) when none is supplied."""
    tasks = tuple(Task(task_id=ref.task_id or ref.trace_id, inputs={}, reference_outputs=None) for ref in refs)
    return Suite(tasks, name="traces", source="trace-source")
