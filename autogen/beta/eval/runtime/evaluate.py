# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""evaluate() — grade traces from a :class:`TraceSource`.

The trace-based counterpart to :func:`~autogen.beta.eval.run`. Where ``run``
executes an agent and captures its event stream, ``evaluate`` takes traces that
already exist — from a just-finished run, from disk, or from a cloud backend —
and grades them. Both funnel through one private grading core (:func:`_grade`),
so ``run(agent)`` and ``evaluate(the trace that agent produced)`` grade through
identical code — there is no second scoring path to drift.

``outputs`` is projected from the trace (the final model response's content),
so reference-based scorers like ``final_answer_matches`` work against a
reconstructed trace. ``reference_outputs``
come from the paired :class:`~autogen.beta.eval.Suite` task (via
``TraceRef.task_id``); traces with no paired task are graded reference-free.
"""

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime, timezone
from functools import partial
from typing import Any
from uuid import uuid4

from autogen.beta.context import ConversationContext
from autogen.beta.events import ModelResponse
from autogen.beta.stream import Stream

from .._types import Feedback
from ..dataset import Suite, Task
from ..events import EvalCompleted, EvalStarted, TaskEvaluated
from ..results import BudgetThresholds, RunResult, TaskResult
from ..scorer import Scorer
from ..sources import TraceRef, TraceSource
from ..trace import Trace

__all__ = ("evaluate",)

logger = logging.getLogger(__name__)


async def evaluate(
    source: TraceSource,
    *,
    scorers: Iterable[Scorer],
    store_dir: str | os.PathLike[str],
    suite: Suite | None = None,
    budgets: BudgetThresholds | None = None,
    concurrency: int = 4,
    run_id: str | None = None,
    label: str | None = None,
    stream: Stream | None = None,
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
        label: Optional user-defined identifier recorded on the run — meant to
            be *shared* across runs of the same eval so they can be grouped and
            trended. ``None`` if unset; the framework never fills it.
        stream: Optional :class:`~autogen.beta.stream.Stream` to publish eval
            lifecycle events to (``EvalStarted`` / ``TaskEvaluated`` /
            ``EvalCompleted``) — observe a grading pass like you observe an agent.
    """
    started = time.perf_counter()
    return await _grade(
        source,
        scorers=tuple(scorers),
        suite=suite,
        store_dir=store_dir,
        budgets=budgets,
        concurrency=concurrency,
        run_id=run_id,
        label=label,
        stream=stream,
        target_path=f"trace-source:{type(source).__name__}",
        started_at=started,
    )


async def _grade(
    source: TraceSource,
    *,
    scorers: tuple[Scorer, ...],
    suite: Suite | None,
    store_dir: str | os.PathLike[str],
    budgets: BudgetThresholds | None,
    concurrency: int,
    run_id: str | None,
    label: str | None,
    stream: Stream | None,
    target_path: str,
    started_at: float,
    variant: str | None = None,
) -> RunResult:
    """Grade every trace from ``source`` into a persisted :class:`RunResult`.

    The single grading path shared by :func:`evaluate` and :func:`~autogen.beta.eval.run`
    (the latter produces traces first, then grades them here). Projects ``outputs``
    from each trace, runs the scorers, applies budgets, aggregates, persists, and —
    when ``stream`` is set — publishes the lifecycle events. ``target_path`` records
    provenance (the agent for ``run``; the trace source for ``evaluate``);
    ``started_at`` is the caller's ``perf_counter`` start so the run-level duration
    spans the caller's whole operation.
    """
    refs = [ref async for ref in source.list()]
    tasks_by_id = {task.task_id: task for task in suite} if suite is not None else {}
    resolved_suite = suite if suite is not None else _suite_from_refs(refs)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    actual_run_id = run_id if run_id is not None else uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()

    eval_stream = stream
    eval_ctx = ConversationContext(stream=eval_stream) if eval_stream is not None else None
    if eval_stream is not None:
        await eval_stream.send(
            EvalStarted(run_id=actual_run_id, label=label, suite=resolved_suite.name, total=len(refs)),
            eval_ctx,
        )

    on_result = (
        partial(_publish_task_evaluated, eval_stream, eval_ctx, actual_run_id, label, variant)
        if eval_stream is not None
        else None
    )
    coros = [
        _evaluate_ref(
            semaphore, source, ref, scorers=scorers, tasks_by_id=tasks_by_id, budgets=budgets, on_result=on_result
        )
        for ref in refs
    ]
    task_results = await asyncio.gather(*coros)
    duration_ms = int((time.perf_counter() - started_at) * 1000)

    result = RunResult(
        run_id=actual_run_id,
        tasks=tuple(task_results),
        suite=resolved_suite,
        target_path=target_path,
        concurrency=max(1, concurrency),
        duration_ms=duration_ms,
        created_at=created_at,
        label=label,
        store_dir=store_dir,
    )
    saved_path = result.save()
    logger.info("Run %s saved to %s", actual_run_id, saved_path)

    if eval_stream is not None:
        await eval_stream.send(EvalCompleted(run_id=actual_run_id, label=label, result=result), eval_ctx)

    return result


async def _evaluate_ref(
    semaphore: asyncio.Semaphore,
    source: TraceSource,
    ref: TraceRef,
    *,
    scorers: tuple[Scorer, ...],
    tasks_by_id: dict[str, Task],
    budgets: BudgetThresholds | None,
    on_result: Callable[[Task, TaskResult], Awaitable[None]] | None = None,
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
        result = TaskResult(task=task, trace=trace, feedback=tuple(feedback), budget_violation=budget_violation)
        if on_result is not None:
            await on_result(task, result)
        return result


async def _publish_task_evaluated(
    stream: Stream,
    ctx: ConversationContext,
    run_id: str,
    label: str | None,
    variant: str | None,
    task: Task,
    result: TaskResult,
) -> None:
    """Publish a :class:`TaskEvaluated` event when one trace finishes scoring."""
    await stream.send(
        TaskEvaluated(run_id=run_id, label=label, task_id=task.task_id, feedback=result.feedback, variant=variant),
        ctx,
    )


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
