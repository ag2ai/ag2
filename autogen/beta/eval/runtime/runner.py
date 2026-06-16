# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""run_agent() — the eval runner.

``run_agent`` is *produce-then-grade*, and produces its trace **through OpenTelemetry** —
the same substrate :func:`~autogen.beta.eval.evaluate_traces` grades. Per task it builds a
fresh :class:`~autogen.beta.Agent`, runs it with a
:class:`~autogen.beta.middleware.builtin.telemetry.TelemetryMiddleware` exporting to an
**in-memory** span exporter, then reconstructs the :class:`~autogen.beta.eval.Trace`
from those spans via ``readable_spans_to_trace`` — exactly the span→Trace path the
trace-based sources use. So ``run_agent()`` and ``evaluate_traces()`` don't merely match; they share
the *same* reconstruction and the *same* grading core
(:func:`~autogen.beta.eval.runtime.evaluate._grade`) — one path, no drift.

Because the trace is reconstructed from spans, ``run_agent`` inherits the OTEL path's
fidelity (halt / tool-not-found are stream-only, so never spanned — see
``sources/_spans.py``). Failures that emit no spans (a factory that raises, or an
``ask`` that errors before the agent span starts) fall back to the caught exception so
they still surface. Tasks run in parallel up to ``concurrency``, bounded by an
:class:`asyncio.Semaphore`.
"""

import asyncio
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any
from uuid import uuid4

from autogen.beta.agent import Agent
from autogen.beta.config import ModelConfig
from autogen.beta.middleware.builtin import TelemetryMiddleware
from autogen.beta.stream import MemoryStream, Stream
from autogen.import_utils import optional_import_block, require_optional_import

with optional_import_block():
    from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ..dataset import Suite, Task
from ..pairwise import PairwiseComparator, PairwiseRunResult, evaluate_pairwise
from ..results import BudgetThresholds, RunResult
from ..scorer import Scorer
from ..sources import InMemoryTraceSource, TraceRef
from ..sources._otel import readable_spans_to_trace
from ..trace import Trace
from .evaluate import _grade

__all__ = (
    "run_agent",
    "run_pairwise",
)


async def run_agent(
    suite: str | Suite,
    *,
    agent: Agent,
    store_dir: str | os.PathLike[str] | None = None,
    scorers: Iterable[Scorer] = (),
    model_config: ModelConfig | dict[str, ModelConfig] | None = None,
    repeats: int = 1,
    budgets: BudgetThresholds | None = None,
    concurrency: int = 4,
    run_id: str | None = None,
    label: str | None = None,
    stream: Stream | None = None,
    variant: str | None = None,
    span_attributes: dict[str, str] | None = None,
    span_processors: "Sequence[SpanProcessor] | None" = None,
) -> RunResult:
    """Run an evaluation suite end-to-end.

    Each task gets a fresh :class:`~autogen.beta.Agent` built
    from ``agent`` (an instance, reused, or a factory), run with a
    :class:`~autogen.beta.middleware.builtin.telemetry.TelemetryMiddleware` exporting to an
    in-memory span exporter; the :class:`~autogen.beta.eval.Trace` is then reconstructed
    from those spans (per-task duration timed around the ``ask``) — the same span→Trace
    path the trace-based sources use. Those traces are graded through the same core
    :func:`~autogen.beta.eval.evaluate_traces` uses, and the run is persisted as
    ``<store_dir>/<run_id>.json``.

    Args:
        suite: A :class:`Suite`, a JSONL path, or an inline list of dict
            task records. Strings / paths are loaded via
            :meth:`Suite.from_jsonl`; lists are loaded via
            :meth:`Suite.from_list`.
        agent: The agent to evaluate — either an :class:`~autogen.beta.Agent`
            *instance*, reused for every task, or a *factory* callable that
            builds a fresh :class:`~autogen.beta.Agent` per task. A factory may
            take a keyword-only ``config`` parameter so the runner can inject
            per-task or global model configs (use a factory, not an instance,
            when you want per-task ``model_config``).
        scorers: Scorer instances (typically produced by ``@scorer``).
            Each is called once per task; the resulting feedback is
            recorded on the task's :class:`TaskResult`.
        store_dir: Directory under which the run JSON is persisted as
            ``<store_dir>/<run_id>.json``. Required — evals are
            comparison artifacts; a run that isn't persisted has no
            shelf life. Use ``tmp_path`` in tests, a repo directory
            for CI, or any path that fits your retention story.
        model_config: ``None`` to let the factory pick (its default),
            a single ``ModelConfig`` to use everywhere, or a
            ``dict[task_id, ModelConfig]`` for per-task configs (e.g.
            one ``TestConfig`` cassette per task).
        repeats: Run each task this many times (default ``1``) — for
            measuring consistency. ``pass_rate`` / ``score_stats`` pool
            all runs; with ``repeats > 1`` each run gets a distinct
            ``task_id`` suffix (``"<id>#1"``, ``"<id>#2"``, …).
        budgets: Optional :class:`BudgetThresholds`. Violations are
            recorded on each task's ``budget_violation`` flag but never
            abort the run.
        concurrency: Maximum number of tasks executed in parallel.
            Clamped to ``>= 1``.
        run_id: Override for the auto-generated UUID4 run id (unique per run).
        label: Optional user-defined identifier recorded on the run. Unlike
            ``run_id`` (unique per run), a ``label`` is meant to be *shared*
            across runs of the same eval, so a sequence of runs can be grouped
            and trended over time. ``None`` if unset; the framework never fills it.
        stream: Optional :class:`~autogen.beta.stream.Stream` to publish eval
            lifecycle events to (``EvalStarted`` / ``TaskEvaluated`` /
            ``EvalCompleted``) — observe a run like you observe an agent.
            Subscribe your own observer to render progress / a live view.
        variant: Tags this run's ``TaskEvaluated`` events with a variant name.
            Set by :func:`~autogen.beta.eval.run_variants` for each variant in a
            sweep; leave ``None`` for a standalone run.
        span_attributes: Extra attributes stamped on **every** span the agent
            emits during production (passed to ``TelemetryMiddleware``). The run
            is auto-seeded with ``ag2.eval.run_id`` and — when set —
            ``ag2.eval.variant`` / ``ag2.eval.label``; each task additionally
            gets ``ag2.eval.task_id``. Caller-supplied keys win on conflict, so
            you can scope spans for an external backend (e.g.
            ``{"ag2.org.id": org_id}``).
        span_processors: Optional OpenTelemetry ``SpanProcessor`` s attached to
            each task's tracer provider **in addition to** the internal
            in-memory exporter that grading reads. Use this to export the same
            spans to your own backend — e.g.
            ``[BatchSpanProcessor(OTLPSpanExporter(...))]``. Export is purely
            additive: the in-memory processor (the grading source) is never
            replaced, so grading output is identical with or without it.

    Returns:
        A :class:`RunResult` containing per-task results and metadata.
        The result has already been written to disk by the time this
        function returns.
    """
    resolved_suite = _resolve_suite(suite)
    tasks_to_run = _expand_repeats(resolved_suite, repeats)

    # Resolve run_id up front so it can be stamped on the produced spans; caller keys win.
    resolved_run_id = run_id if run_id is not None else uuid4().hex
    run_span_attributes = {
        "ag2.eval.run_id": resolved_run_id,
        **({"ag2.eval.variant": variant} if variant else {}),
        **({"ag2.eval.label": label} if label else {}),
        **(span_attributes or {}),
    }

    started = time.perf_counter()

    source = await _produce(
        tasks_to_run,
        agent,
        model_config=model_config,
        concurrency=concurrency,
        span_attributes=run_span_attributes,
        span_processors=span_processors,
    )

    suite_to_grade = Suite(tuple(tasks_to_run), name=resolved_suite.name, source=resolved_suite.source)
    return await _grade(
        source,
        scorers=tuple(scorers),
        suite=suite_to_grade,
        store_dir=store_dir,
        budgets=budgets,
        concurrency=concurrency,
        run_id=resolved_run_id,
        label=label,
        stream=stream,
        variant=variant,
        target_path=f"{type(agent).__module__}:{type(agent).__qualname__}",
        started_at=started,
    )


async def run_pairwise(
    suite: str | Suite,
    *,
    variant_a: Agent[Any],
    variant_b: Agent[Any],
    comparators: Iterable[PairwiseComparator],
    store_dir: str | os.PathLike[str] | None = None,
    model_config: ModelConfig | dict[str, ModelConfig] | None = None,
    variant_a_name: str = "A",
    variant_b_name: str = "B",
    concurrency: int = 4,
    run_id: str | None = None,
    label: str | None = None,
    stream: Stream | None = None,
) -> PairwiseRunResult:
    """Produce traces for two variants over a suite, then compare them.

    Convenience over :func:`~autogen.beta.eval.evaluate_pairwise`: runs each
    variant across the suite (capturing a :class:`Trace` per task,
    keyed by ``task_id``), then pairwise-compares the two sets. Mirrors how
    :func:`run_agent` is produce-then-:func:`~autogen.beta.eval.evaluate_traces` for one
    variant. For decoupled grading of pre-existing traces, call
    ``evaluate_pairwise`` directly.

    ``label`` is a shared identifier recorded on the result (like :func:`run_agent`);
    pass ``stream`` to observe ``PairwiseStarted`` / ``PairwiseCompared`` /
    ``PairwiseCompleted`` lifecycle events as the comparison runs.
    """
    resolved_suite = _resolve_suite(suite)
    source_a = await _produce(resolved_suite, variant_a, model_config=model_config, concurrency=concurrency)
    source_b = await _produce(resolved_suite, variant_b, model_config=model_config, concurrency=concurrency)
    return await evaluate_pairwise(
        source_a,
        source_b,
        comparators=comparators,
        variant_a=variant_a_name,
        variant_b=variant_b_name,
        suite=resolved_suite,
        store_dir=store_dir,
        concurrency=concurrency,
        run_id=run_id,
        label=label,
        stream=stream,
    )


async def _produce(
    tasks: Iterable[Task],
    agent: Agent[Any],
    *,
    model_config: ModelConfig | dict[str, ModelConfig] | None,
    concurrency: int,
    span_attributes: dict[str, str] | None = None,
    span_processors: "Sequence[SpanProcessor] | None" = None,
) -> InMemoryTraceSource:
    """Run ``factory``'s agent across ``tasks``, returning one Trace per task (in order)."""
    tasks = list(tasks)
    missing = [t.task_id for t in tasks if "input" not in t.inputs]
    if missing:
        raise ValueError(
            f"run_agent asks the agent inputs['input'], but these task(s) have no 'input' key: {missing}. "
            'Add an "input" to each (use "" for an intentionally empty prompt).'
        )
    semaphore = asyncio.Semaphore(max(1, concurrency))
    produced = await asyncio.gather(
        *(
            _produce_one(
                semaphore,
                task,
                agent,
                model_config,
                span_attributes=span_attributes,
                span_processors=span_processors,
            )
            for task in tasks
        )
    )
    return InMemoryTraceSource(produced)


@require_optional_import("opentelemetry.sdk", "tracing")
async def _produce_one(
    semaphore: asyncio.Semaphore,
    task: Task,
    agent: Agent,
    model_config: ModelConfig | dict[str, ModelConfig] | None,
    *,
    span_attributes: dict[str, str] | None = None,
    span_processors: "Sequence[SpanProcessor] | None" = None,
) -> tuple[TraceRef, Trace]:
    async with semaphore:
        config = _resolve_task_config(task, model_config)
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        # In-memory processor is the grading source; caller processors are additive, never replace it.
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        for processor in span_processors or ():
            provider.add_span_processor(processor)
        task_attributes = {**(span_attributes or {}), "ag2.eval.task_id": task.task_id}
        stream = MemoryStream()
        exception: BaseException | None = None
        duration_ms = 0

        telemetry = TelemetryMiddleware(
            tracer_provider=provider,
            agent_name=agent.name,
            capture_content=True,
            span_attributes=task_attributes,
        )
        started = time.perf_counter()
        try:
            await agent.ask(
                task.inputs["input"],
                stream=stream,
                middleware=[telemetry],
                # passed config overloads any original one
                config=config,
            )
        except Exception as exc:
            exception = exc
        finally:
            duration_ms = int((time.perf_counter() - started) * 1000)

        # run_agent()'s Trace is reconstructed from the emitted spans — the same span→Trace path
        # evaluate_traces() uses, so the two grade identically. Pre-span failures (factory raise, or
        # ask erroring before the agent span starts) emit no spans → fall back to the caught exception.
        spans = exporter.get_finished_spans()
        # Real root-span trace id, for deep-linking to an external backend; synthetic fallback if no spans.
        otel_trace_id = format(spans[0].context.trace_id, "032x") if spans else uuid4().hex
        trace = readable_spans_to_trace(spans, duration_ms=duration_ms)
        if trace.exception is None and exception is not None:
            trace = Trace(events=trace.events, exception=exception, duration_ms=duration_ms)
        return (TraceRef(trace_id=otel_trace_id, task_id=task.task_id), trace)


def _resolve_suite(suite: str | Suite) -> Suite:
    """Normalize the ``suite`` argument into a :class:`Suite` instance."""
    if isinstance(suite, Suite):
        return suite
    return Suite(msg=suite)


def _expand_repeats(suite: Suite, repeats: int) -> list[Task]:
    """Expand each task into ``repeats`` runs with distinct ids (consistency sugar)."""
    if repeats <= 1:
        return list(suite)
    return [replace(task, task_id=f"{task.task_id}#{i + 1}") for task in suite for i in range(repeats)]


def _resolve_task_config(
    task: Task,
    model_config: ModelConfig | dict[str, ModelConfig] | None,
) -> ModelConfig | None:
    """Pick the right ``ModelConfig`` for a task from the ``model_config`` argument."""
    if model_config is None:
        return None
    if isinstance(model_config, dict):
        return model_config.get(task.task_id)
    return model_config
