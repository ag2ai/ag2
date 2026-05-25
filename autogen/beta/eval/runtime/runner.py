# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""run() — the eval runner.

The runner builds a fresh :class:`~autogen.beta.eval.EvalTarget` per task
via the user-supplied factory, attaches the framework's
:class:`~autogen.beta.eval.runtime._capture.EventCapture` observer, and calls
``target.ask(task.inputs["input"], stream=stream, observers=[capture])``.
The observer rides on the same extension point users pass their own
observers through, so the runner composes with rather than replaces user
setup.

After the turn finishes, the captured events plus wall-clock duration
and the final reply (or exception) are bundled into a
:class:`~autogen.beta.eval.Trace`, the scorers run over the trace, and a
per-task :class:`~autogen.beta.eval.TaskResult` is produced. Tasks run
in parallel up to ``concurrency``, bounded by an :class:`asyncio.Semaphore`.
"""

import asyncio
import inspect
import logging
import os
import time
import warnings
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from autogen.beta.config import ModelConfig
from autogen.beta.stream import MemoryStream

from ..dataset import EvalTarget, Suite, Task
from ..pairwise import PairwiseComparator, PairwiseRunResult, evaluate_pairwise
from ..results import BudgetThresholds, RunResult, TaskResult
from ..scorer import Scorer
from ..sources import InMemoryTraceSource, TraceRef
from ..trace import Trace
from ._capture import EventCapture

__all__ = ("run", "run_pairwise")


logger = logging.getLogger(__name__)


async def run(
    suite: Suite | str | os.PathLike[str] | list[dict[str, Any]],
    *,
    target_factory: Callable[..., EvalTarget],
    scorers: Iterable[Scorer],
    store_dir: str | os.PathLike[str],
    model_config: ModelConfig | dict[str, ModelConfig] | None = None,
    budgets: BudgetThresholds | None = None,
    concurrency: int = 4,
    run_id: str | None = None,
) -> RunResult:
    """Run an evaluation suite end-to-end.

    Each task gets a fresh :class:`~autogen.beta.eval.EvalTarget` built
    via ``target_factory``. Events emitted during ``target.ask(...)`` are
    captured through an Observer registered alongside any user observers
    the target uses internally. Wall-clock duration is measured around
    the whole ``ask`` call. Scorers then run over the resulting Trace,
    and the run is persisted as ``<store_dir>/<run_id>.json``.

    Args:
        suite: A :class:`Suite`, a JSONL path, or an inline list of dict
            task records. Strings / paths are loaded via
            :meth:`Suite.from_jsonl`; lists are loaded via
            :meth:`Suite.from_list`.
        target_factory: Callable that returns a fresh
            :class:`~autogen.beta.eval.EvalTarget` — most often an
            :class:`~autogen.beta.Agent`, but any async-``.ask(prompt, *,
            stream, observers)`` shape qualifies. Should accept a
            keyword-only ``config`` parameter so the runner can inject
            per-task or global model configs; factories without
            ``config`` are called with no arguments (with a warning).
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
        budgets: Optional :class:`BudgetThresholds`. Violations are
            recorded on each task's ``budget_violation`` flag but never
            abort the run.
        concurrency: Maximum number of tasks executed in parallel.
            Clamped to ``>= 1``.
        run_id: Override for the auto-generated UUID4 run id.

    Returns:
        A :class:`RunResult` containing per-task results and metadata.
        The result has already been written to disk by the time this
        function returns.
    """
    resolved_suite = _resolve_suite(suite)
    scorer_list = tuple(scorers)
    accepts_config = _factory_accepts_config(target_factory)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    actual_run_id = run_id if run_id is not None else uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    run_started = time.perf_counter()

    coros = [
        _execute_with_limit(
            semaphore,
            task,
            target_factory=target_factory,
            accepts_config=accepts_config,
            model_config=model_config,
            scorers=scorer_list,
            budgets=budgets,
        )
        for task in resolved_suite
    ]
    task_results = await asyncio.gather(*coros)

    duration_ms = int((time.perf_counter() - run_started) * 1000)

    result = RunResult(
        run_id=actual_run_id,
        tasks=tuple(task_results),
        suite=resolved_suite,
        target_factory_path=_factory_path(target_factory),
        concurrency=max(1, concurrency),
        duration_ms=duration_ms,
        created_at=created_at,
        store_dir=store_dir,
    )

    saved_path = result.save()
    logger.info("Run %s saved to %s", actual_run_id, saved_path)

    return result


def _resolve_suite(
    suite: Suite | str | os.PathLike[str] | list[dict[str, Any]],
) -> Suite:
    """Normalize the ``suite`` argument into a :class:`Suite` instance."""
    if isinstance(suite, Suite):
        return suite
    if isinstance(suite, list):
        return Suite.from_list(suite)
    return Suite.from_jsonl(suite)


def _factory_accepts_config(factory: Callable[..., EvalTarget]) -> bool:
    """Detect whether the factory takes a ``config`` parameter.

    A bare factory like ``def build() -> Agent`` is supported — the
    runner will call it with no args and skip injecting model_config.
    """
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    return "config" in sig.parameters


def _factory_path(factory: Callable[..., EvalTarget]) -> str:
    """Return ``"<module>:<qualname>"`` for the factory, for the run JSON."""
    module = getattr(factory, "__module__", "<unknown>")
    qualname = getattr(factory, "__qualname__", getattr(factory, "__name__", "<unknown>"))
    return f"{module}:{qualname}"


async def _execute_with_limit(
    semaphore: asyncio.Semaphore,
    task: Task,
    **kwargs: Any,
) -> TaskResult:
    async with semaphore:
        return await _execute_task(task, **kwargs)


async def _execute_task(
    task: Task,
    *,
    target_factory: Callable[..., EvalTarget],
    accepts_config: bool,
    model_config: ModelConfig | dict[str, ModelConfig] | None,
    scorers: tuple[Scorer, ...],
    budgets: BudgetThresholds | None,
) -> TaskResult:
    """Execute one task and return its :class:`TaskResult`.

    Construction-time and ``target.ask`` exceptions are both captured —
    neither aborts the run. They surface on ``trace.exception`` for
    scorers and reports to consume.
    """
    capture = EventCapture()
    config = _resolve_task_config(task, model_config)

    stream = MemoryStream()
    reply = None
    exception: BaseException | None = None
    duration_ms = 0

    try:
        target = _build_target(target_factory, accepts_config=accepts_config, config=config)
    except Exception as exc:
        logger.warning("target_factory raised for task %r: %s", task.task_id, exc)
        exception = exc
    else:
        prompt = task.inputs.get("input", "")
        # Time around ``target.ask`` (not via middleware) so any internal
        # ``reply.ask`` continuations the target drives are included in
        # the total task duration.
        task_started = time.perf_counter()
        try:
            reply = await target.ask(
                prompt,
                stream=stream,
                observers=[capture],
            )
        except Exception as exc:
            logger.warning("target.ask raised for task %r: %s", task.task_id, exc)
            exception = exc
        finally:
            duration_ms = int((time.perf_counter() - task_started) * 1000)

    trace = Trace(
        events=capture.events,
        reply=reply,
        exception=exception,
        duration_ms=duration_ms,
    )
    outputs = _outputs_from_reply(reply)

    feedback: list = []
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

    budget_violation = _exceeds_budget(trace, budgets)

    return TaskResult(
        task=task,
        trace=trace,
        feedback=tuple(feedback),
        budget_violation=budget_violation,
    )


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


def _build_target(
    factory: Callable[..., EvalTarget],
    *,
    accepts_config: bool,
    config: ModelConfig | None,
) -> EvalTarget:
    """Call the user's factory, threading ``config`` only if it accepts one."""
    if not accepts_config:
        if config is not None:
            warnings.warn(
                "target_factory does not accept a 'config' parameter; model_config will be ignored for this run.",
                category=RuntimeWarning,
                stacklevel=3,
            )
        return factory()
    return factory(config=config)


def _outputs_from_reply(reply: Any) -> dict[str, Any]:
    """Normalize the target's reply into the dict scorers receive as ``outputs``.

    ``reply.response`` is used if present (structured output mode), else
    ``{"body": reply.body}``. When the run raised, an empty dict is used
    so reference-free scorers can still inspect ``trace``.
    """
    if reply is None:
        return {}
    response = getattr(reply, "response", None)
    if isinstance(response, dict):
        return response
    body = getattr(reply, "body", None)
    return {"body": body}


def _exceeds_budget(trace: Trace, budgets: BudgetThresholds | None) -> bool:
    """Apply the :class:`BudgetThresholds` checks to one task's trace."""
    return budgets.exceeded_by(trace) if budgets is not None else False


async def run_pairwise(
    suite: Suite | str | os.PathLike[str] | list[dict[str, Any]],
    *,
    variant_a: Callable[..., EvalTarget],
    variant_b: Callable[..., EvalTarget],
    comparators: Iterable[PairwiseComparator],
    store_dir: str | os.PathLike[str],
    model_config: ModelConfig | dict[str, ModelConfig] | None = None,
    variant_a_name: str = "A",
    variant_b_name: str = "B",
    concurrency: int = 4,
    run_id: str | None = None,
) -> PairwiseRunResult:
    """Produce traces for two variants over a suite, then compare them.

    Convenience over :func:`~autogen.beta.eval.evaluate_pairwise`: runs each
    variant factory across the suite (capturing a :class:`Trace` per task,
    keyed by ``task_id``), then pairwise-compares the two sets. Mirrors how
    :func:`run` is produce-then-:func:`~autogen.beta.eval.evaluate` for one
    variant. For decoupled grading of pre-existing traces, call
    ``evaluate_pairwise`` directly.
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
    )


async def _produce(
    suite: Suite,
    factory: Callable[..., EvalTarget],
    *,
    model_config: ModelConfig | dict[str, ModelConfig] | None,
    concurrency: int,
) -> InMemoryTraceSource:
    """Run ``factory``'s agent across the suite, returning one Trace per task."""
    accepts_config = _factory_accepts_config(factory)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    produced = await asyncio.gather(
        *(_produce_one(semaphore, task, factory, accepts_config, model_config) for task in suite)
    )
    return InMemoryTraceSource(produced)


async def _produce_one(
    semaphore: asyncio.Semaphore,
    task: Task,
    factory: Callable[..., EvalTarget],
    accepts_config: bool,
    model_config: ModelConfig | dict[str, ModelConfig] | None,
) -> tuple[TraceRef, Trace]:
    async with semaphore:
        config = _resolve_task_config(task, model_config)
        capture = EventCapture()
        stream = MemoryStream()
        reply = None
        exception: BaseException | None = None
        duration_ms = 0
        try:
            target = _build_target(factory, accepts_config=accepts_config, config=config)
        except Exception as exc:
            exception = exc
        else:
            started = time.perf_counter()
            try:
                reply = await target.ask(task.inputs.get("input", ""), stream=stream, observers=[capture])
            except Exception as exc:
                exception = exc
            finally:
                duration_ms = int((time.perf_counter() - started) * 1000)
        trace = Trace(events=capture.events, reply=reply, exception=exception, duration_ms=duration_ms)
        return (TraceRef(trace_id=uuid4().hex, task_id=task.task_id), trace)
