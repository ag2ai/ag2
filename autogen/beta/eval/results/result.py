# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""RunResult, TaskResult, Aggregates, ScoreStats — what :func:`run_agent` hands back.

Aggregation rules:

* **pass_rate** — fraction of ``True`` outcomes among feedback whose ``score``
  is a ``bool``. Computed per scorer key. Numeric scores never contribute
  to pass-rate.
* **score_stats** — ``mean`` / ``p50`` / ``p95`` / ``n`` over numeric scores
  (``int`` / ``float``, excluding ``bool``). Per scorer key.
* **value_counts** — categorical label frequencies, per scorer key.
* **tokens** — input/output totals summed across every task's Trace.
* **errors** — count of tasks where ``trace.exception is not None``.
* **budget_violations** — count of tasks where ``budget_violation`` is set.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .._types import Feedback
from ..dataset import Suite, Task
from ..sources.trace_source import TraceRef
from ..trace import TokenUsage, Trace

if TYPE_CHECKING:
    from .diff import RunDiff

__all__ = (
    "Aggregates",
    "RunResult",
    "ScoreStats",
    "TaskResult",
)


_SCHEMA_VERSION = "0.1"


@dataclass(frozen=True, slots=True)
class TaskResult:
    """One task's outcome — definition, Trace, scorer feedback, budget status."""

    task: Task
    trace: Trace
    feedback: tuple[Feedback, ...]
    budget_violation: bool = False
    trace_ref: TraceRef | None = None


@dataclass(frozen=True, slots=True)
class ScoreStats:
    """Summary statistics for a numeric scorer across all tasks."""

    mean: float
    p50: float
    p95: float
    n: int


@dataclass(frozen=True, slots=True)
class Aggregates:
    """Run-level rollups over all per-task :class:`TaskResult` records."""

    pass_rate: dict[str, float]
    score_stats: dict[str, ScoreStats]
    value_counts: dict[str, dict[str, int]]
    tokens: TokenUsage
    errors: int
    budget_violations: int


class RunResult:
    """The result of a full :func:`run_agent`.

    Holds per-task records, run-level metadata, and computed aggregates.
    Lookup helpers (``pass_rate``, ``score_stats``, ``value_counts``)
    surface single keys; :meth:`summary` renders a printable table;
    :meth:`save` writes the schema-0.1 JSON.

    Aggregates are computed once at construction time so repeated lookups
    are cheap.
    """

    __slots__ = (
        "_run_id",
        "_tasks",
        "_suite",
        "_target_path",
        "_concurrency",
        "_duration_ms",
        "_created_at",
        "_label",
        "_store_dir",
        "_aggregates",
        "_sliced",
    )

    def __init__(
        self,
        *,
        run_id: str,
        tasks: tuple[TaskResult, ...],
        suite: Suite,
        target_path: str,
        concurrency: int,
        duration_ms: int,
        created_at: str,
        label: str | None = None,
        store_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self._run_id = run_id
        self._tasks = tasks
        self._suite = suite
        self._target_path = target_path
        self._concurrency = concurrency
        self._duration_ms = duration_ms
        self._created_at = created_at
        self._label = label
        self._store_dir = Path(store_dir) if store_dir is not None else None
        self._aggregates = _compute_aggregates(tasks)
        self._sliced: dict[str, Aggregates] = {}

    @property
    def run_id(self) -> str:
        """Stable identifier for this run — UUID4 hex unless the caller passed one."""
        return self._run_id

    @property
    def schema_version(self) -> str:
        """Run JSON schema version. Always ``"0.1"`` in v0."""
        return _SCHEMA_VERSION

    @property
    def tasks(self) -> tuple[TaskResult, ...]:
        """Per-task records, in suite order."""
        return self._tasks

    @property
    def suite(self) -> Suite:
        """The Suite that was executed."""
        return self._suite

    @property
    def target_path(self) -> str:
        """``"<module>:<name>"`` provenance of the evaluated agent (its instance type)."""
        return self._target_path

    @property
    def concurrency(self) -> int:
        """Concurrency cap the runner used (``asyncio.Semaphore`` bound)."""
        return self._concurrency

    @property
    def duration_ms(self) -> int:
        """Wall-clock duration of the full run, in milliseconds."""
        return self._duration_ms

    @property
    def created_at(self) -> str:
        """ISO-8601 UTC timestamp of when this run started."""
        return self._created_at

    @property
    def label(self) -> str | None:
        """User-defined identifier grouping runs of the same eval over time (``None`` if unset)."""
        return self._label

    @property
    def aggregates(self) -> Aggregates:
        """Run-level rollups computed from the per-task feedback and traces."""
        return self._aggregates

    @property
    def tags(self) -> frozenset[str]:
        """Every tag present across the run's tasks — the values usable in ``tag=`` lookups."""
        return frozenset(tag for tr in self._tasks for tag in tr.task.tags)

    def pass_rate(self, key: str, *, tag: str | None = None) -> float:
        """Pass rate for a boolean scorer (``0.0`` if no boolean feedback under ``key``).

        Pass ``tag`` to compute it over only the tasks carrying that tag — e.g.
        ``pass_rate("tool_called", tag="adversarial")``. Unset slices the whole run.
        """
        return self._agg(tag).pass_rate.get(key, 0.0)

    def score_stats(self, key: str, *, tag: str | None = None) -> ScoreStats:
        """Numeric stats for a scorer (zeros when nothing numeric under ``key``).

        Pass ``tag`` to restrict to tasks carrying that tag.
        """
        return self._agg(tag).score_stats.get(key, ScoreStats(mean=0.0, p50=0.0, p95=0.0, n=0))

    def value_counts(self, key: str, *, tag: str | None = None) -> dict[str, int]:
        """Categorical label counts for a scorer (empty dict when nothing categorical under ``key``).

        Pass ``tag`` to restrict to tasks carrying that tag.
        """
        return dict(self._agg(tag).value_counts.get(key, {}))

    def _agg(self, tag: str | None) -> Aggregates:
        """Aggregates for the whole run (``tag`` is ``None``) or only the tasks carrying ``tag`` (memoized)."""
        if tag is None:
            return self._aggregates
        if tag not in self._sliced:
            self._sliced[tag] = _compute_aggregates(tuple(tr for tr in self._tasks if tag in tr.task.tags))
        return self._sliced[tag]

    def diff(self, baseline: "RunResult", *, strict: bool = True) -> "RunDiff":
        """Compare this run against ``baseline`` — "did my change help or hurt?".

        Reports per-scorer pass-rate / mean deltas and the tasks that flipped
        pass<->fail, over the tasks and checks the two runs **share**. By default
        (``strict=True``) raises :class:`~autogen.beta.eval.RunsNotComparableError` if the
        runs didn't grade the same tasks + checks; pass ``strict=False`` to diff the
        overlap and have the mismatches reported on the returned
        :class:`~autogen.beta.eval.RunDiff`.
        """
        # Local import: result.py <-> diff.py circular-import shim (AGENTS.md exempt).
        from .diff import compute_diff

        return compute_diff(self, baseline, strict=strict)

    def summary(self) -> str:
        """Human-readable multi-line table of run metadata and aggregates.

        Format is plain ASCII (no charting libraries) so the output can be
        copied straight into a CI log or a CHANGELOG.
        """
        return _render_summary(self)

    def save(self, path: str | os.PathLike[str] | None = None) -> Path:
        """Write the run as schema-0.1 JSON.

        If ``path`` is ``None``, saves under the run's configured
        ``store_dir`` (set by the runner via ``run_agent(..., store_dir=...)``)
        as ``<run_id>.json``. If ``path`` ends in ``.json`` it's used
        verbatim; otherwise ``path`` is treated as a directory and
        ``<run_id>.json`` is appended.

        Raises:
            ValueError: if ``path`` is ``None`` and no ``store_dir`` was
                configured on this run.

        Returns:
            The :class:`Path` that was written.
        """
        # Local import: store.py also depends on result.py for type names,
        # so importing at module top would create a circular import.
        # AGENTS.md exempts circular-import shims from the no-function-
        # level-imports rule.
        from .store import dump

        target = self._resolve_save_path(path)
        return dump(self, target)

    def _resolve_save_path(self, path: str | os.PathLike[str] | None) -> Path:
        if path is None:
            if self._store_dir is None:
                raise ValueError(
                    "RunResult.save(): no path given and no store_dir was set on the run; "
                    "pass an explicit path or pass store_dir= to run_agent()."
                )
            return self._store_dir / f"{self._run_id}.json"
        p = Path(path)
        if p.suffix == ".json":
            return p
        return p / f"{self._run_id}.json"


def _compute_aggregates(tasks: tuple[TaskResult, ...]) -> Aggregates:
    """Build :class:`Aggregates` once at run-construction time."""
    bool_outcomes: dict[str, list[bool]] = {}
    numeric_scores: dict[str, list[float]] = {}
    value_counts: dict[str, dict[str, int]] = {}

    for tr in tasks:
        for fb in tr.feedback:
            if isinstance(fb.score, bool):
                bool_outcomes.setdefault(fb.key, []).append(fb.score)
            elif isinstance(fb.score, (int, float)):
                numeric_scores.setdefault(fb.key, []).append(float(fb.score))
            if fb.value is not None:
                bucket = value_counts.setdefault(fb.key, {})
                bucket[fb.value] = bucket.get(fb.value, 0) + 1

    pass_rate = {key: sum(1 for v in scores if v) / len(scores) for key, scores in bool_outcomes.items()}
    score_stats = {key: _stats(scores) for key, scores in numeric_scores.items()}

    tokens_total = TokenUsage(
        input=sum(tr.trace.tokens.input for tr in tasks),
        output=sum(tr.trace.tokens.output for tr in tasks),
        cache_creation=sum(tr.trace.tokens.cache_creation for tr in tasks),
        cache_read=sum(tr.trace.tokens.cache_read for tr in tasks),
    )

    errors = sum(1 for tr in tasks if tr.trace.exception is not None)
    budget_violations = sum(1 for tr in tasks if tr.budget_violation)

    return Aggregates(
        pass_rate=pass_rate,
        score_stats=score_stats,
        value_counts=value_counts,
        tokens=tokens_total,
        errors=errors,
        budget_violations=budget_violations,
    )


def _stats(scores: list[float]) -> ScoreStats:
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    mean = sum(sorted_scores) / n if n else 0.0
    return ScoreStats(
        mean=mean,
        p50=_percentile(sorted_scores, 0.50),
        p95=_percentile(sorted_scores, 0.95),
        n=n,
    )


def _percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolated percentile (numpy convention).

    Linear interpolation is smoother than nearest-rank on small samples,
    which matters because eval suites are typically 5–50 tasks.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = p * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def _pass_counts(tasks: tuple[TaskResult, ...]) -> dict[str, tuple[int, int]]:
    """(passed, total) per boolean scorer key — for showing pass-rate denominators."""
    passed: dict[str, int] = {}
    total: dict[str, int] = {}
    for tr in tasks:
        for fb in tr.feedback:
            if isinstance(fb.score, bool):
                total[fb.key] = total.get(fb.key, 0) + 1
                passed[fb.key] = passed.get(fb.key, 0) + (1 if fb.score else 0)
    return {key: (passed[key], total[key]) for key in total}


def _render_summary(result: RunResult) -> str:
    """Format a printable multi-line summary table."""
    lines: list[str] = []
    lines.append(f"Run {result.run_id}")
    lines.append(f"  Schema:      {result.schema_version}")
    lines.append(f"  Created:     {result.created_at}")
    lines.append(f"  Duration:    {result.duration_ms}ms")
    lines.append(f"  Suite:       {result.suite.name} ({len(result.suite)} tasks, source: {result.suite.source})")
    lines.append(f"  Runs:        {len(result.tasks)}")
    lines.append(f"  Concurrency: {result.concurrency}")

    aggs = result.aggregates
    lines.append(f"  Errors:      {aggs.errors}")
    lines.append(f"  Budget violations: {aggs.budget_violations}")
    lines.append(f"  Tokens:      input={aggs.tokens.input} output={aggs.tokens.output} total={aggs.tokens.total}")

    if aggs.pass_rate:
        lines.append("")
        lines.append("Pass rates:")
        counts = _pass_counts(result.tasks)
        width = max(len(k) for k in aggs.pass_rate)
        for key in sorted(aggs.pass_rate):
            rate = aggs.pass_rate[key]
            passed, total = counts.get(key, (0, 0))
            lines.append(f"  {key:<{width}}  {rate * 100:5.1f}% ({passed}/{total})")

    if aggs.score_stats:
        lines.append("")
        lines.append("Score stats:")
        width = max(len(k) for k in aggs.score_stats)
        for key in sorted(aggs.score_stats):
            s = aggs.score_stats[key]
            lines.append(f"  {key:<{width}}  mean={s.mean:.2f} p50={s.p50:.2f} p95={s.p95:.2f} n={s.n}")

    if aggs.value_counts:
        lines.append("")
        lines.append("Value counts:")
        width = max(len(k) for k in aggs.value_counts)
        for key in sorted(aggs.value_counts):
            counts = aggs.value_counts[key]
            joined = " ".join(f"{label}={count}" for label, count in sorted(counts.items()))
            lines.append(f"  {key:<{width}}  {joined}")

    return "\n".join(lines)
