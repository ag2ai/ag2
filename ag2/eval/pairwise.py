# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Pairwise comparison — compare two agent variants (A vs B) on the same tasks.

The unit is a :class:`PairwiseComparator`: given a task and the two variants'
traces, it returns a :class:`PairwiseOutcome` (which won, or tie). *Who* decides
is swappable behind the protocol — an LLM judge (``pairwise_judge``), a human
(``human_labels`` / inline HITL), or a user's own implementation. Each
comparator encapsulates its own position strategy, so the runner just calls
``compare()`` and tallies.

:func:`evaluate_pairwise` pairs the two sources' traces by ``task_id``, runs the
comparators, and returns a :class:`PairwiseRunResult` — per-key win/loss/tie,
**win-rate(B)** with a Wilson confidence interval, position-flip counts, and
judge-vs-human **agreement** (Cohen's kappa).
"""

import asyncio
import json
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import uuid4

from ag2.context import ConversationContext
from ag2.stream import Stream

from .dataset import Suite, Task
from .events import PairwiseCompared, PairwiseCompleted, PairwiseStarted
from .sources import TraceRef, TraceSource
from .trace import Trace

__all__ = (
    "Agreement",
    "PairwiseCase",
    "PairwiseComparator",
    "PairwiseOutcome",
    "PairwiseRunResult",
    "WinRate",
    "evaluate_pairwise",
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "0.1"
_Z_95 = 1.96


@dataclass(frozen=True, slots=True)
class PairwiseOutcome:
    """The result of comparing variant A against variant B on one task.

    ``winner`` is ``"a"`` / ``"b"`` / ``"tie"``. ``detail`` carries audit info
    such as the per-order verdicts behind a swapped LLM judgment.
    """

    winner: Literal["a", "b", "tie"]
    reasoning: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class PairwiseComparator(Protocol):
    """Decides A vs B for one task. LLM / human / custom implementations interchange."""

    key: str
    """Result key this comparator reports under (its column in the result)."""

    async def compare(
        self,
        *,
        task: Task,
        trace_a: Trace,
        trace_b: Trace,
        reference_outputs: dict[str, Any] | None,
    ) -> PairwiseOutcome: ...


@dataclass(frozen=True, slots=True)
class PairwiseCase:
    """One comparator's outcome for one paired task."""

    task_id: str
    key: str
    winner: Literal["a", "b", "tie"]
    reasoning: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WinRate:
    """Variant B's win-rate on one key (ties count 0.5), with a Wilson 95% CI."""

    variant: str
    rate: float
    wins: int  # B
    losses: int  # A
    ties: int
    n: int
    ci: tuple[float, float]


@dataclass(frozen=True, slots=True)
class Agreement:
    """Agreement between two comparator keys (e.g. judge vs human)."""

    rate: float
    cohen_kappa: float
    n: int
    disagreements: tuple[tuple[str, str, str], ...]  # (task_id, winner_x, winner_y)


class PairwiseRunResult:
    """Per-key win/loss/tie, win-rate(B) + Wilson CI, flips, and agreement."""

    __slots__ = (
        "_run_id",
        "_label",
        "_cases",
        "_variant_a",
        "_variant_b",
        "_keys",
        "_created_at",
        "_duration_ms",
        "_n_pairs",
        "_store_dir",
    )

    def __init__(
        self,
        *,
        run_id: str,
        cases: tuple[PairwiseCase, ...],
        variant_a: str,
        variant_b: str,
        keys: tuple[str, ...],
        created_at: str,
        duration_ms: int,
        n_pairs: int,
        label: str | None = None,
        store_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self._run_id = run_id
        self._label = label
        self._cases = cases
        self._variant_a = variant_a
        self._variant_b = variant_b
        self._keys = keys
        self._created_at = created_at
        self._duration_ms = duration_ms
        self._n_pairs = n_pairs
        self._store_dir = Path(store_dir) if store_dir is not None else None

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def label(self) -> str | None:
        return self._label

    @property
    def cases(self) -> tuple[PairwiseCase, ...]:
        return self._cases

    def tally(self, key: str) -> tuple[int, int, int]:
        """``(a_wins, b_wins, ties)`` for one key."""
        a = b = t = 0
        for case in self._cases:
            if case.key != key:
                continue
            if case.winner == "a":
                a += 1
            elif case.winner == "b":
                b += 1
            else:
                t += 1
        return a, b, t

    def win_rate(self, key: str) -> WinRate:
        """Variant B's win-rate on ``key`` (ties = 0.5) with a Wilson 95% CI."""
        a, b, t = self.tally(key)
        n = a + b + t
        rate = (b + 0.5 * t) / n if n else 0.0
        return WinRate(variant=self._variant_b, rate=rate, wins=b, losses=a, ties=t, n=n, ci=_wilson_ci(rate, n))

    def flips(self, key: str) -> int:
        """Cases where the swapped orders disagreed (position sensitivity)."""
        count = 0
        for case in self._cases:
            if case.key != key:
                continue
            detail = case.detail
            if (
                "order1" in detail
                and "order2" in detail
                and _pos_to_ab(detail["order1"], "a", "b") != _pos_to_ab(detail["order2"], "b", "a")
            ):
                count += 1
        return count

    def agreement(self, key_x: str, key_y: str) -> Agreement:
        """Agreement between two keys over the tasks both scored (Cohen's kappa)."""
        x = {c.task_id: c.winner for c in self._cases if c.key == key_x}
        y = {c.task_id: c.winner for c in self._cases if c.key == key_y}
        shared = sorted(set(x) & set(y))
        pairs = [(x[t], y[t]) for t in shared]
        n = len(pairs)
        rate = sum(1 for a, b in pairs if a == b) / n if n else 0.0
        disagreements = tuple((t, x[t], y[t]) for t in shared if x[t] != y[t])
        return Agreement(rate=rate, cohen_kappa=_cohen_kappa(pairs), n=n, disagreements=disagreements)

    def summary(self) -> str:
        lines = [
            f"Pairwise {self._run_id}  —  {self._variant_b!r} (B) vs {self._variant_a!r} (A) · {self._n_pairs} cases",
            f"  {'key':<18} {'B':>3} {'A':>3} {'tie':>4}   win-rate({self._variant_b})   95% CI",
        ]
        for key in self._keys:
            wr = self.win_rate(key)
            lines.append(
                f"  {key:<18} {wr.wins:>3} {wr.losses:>3} {wr.ties:>4}      {wr.rate * 100:5.1f}%   "
                f"[{wr.ci[0] * 100:.0f}%, {wr.ci[1] * 100:.0f}%]"
            )
            flips = self.flips(key)
            if flips:
                lines.append(f"  {'':<18} position-flips → tie: {flips}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "run_id": self._run_id,
            "label": self._label,
            "created_at": self._created_at,
            "duration_ms": self._duration_ms,
            "variant_a": self._variant_a,
            "variant_b": self._variant_b,
            "n_pairs": self._n_pairs,
            "keys": list(self._keys),
            "win_rates": {k: _win_rate_to_dict(self.win_rate(k)) for k in self._keys},
            "flips": {k: self.flips(k) for k in self._keys},
            "cases": [
                {
                    "task_id": c.task_id,
                    "key": c.key,
                    "winner": c.winner,
                    "reasoning": c.reasoning,
                    "detail": dict(c.detail),
                }
                for c in self._cases
            ],
        }

    def save(self, path: str | os.PathLike[str] | None = None) -> Path:
        target = self._resolve_save_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return target

    def _resolve_save_path(self, path: str | os.PathLike[str] | None) -> Path:
        if path is None:
            if self._store_dir is None:
                raise ValueError("PairwiseRunResult.save(): no path given and no store_dir was set on the run.")
            return self._store_dir / f"{self._run_id}.json"
        p = Path(path)
        return p if p.suffix == ".json" else p / f"{self._run_id}.json"


async def evaluate_pairwise(
    source_a: TraceSource,
    source_b: TraceSource,
    *,
    comparators: Iterable[PairwiseComparator],
    store_dir: str | os.PathLike[str] | None,
    variant_a: str = "A",
    variant_b: str = "B",
    suite: Suite | None = None,
    concurrency: int = 4,
    run_id: str | None = None,
    label: str | None = None,
    stream: Stream | None = None,
) -> PairwiseRunResult:
    """Compare variant A vs variant B over the tasks both produced traces for.

    Traces are paired by ``TraceRef.task_id`` (stamp ``ag2.eval.task_id`` at
    produce time). Each comparator runs over every pair; results roll up per key.

    ``label`` is a shared identifier recorded on the run; pass ``stream`` to publish
    pairwise lifecycle events (``PairwiseStarted`` / ``PairwiseCompared`` per case /
    ``PairwiseCompleted``) for live observation.
    """
    comparator_list = tuple(comparators)
    keys = tuple(c.key for c in comparator_list)
    tasks_by_id = {task.task_id: task for task in suite} if suite is not None else {}

    refs_a = [ref async for ref in source_a.list()]
    b_by_task: dict[str, TraceRef] = {}
    async for ref in source_b.list():
        if ref.task_id is not None:
            b_by_task[ref.task_id] = ref
    pairs = [(ra, b_by_task[ra.task_id]) for ra in refs_a if ra.task_id is not None and ra.task_id in b_by_task]
    if not pairs:
        logger.warning(
            "evaluate_pairwise: no task_id-matched pairs between the sources (need ag2.eval.task_id on both)."
        )

    semaphore = asyncio.Semaphore(max(1, concurrency))
    actual_run_id = run_id if run_id is not None else uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()

    if stream is not None:
        eval_ctx = ConversationContext(stream=stream)
        await eval_ctx.send(
            PairwiseStarted(
                run_id=actual_run_id, label=label, variant_a=variant_a, variant_b=variant_b, total=len(pairs)
            ),
        )
        on_case = partial(_publish_pairwise_compared, eval_ctx, actual_run_id, label)

    else:
        eval_ctx, on_case = None, None

    case_lists = await asyncio.gather(
        *(
            _evaluate_pair(semaphore, source_a, source_b, ra, rb, comparator_list, tasks_by_id, on_case)
            for ra, rb in pairs
        )
    )
    cases = tuple(case for case_list in case_lists for case in case_list)
    duration_ms = int((time.perf_counter() - started) * 1000)

    result = PairwiseRunResult(
        run_id=actual_run_id,
        cases=cases,
        variant_a=variant_a,
        variant_b=variant_b,
        keys=keys,
        created_at=created_at,
        duration_ms=duration_ms,
        n_pairs=len(pairs),
        label=label,
        store_dir=store_dir,
    )
    if store_dir:
        result.save()

    if eval_ctx is not None:
        await eval_ctx.send(PairwiseCompleted(run_id=actual_run_id, label=label, result=result))

    return result


async def _evaluate_pair(
    semaphore: asyncio.Semaphore,
    source_a: TraceSource,
    source_b: TraceSource,
    ref_a: TraceRef,
    ref_b: TraceRef,
    comparators: tuple[PairwiseComparator, ...],
    tasks_by_id: dict[str, Task],
    on_case: Callable[[PairwiseCase], Awaitable[None]] | None = None,
) -> list[PairwiseCase]:
    async with semaphore:
        trace_a = await source_a.load(ref_a)
        trace_b = await source_b.load(ref_b)
        task_id = ref_a.task_id or ""
        task = tasks_by_id.get(task_id) or Task(task_id=task_id, inputs={}, reference_outputs=None)

        out: list[PairwiseCase] = []
        for comparator in comparators:
            try:
                outcome = await comparator.compare(
                    task=task, trace_a=trace_a, trace_b=trace_b, reference_outputs=task.reference_outputs
                )
            except Exception as exc:
                logger.warning("pairwise comparator %r raised: %s", comparator.key, exc)
                outcome = PairwiseOutcome(
                    winner="tie", reasoning=f"comparator raised: {type(exc).__name__}: {exc}", detail={"error": True}
                )
            case = PairwiseCase(
                task_id=task.task_id,
                key=comparator.key,
                winner=outcome.winner,
                reasoning=outcome.reasoning,
                detail=dict(outcome.detail),
            )
            out.append(case)
            if on_case is not None:
                await on_case(case)
        return out


async def _publish_pairwise_compared(
    ctx: ConversationContext,
    run_id: str,
    label: str | None,
    case: PairwiseCase,
) -> None:
    """Publish a :class:`PairwiseCompared` event when one comparator finishes one pair."""
    await ctx.send(PairwiseCompared(run_id=run_id, label=label, task_id=case.task_id, key=case.key, winner=case.winner))


def _pos_to_ab(preferred: str, first: str, second: str) -> str:
    if preferred == "first":
        return first
    if preferred == "second":
        return second
    return "tie"


def _wilson_ci(phat: float, n: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion (small-N friendly)."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    """Chance-corrected agreement between two raters over labels a/b/tie."""
    n = len(pairs)
    if n == 0:
        return 0.0
    labels = ("a", "b", "tie")
    observed = sum(1 for x, y in pairs if x == y) / n
    px = {lbl: sum(1 for x, _ in pairs if x == lbl) / n for lbl in labels}
    py = {lbl: sum(1 for _, y in pairs if y == lbl) / n for lbl in labels}
    expected = sum(px[lbl] * py[lbl] for lbl in labels)
    return 1.0 if expected >= 1.0 else (observed - expected) / (1 - expected)


def _win_rate_to_dict(wr: WinRate) -> dict[str, Any]:
    return {
        "variant": wr.variant,
        "rate": wr.rate,
        "wins": wr.wins,
        "losses": wr.losses,
        "ties": wr.ties,
        "n": wr.n,
        "ci": list(wr.ci),
    }
