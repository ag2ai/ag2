# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""run_variants — compare named variants of a target and rank them.

A :class:`Variants` is a *single-axis* set of named builds — all configs, or
all prompts, … — constructed via the typed ``Variants.from_*`` classmethods
(mirroring ``Suite.from_*``). :func:`run_variants` runs each variant over the
suite via :func:`~autogen.beta.eval.run` and returns a :class:`VariantRunResult`
whose ``leaderboard(key)`` ranks the variants by a scorer.

Each ``from_*`` fixes the axis and types its values, so one call can only vary
one thing — comparisons stay controlled (vary one axis, hold the rest fixed).
For an entirely different build per variant, use :meth:`Variants.from_targets`.
"""

import os
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from operator import itemgetter
from typing import Any
from uuid import uuid4

from autogen.beta.config import ModelConfig
from autogen.beta.context import ConversationContext
from autogen.beta.middleware.base import MiddlewareFactory
from autogen.beta.stream import Stream
from autogen.beta.tools.tool import Tool

from ..dataset import EvalTarget, Suite
from ..events import VariantCompleted, VariantStarted
from ..results import RunResult
from ..scorer import Scorer
from .runner import run

__all__ = (
    "LeaderboardRow",
    "VariantRunResult",
    "Variants",
    "run_variants",
)

_Build = Callable[..., EvalTarget] | EvalTarget


@dataclass(frozen=True, slots=True)
class Variants:
    """A single-axis set of named target builds. Construct via the ``from_*`` classmethods."""

    axis: str
    builds: Mapping[str, _Build]

    @classmethod
    def from_configs(cls, factory: Callable[..., EvalTarget], variants: Mapping[str, ModelConfig]) -> "Variants":
        """One variant per model config — varies ``config`` on a shared factory."""
        return cls("config", {name: partial(factory, config=cfg) for name, cfg in variants.items()})

    @classmethod
    def from_prompts(cls, factory: Callable[..., EvalTarget], variants: Mapping[str, str]) -> "Variants":
        """One variant per system prompt — varies ``prompt`` on a shared factory."""
        return cls("prompt", {name: partial(factory, prompt=prompt) for name, prompt in variants.items()})

    @classmethod
    def from_tools(
        cls, factory: Callable[..., EvalTarget], variants: Mapping[str, Sequence[Callable[..., Any] | Tool]]
    ) -> "Variants":
        """One variant per tool set — varies ``tools`` on a shared factory."""
        return cls("tools", {name: partial(factory, tools=list(tools)) for name, tools in variants.items()})

    @classmethod
    def from_middleware(
        cls, factory: Callable[..., EvalTarget], variants: Mapping[str, Sequence[MiddlewareFactory]]
    ) -> "Variants":
        """One variant per middleware stack — varies ``middleware`` on a shared factory."""
        return cls("middleware", {name: partial(factory, middleware=list(mw)) for name, mw in variants.items()})

    @classmethod
    def from_targets(cls, variants: Mapping[str, _Build]) -> "Variants":
        """One variant per whole build — each value is a factory or an instance (the escape hatch)."""
        return cls("target", dict(variants))


@dataclass(frozen=True, slots=True)
class LeaderboardRow:
    """One ranked row: variant, its score, sample size, and rank (tied scores share a rank)."""

    variant: str
    score: float
    n: int
    rank: int


@dataclass(frozen=True, slots=True)
class VariantRunResult:
    """Result of :func:`run_variants`: a per-variant :class:`RunResult` plus ranking helpers."""

    run_id: str
    axis: str
    results: dict[str, RunResult]
    created_at: str
    duration_ms: int

    def leaderboard(self, key: str) -> list[LeaderboardRow]:
        """Variants ranked best-first by scorer ``key`` (pass-rate for boolean, mean for numeric).

        Tied scores share a rank (competition ranking — ``1, 1, 1, 4``).
        """
        scored = sorted(
            ((name, *self._metric(res, key)) for name, res in self.results.items()),
            key=itemgetter(1),
            reverse=True,
        )
        return [
            LeaderboardRow(variant=name, score=score, n=n, rank=1 + sum(1 for _, other, _ in scored if other > score))
            for name, score, n in scored
        ]

    def best(self, key: str) -> str | None:
        """Top-ranked variant for ``key`` — or ``None`` when the top score is shared (no unique winner)."""
        leaders = [row for row in self.leaderboard(key) if row.rank == 1]
        return leaders[0].variant if len(leaders) == 1 else None

    def summary(self, key: str) -> str:
        """Printable ranked leaderboard for ``key``."""
        rows = self.leaderboard(key)
        is_rate = any(key in res.aggregates.pass_rate for res in self.results.values())
        width = max((len(row.variant) for row in rows), default=0)
        lines = [f"Variants (axis: {self.axis}) — ranked by {key}:"]
        for row in rows:
            value = f"{row.score * 100:5.1f}%" if is_rate else f"{row.score:.3f}"
            lines.append(f"  {row.rank}. {row.variant:<{width}}  {value}  (n={row.n})")
        return "\n".join(lines)

    @staticmethod
    def _metric(res: RunResult, key: str) -> tuple[float, int]:
        if key in res.aggregates.pass_rate:
            return res.pass_rate(key), len(res.tasks)
        stats = res.score_stats(key)
        return stats.mean, stats.n


async def run_variants(
    suite: Suite | str | os.PathLike[str] | list[dict[str, Any]],
    *,
    variants: Variants,
    scorers: Iterable[Scorer],
    store_dir: str | os.PathLike[str],
    repeats: int = 1,
    concurrency: int = 4,
    run_id: str | None = None,
    label: str | None = None,
    stream: Stream | None = None,
) -> VariantRunResult:
    """Run each variant over ``suite`` and return a ranked :class:`VariantRunResult`.

    Variants run sequentially; within each, tasks run up to ``concurrency`` in
    parallel (and ``repeats`` times each). Every variant's run is persisted as
    its own schema-0.1 JSON under ``store_dir`` (``<run_id>-<variant>.json``).

    Pass ``stream`` to observe the sweep: ``VariantStarted`` / ``VariantCompleted``
    wrap each variant, and that variant's own ``EvalStarted`` / ``TaskEvaluated``
    (tagged with the variant name) / ``EvalCompleted`` flow through the same
    stream — so a single observer sees both the sweep and the per-task detail.
    """
    scorer_list = tuple(scorers)
    base_run_id = run_id if run_id is not None else uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()

    eval_stream = stream
    eval_ctx = ConversationContext(stream=eval_stream) if eval_stream is not None else None

    results: dict[str, RunResult] = {}
    total = len(variants.builds)
    for index, (name, build) in enumerate(variants.builds.items(), start=1):
        if eval_stream is not None:
            await eval_stream.send(
                VariantStarted(run_id=base_run_id, label=label, variant=name, index=index, total=total),
                eval_ctx,
            )
        results[name] = await run(
            suite,
            target=build,
            scorers=scorer_list,
            store_dir=store_dir,
            repeats=repeats,
            concurrency=concurrency,
            run_id=f"{base_run_id}-{_slug(name)}",
            label=label,
            stream=eval_stream,
            variant=name,
        )
        if eval_stream is not None:
            await eval_stream.send(
                VariantCompleted(run_id=base_run_id, label=label, variant=name, result=results[name]),
                eval_ctx,
            )

    duration_ms = int((time.perf_counter() - started) * 1000)
    return VariantRunResult(
        run_id=base_run_id,
        axis=variants.axis,
        results=results,
        created_at=created_at,
        duration_ms=duration_ms,
    )


def _slug(name: str) -> str:
    """Filesystem-safe form of a variant name for the run-JSON filename."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)
