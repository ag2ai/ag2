# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Verification results: per boundary, and aggregated per strategy."""

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .actions import Action
from .boundary import ContextSize
from .burden import BoundaryDelta
from .stats import Interval, bootstrap_mean, exact_permutation_test, exact_sign_test, median

__all__ = (
    "BoundaryResult",
    "HorizonSummary",
    "Rollout",
    "StrategyReport",
    "VerificationReport",
)

Arm = Literal["pre", "post"]
Ending = Literal["budget", "answer", "error"]

MIN_BOUNDARIES_FOR_SIGNIFICANCE = 10
"""Below this many boundaries :meth:`VerificationReport.summary` marks no interval
as significant: a percentile bootstrap over so few units is not reliable, and
over one unit it collapses to the point estimate."""


@dataclass(frozen=True, slots=True)
class Rollout:
    """One free-running continuation from a boundary.

    Attributes:
        arm: ``"pre"`` (resumed from the raw history) or ``"post"`` (from the
            compacted history).
        sample: Index of this rollout among its arm's samples.
        actions: The calls the agent executed after resuming, in order.
        ending: ``"budget"`` when the action budget was reached, ``"answer"``
            when the agent finished first, ``"error"`` when the run raised.
        error: The exception, for an ``"error"`` rollout. Such rollouts are
            not scored.
    """

    arm: Arm
    sample: int
    actions: tuple[Action, ...]
    ending: Ending
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BoundaryResult:
    """One strategy at one boundary."""

    trajectory: int
    cut: int
    strategy: str
    deltas: tuple[BoundaryDelta, ...]
    """POST minus PRE for every horizon ``1..k``; empty if an arm had no scorable rollout."""
    pre_size: ContextSize
    post_size: ContextSize
    pre: tuple[Rollout, ...]
    post: tuple[Rollout, ...]
    compaction_tokens: int
    """Tokens the strategy itself spent compacting this boundary."""
    unchanged: bool
    """The strategy returned the history unchanged: POST saw exactly the PRE context."""

    @property
    def scored(self) -> bool:
        return bool(self.deltas)

    def at(self, horizon: int) -> BoundaryDelta:
        for d in self.deltas:
            if d.horizon == horizon:
                return d
        raise KeyError(horizon)


@dataclass(frozen=True, slots=True)
class HorizonSummary:
    """Mean POST-minus-PRE burden across boundaries at one horizon."""

    horizon: int
    boundaries: int
    blocked: Interval
    refetch: Interval
    wasted: Interval
    harm: Interval
    stopped: Interval


@dataclass(frozen=True, slots=True)
class StrategyReport:
    """Everything measured for one strategy."""

    strategy: str
    horizons: tuple[HorizonSummary, ...]
    boundaries: tuple[BoundaryResult, ...]
    sign_test_p: float
    """Exact sign test on per-boundary ``delta_wasted`` at the full horizon."""
    permutation_p: float
    """Exact sign-flip permutation test on the same differences."""
    median_token_ratio: float
    """Median of POST tokens / PRE tokens: how much of the context the strategy kept."""
    unchanged_boundaries: int
    """Boundaries where the strategy returned the history unchanged, so POST was PRE."""
    failed_rollouts: int

    def at(self, horizon: int) -> HorizonSummary:
        for h in self.horizons:
            if h.horizon == horizon:
                return h
        raise KeyError(horizon)

    @classmethod
    def build(cls, strategy: str, results: Sequence[BoundaryResult], horizon: int) -> "StrategyReport":
        scored = [r for r in results if r.scored]
        horizons = tuple(_summarize(scored, k) for k in range(1, horizon + 1)) if scored else ()
        full = [r.at(horizon).wasted for r in scored]
        ratios = [r.post_size.tokens / r.pre_size.tokens for r in results if r.pre_size.tokens]
        failed = sum(1 for r in results for x in r.post if x.ending == "error")
        unchanged = sum(1 for r in results if r.unchanged)
        return cls(
            strategy=strategy,
            horizons=horizons,
            boundaries=tuple(results),
            sign_test_p=exact_sign_test(full),
            permutation_p=exact_permutation_test(full),
            median_token_ratio=median(ratios) if ratios else 1.0,
            unchanged_boundaries=unchanged,
            failed_rollouts=failed,
        )


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """The verdict on every strategy, over the same boundaries and PRE rollouts."""

    horizon: int
    samples: int
    strategies: Mapping[str, StrategyReport]
    failed_pre_rollouts: int

    def summary(self) -> str:
        """A short plain-text table: per strategy, the mean delta at the full horizon."""
        k = self.horizon
        lines = [
            f"compaction verifier: POST minus PRE over the first {k} actions "
            f"({self.samples} samples per arm; * = 95% CI excludes 0)",
            f"{'strategy':<20}{'n':>4}  {'wasted':>22}  {'blocked':>22}  {'refetch':>22}  "
            f"{'stopped':>22}  {'kept':>5}  {'noop':>4}  {'p':>6}",
        ]
        few = False
        for name, report in self.strategies.items():
            if not report.horizons:
                lines.append(f"{name:<20}{0:>4}  no scorable boundaries")
                continue
            h = report.at(k)
            few = few or h.boundaries < MIN_BOUNDARIES_FOR_SIGNIFICANCE
            lines.append(
                f"{name:<20}{h.boundaries:>4}  {_fmt(h.wasted):>22}  {_fmt(h.blocked):>22}  "
                f"{_fmt(h.refetch):>22}  {_fmt(h.stopped):>22}  {report.median_token_ratio:>5.0%}  "
                f"{report.unchanged_boundaries:>4}  {report.permutation_p:>6.3f}"
            )
        lines.append(
            "stopped = share of rollouts that ended before the horizon; kept = median share of "
            "context tokens kept; noop = boundaries left unchanged; p = exact sign-flip "
            "permutation test on per-boundary wasted delta"
        )
        if few:
            lines.append(
                f"fewer than {MIN_BOUNDARIES_FOR_SIGNIFICANCE} boundaries: intervals are shown "
                "without significance marks"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable view, rollouts and actions included."""
        return {
            "horizon": self.horizon,
            "samples": self.samples,
            "failed_pre_rollouts": self.failed_pre_rollouts,
            "strategies": {name: _strategy_dict(r) for name, r in self.strategies.items()},
        }


def _summarize(results: Sequence[BoundaryResult], horizon: int) -> HorizonSummary:
    deltas = [r.at(horizon) for r in results]
    return HorizonSummary(
        horizon=horizon,
        boundaries=len(deltas),
        blocked=bootstrap_mean([d.blocked for d in deltas]),
        refetch=bootstrap_mean([d.refetch for d in deltas]),
        wasted=bootstrap_mean([d.wasted for d in deltas]),
        harm=bootstrap_mean([d.harm for d in deltas]),
        stopped=bootstrap_mean([d.stopped for d in deltas]),
    )


def _fmt(i: Interval) -> str:
    star = "*" if i.excludes_zero and i.n >= MIN_BOUNDARIES_FOR_SIGNIFICANCE else " "
    return f"{i.mean:+.3f} [{i.low:+.3f},{i.high:+.3f}]{star}"


def _strategy_dict(report: StrategyReport) -> dict[str, Any]:
    return {
        "strategy": report.strategy,
        "sign_test_p": report.sign_test_p,
        "permutation_p": report.permutation_p,
        "median_token_ratio": report.median_token_ratio,
        "unchanged_boundaries": report.unchanged_boundaries,
        "failed_rollouts": report.failed_rollouts,
        "horizons": [dataclasses.asdict(h) for h in report.horizons],
        "boundaries": [_boundary_dict(b) for b in report.boundaries],
    }


def _boundary_dict(result: BoundaryResult) -> dict[str, Any]:
    return {
        "trajectory": result.trajectory,
        "cut": result.cut,
        "pre_size": dataclasses.asdict(result.pre_size),
        "post_size": dataclasses.asdict(result.post_size),
        "compaction_tokens": result.compaction_tokens,
        "unchanged": result.unchanged,
        "deltas": [
            {
                "horizon": d.horizon,
                "blocked": d.blocked,
                "refetch": d.refetch,
                "wasted": d.wasted,
                "harm": d.harm,
                "stopped": d.stopped,
                "pre": dataclasses.asdict(d.pre),
                "post": dataclasses.asdict(d.post),
            }
            for d in result.deltas
        ],
        "pre": [dataclasses.asdict(r) for r in result.pre],
        "post": [dataclasses.asdict(r) for r in result.post],
    }
