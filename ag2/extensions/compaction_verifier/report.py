# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Verification results: per boundary, and aggregated per strategy."""

import dataclasses
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
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
    post_size: ContextSize | None
    """``None`` when the strategy failed to compact this boundary."""
    pre: tuple[Rollout, ...]
    post: tuple[Rollout, ...]
    compaction_tokens: int
    """Tokens the strategy itself spent compacting this boundary."""
    unchanged: bool
    """The strategy returned the history unchanged: POST saw exactly the PRE context."""
    compaction_error: str | None = None
    """Why the strategy produced no POST context here; no POST rollout ran, and the boundary is not scored."""

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
    unscored_boundaries: int
    """Boundaries left out of every average because an arm had no rollout that ran."""
    post_rollouts: int
    failed_rollouts: int
    """POST rollouts that raised. They are not scored."""
    pre_rollouts: int
    """PRE rollouts at this strategy's boundaries (shared with every strategy)."""
    failed_pre_rollouts: int
    post_errors: tuple[tuple[str, int], ...]
    """Distinct POST errors with their counts, most common first."""
    failed_compactions: int = 0
    """Boundaries where the strategy raised instead of returning a context; they are not scored."""
    compaction_errors: tuple[tuple[str, int], ...] = ()
    """Distinct compaction errors with their counts, most common first."""
    grown_boundaries: int = 0
    """Boundaries where the compacted context was longer than the history it replaced."""

    def at(self, horizon: int) -> HorizonSummary:
        for h in self.horizons:
            if h.horizon == horizon:
                return h
        raise KeyError(horizon)

    @property
    def failure_asymmetry(self) -> Arm | None:
        """The arm that failed more often at this strategy's boundaries, or ``None`` if neither did.

        Failed rollouts are not scored, so an asymmetry biases the delta: more
        POST failures (say, a provider rejecting the compacted context) leave only
        the POST samples that ran and can understate the harm; more PRE failures
        (say, the raw history overflowing the context window) can understate
        what compaction saves.
        """
        post = self.failed_rollouts * self.pre_rollouts
        pre = self.failed_pre_rollouts * self.post_rollouts
        if post > pre:
            return "post"
        if pre > post:
            return "pre"
        return None

    @classmethod
    def build(cls, strategy: str, results: Sequence[BoundaryResult], horizon: int) -> "StrategyReport":
        scored = [r for r in results if r.scored]
        horizons = tuple(_summarize(scored, k) for k in range(1, horizon + 1)) if scored else ()
        full = [r.at(horizon).wasted for r in scored]
        ratios = [r.post_size.tokens / r.pre_size.tokens for r in results if r.post_size and r.pre_size.tokens]
        post_runs = [x for r in results for x in r.post]
        pre_runs = [x for r in results for x in r.pre]
        return cls(
            strategy=strategy,
            horizons=horizons,
            boundaries=tuple(results),
            sign_test_p=exact_sign_test(full),
            permutation_p=exact_permutation_test(full),
            median_token_ratio=median(ratios) if ratios else 1.0,
            unchanged_boundaries=sum(1 for r in results if r.unchanged),
            unscored_boundaries=len(results) - len(scored),
            post_rollouts=len(post_runs),
            failed_rollouts=sum(1 for x in post_runs if x.ending == "error"),
            pre_rollouts=len(pre_runs),
            failed_pre_rollouts=sum(1 for x in pre_runs if x.ending == "error"),
            post_errors=error_counts(post_runs),
            failed_compactions=sum(1 for r in results if r.compaction_error),
            compaction_errors=tuple(Counter(r.compaction_error for r in results if r.compaction_error).most_common()),
            grown_boundaries=sum(1 for r in results if r.post_size and r.post_size.tokens > r.pre_size.tokens),
        )


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """The verdict on every strategy, over the same boundaries and PRE rollouts."""

    horizon: int
    samples: int
    strategies: Mapping[str, StrategyReport]
    failed_pre_rollouts: int
    pre_rollouts: int = 0
    pre_errors: tuple[tuple[str, int], ...] = ()
    """Distinct PRE errors with their counts, most common first."""

    def summary(self) -> str:
        """A short plain-text table: per strategy, the mean delta at the full horizon."""
        k = self.horizon
        lines = [
            f"compaction verifier: POST minus PRE over the first {k} actions "
            f"({self.samples} samples per arm; * = 95% CI excludes 0)",
            f"PRE rollouts failed: {self.failed_pre_rollouts} of {self.pre_rollouts} (shared by every strategy)",
            *self._sample_note(),
            f"{'strategy':<20}{'n':>7}  {'wasted':>22}  {'blocked':>22}  {'refetch':>22}  "
            f"{'stopped':>22}  {'kept':>5}  {'noop':>4}  {'failed':>9}  {'p':>6}",
        ]
        notes: list[str] = []
        few = False
        for name, report in self.strategies.items():
            tested = len(report.boundaries)
            n = f"{tested - report.unscored_boundaries}/{tested}" if report.unscored_boundaries else f"{tested}"
            mark = "!" if report.failure_asymmetry or report.failed_compactions else ""
            failed = f"{report.failed_rollouts}/{report.post_rollouts}{mark}"
            if not report.horizons:
                why = []
                if report.failed_compactions:
                    why.append(f"compaction failed at {report.failed_compactions} of {tested} boundaries")
                if report.post_rollouts or report.failed_pre_rollouts:
                    why.append(
                        f"{report.failed_rollouts} of {report.post_rollouts} POST and "
                        f"{report.failed_pre_rollouts} of {report.pre_rollouts} PRE rollouts failed"
                    )
                lines.append(f"{name:<20}{n:>7}  no scorable boundaries: {'; '.join(why)}")
            else:
                h = report.at(k)
                few = few or h.boundaries < MIN_BOUNDARIES_FOR_SIGNIFICANCE
                lines.append(
                    f"{name:<20}{n:>7}  {_fmt(h.wasted):>22}  {_fmt(h.blocked):>22}  "
                    f"{_fmt(h.refetch):>22}  {_fmt(h.stopped):>22}  {report.median_token_ratio:>5.0%}  "
                    f"{report.unchanged_boundaries:>4}  {failed:>9}  {report.permutation_p:>6.3f}"
                )
            notes.extend(_failure_notes(name, report))
            notes.extend(_growth_note(name, report))
        lines.append(
            "n = boundaries scored / tested; stopped = share of rollouts that ended before the horizon; "
            "kept = median share of context tokens kept; noop = boundaries left unchanged; "
            "failed = POST rollouts that failed, not scored; "
            "p = exact sign-flip permutation test on per-boundary wasted delta"
        )
        if few:
            lines.append(
                f"fewer than {MIN_BOUNDARIES_FOR_SIGNIFICANCE} boundaries: intervals are shown "
                "without significance marks"
            )
        if self.pre_errors:
            message, count = self.pre_errors[0]
            lines.append(f"PRE: most common error ({count}x): {_clip(message)}")
        lines.extend(notes)
        return "\n".join(lines)

    @property
    def boundaries(self) -> int:
        """Boundaries tested, counted once however many strategies were compared."""
        return len({(b.trajectory, b.cut) for s in self.strategies.values() for b in s.boundaries})

    @property
    def recordings(self) -> int:
        """Recordings the boundaries were cut from."""
        return len({b.trajectory for s in self.strategies.values() for b in s.boundaries})

    def _sample_note(self) -> list[str]:
        boundaries, recordings = self.boundaries, self.recordings
        if not boundaries:
            return []
        note = f"{boundaries} boundar{'y' if boundaries == 1 else 'ies'} from {recordings} recording{'' if recordings == 1 else 's'}"
        if boundaries > recordings:
            note += (
                "; intervals and p treat boundaries as independent, but boundaries from one recording "
                "are correlated, so both are optimistic"
            )
        return [note]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable view, rollouts and actions included."""
        return {
            "horizon": self.horizon,
            "samples": self.samples,
            "boundaries": self.boundaries,
            "recordings": self.recordings,
            "pre_rollouts": self.pre_rollouts,
            "failed_pre_rollouts": self.failed_pre_rollouts,
            "pre_errors": [list(e) for e in self.pre_errors],
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


def error_counts(rollouts: Iterable[Rollout]) -> tuple[tuple[str, int], ...]:
    """Distinct errors of the failed ``rollouts`` with their counts, most common first."""
    return tuple(Counter(r.error or "unknown error" for r in rollouts if r.ending == "error").most_common())


def _failure_notes(name: str, report: StrategyReport) -> list[str]:
    notes = []
    if report.failed_compactions:
        notes.append(
            f"! {name}: compaction failed at {report.failed_compactions} of {len(report.boundaries)} boundaries; "
            "no POST rollout ran there and those boundaries are not scored."
        )
        message, count = report.compaction_errors[0]
        notes.append(f"{name}: most common compaction error ({count}x): {_clip(message)}")
    post = f"POST {report.failed_rollouts}/{report.post_rollouts}"
    pre = f"PRE {report.failed_pre_rollouts}/{report.pre_rollouts}"
    if report.failure_asymmetry == "post":
        consequence = (
            "Failed rollouts are not scored, so the delta covers only the POST samples that ran and may "
            "understate the harm."
            if report.horizons
            else "No boundary could be scored."
        )
        notes.append(f"! {name}: failed more often after compaction ({post} vs {pre}). {consequence}")
    elif report.failure_asymmetry == "pre":
        consequence = (
            "Failed rollouts are not scored, so the delta covers only the PRE samples that ran and may "
            "understate what compaction saves."
            if report.horizons
            else "No boundary could be scored."
        )
        notes.append(f"! {name}: failed more often without compaction ({pre} vs {post}). {consequence}")
    if report.post_errors:
        message, count = report.post_errors[0]
        notes.append(f"{name}: most common POST error ({count}x): {_clip(message)}")
    return notes


def _growth_note(name: str, report: StrategyReport) -> list[str]:
    if not report.grown_boundaries:
        return []
    compacted = len(report.boundaries) - report.failed_compactions
    return [
        f"{name}: the compacted context was longer than the history it replaced at {report.grown_boundaries} "
        f"of {compacted} boundaries (median kept {report.median_token_ratio:.0%})"
    ]


def _clip(message: str, limit: int = 160) -> str:
    message = " ".join(message.split())
    return message if len(message) <= limit else message[: limit - 1] + "…"


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
        "unscored_boundaries": report.unscored_boundaries,
        "post_rollouts": report.post_rollouts,
        "failed_rollouts": report.failed_rollouts,
        "pre_rollouts": report.pre_rollouts,
        "failed_pre_rollouts": report.failed_pre_rollouts,
        "failure_asymmetry": report.failure_asymmetry,
        "post_errors": [list(e) for e in report.post_errors],
        "failed_compactions": report.failed_compactions,
        "compaction_errors": [list(e) for e in report.compaction_errors],
        "grown_boundaries": report.grown_boundaries,
        "horizons": [dataclasses.asdict(h) for h in report.horizons],
        "boundaries": [_boundary_dict(b) for b in report.boundaries],
    }


def _boundary_dict(result: BoundaryResult) -> dict[str, Any]:
    return {
        "trajectory": result.trajectory,
        "cut": result.cut,
        "pre_size": dataclasses.asdict(result.pre_size),
        "post_size": dataclasses.asdict(result.post_size) if result.post_size else None,
        "compaction_tokens": result.compaction_tokens,
        "unchanged": result.unchanged,
        "compaction_error": result.compaction_error,
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
