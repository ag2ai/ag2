# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare two runs — regression diffing over the tasks and checks they share.

``RunResult.diff(baseline)`` answers the question evals exist for — "did my change
help or hurt?". It lines the two runs up by ``task_id`` and scorer key, computes
per-scorer pass-rate / mean deltas and the tasks that flipped pass<->fail, and lists
anything that did not line up.

A ``(task, scorer)`` pair is **comparable** only when the ``task_id`` is in both runs,
its ``inputs`` / ``reference_outputs`` are identical (same id with changed content is
*content drift* — never a valid comparison), and the scorer key exists in both. The
delta math runs only over comparable pairs.

By default (``strict=True``) any mismatch — a task or scorer present in only one run,
or content drift — raises :class:`RunsNotComparableError`, whose message points at
``strict=False``. With ``strict=False`` the overlap is diffed and every mismatch is
reported on the :class:`RunDiff` (``content_changed`` / ``only_in_*`` / ``scorers_only_in_*``),
so you are never silently shown an apples-to-oranges number.
"""

from dataclasses import dataclass

from .._types import Feedback
from .result import RunResult, TaskResult

__all__ = (
    "RunDiff",
    "RunsNotComparableError",
    "compute_diff",
)


class RunsNotComparableError(ValueError):
    """Raised by ``RunResult.diff(..., strict=True)`` when the runs didn't grade the same eval.

    The message itemizes every mismatch and notes that ``strict=False`` will diff the
    overlap instead.
    """


@dataclass(frozen=True, slots=True)
class RunDiff:
    """The result of comparing a run against a baseline over what they share.

    ``pass_rate_deltas`` / ``mean_deltas`` map a scorer key to ``(baseline, current)``
    over the comparable tasks. ``flipped_to_fail`` / ``flipped_to_pass`` are
    ``(scorer_key, task_id)`` pairs whose boolean verdict changed. The remaining tuples
    list everything excluded from the comparison.
    """

    current_run_id: str
    baseline_run_id: str
    comparable_tasks: tuple[str, ...]
    pass_rate_deltas: dict[str, tuple[float, float]]
    mean_deltas: dict[str, tuple[float, float]]
    flipped_to_fail: tuple[tuple[str, str], ...]
    flipped_to_pass: tuple[tuple[str, str], ...]
    only_in_current: tuple[str, ...]
    only_in_baseline: tuple[str, ...]
    content_changed: tuple[str, ...]
    scorers_only_in_current: tuple[str, ...]
    scorers_only_in_baseline: tuple[str, ...]

    @property
    def regressions(self) -> tuple[tuple[str, str], ...]:
        """``(scorer_key, task_id)`` pairs that flipped pass -> fail — the CI gate (``assert not diff.regressions``)."""
        return self.flipped_to_fail

    def summary(self) -> str:
        """A printable diff: per-scorer deltas, the flips, and everything excluded."""
        lines = [
            f"Diff {self.current_run_id} vs {self.baseline_run_id}  —  {len(self.comparable_tasks)} comparable task(s)"
        ]
        for key in sorted(self.pass_rate_deltas):
            base, cur = self.pass_rate_deltas[key]
            mark = "   REGRESSION" if cur < base else ""
            lines.append(f"  {key:<24} {base * 100:5.1f}% -> {cur * 100:5.1f}%   {(cur - base) * 100:+5.1f}{mark}")
        for key in sorted(self.mean_deltas):
            base, cur = self.mean_deltas[key]
            lines.append(f"  {key:<24} mean {base:.2f} -> {cur:.2f}   {cur - base:+.2f}")
        if self.flipped_to_fail:
            lines.append(f"  flipped pass->fail: {[f'{k}:{t}' for k, t in self.flipped_to_fail]}")
        if self.flipped_to_pass:
            lines.append(f"  flipped fail->pass: {[f'{k}:{t}' for k, t in self.flipped_to_pass]}")
        excluded = _excluded_lines(self)
        if excluded:
            lines.append("  — excluded (not comparable) —")
            lines.extend(excluded)
        return "\n".join(lines)


def compute_diff(current: RunResult, baseline: RunResult, *, strict: bool = True) -> RunDiff:
    """Compare ``current`` against ``baseline``; see :class:`RunDiff` and :meth:`RunResult.diff`."""
    cur = {tr.task.task_id: tr for tr in current.tasks}
    base = {tr.task.task_id: tr for tr in baseline.tasks}

    only_in_current = tuple(sorted(set(cur) - set(base)))
    only_in_baseline = tuple(sorted(set(base) - set(cur)))
    shared = set(cur) & set(base)
    content_changed = tuple(
        sorted(
            tid
            for tid in shared
            if cur[tid].task.inputs != base[tid].task.inputs
            or cur[tid].task.reference_outputs != base[tid].task.reference_outputs
        )
    )
    comparable = sorted(shared - set(content_changed))

    cur_keys = _keys(current.tasks)
    base_keys = _keys(baseline.tasks)
    scorers_only_in_current = tuple(sorted(cur_keys - base_keys))
    scorers_only_in_baseline = tuple(sorted(base_keys - cur_keys))
    shared_keys = sorted(cur_keys & base_keys)

    if strict:
        problems = _problems(
            only_in_current, only_in_baseline, content_changed, scorers_only_in_current, scorers_only_in_baseline
        )
        if problems:
            raise RunsNotComparableError(_message(problems))

    cur_fb = {tid: _by_key(cur[tid]) for tid in comparable}
    base_fb = {tid: _by_key(base[tid]) for tid in comparable}

    pass_rate_deltas: dict[str, tuple[float, float]] = {}
    mean_deltas: dict[str, tuple[float, float]] = {}
    flipped_to_fail: list[tuple[str, str]] = []
    flipped_to_pass: list[tuple[str, str]] = []

    for key in shared_keys:
        base_bools = [s for t in comparable if (s := _bool(base_fb[t].get(key))) is not None]
        cur_bools = [s for t in comparable if (s := _bool(cur_fb[t].get(key))) is not None]
        if base_bools or cur_bools:
            pass_rate_deltas[key] = (_rate(base_bools), _rate(cur_bools))

        base_nums = [float(base_fb[t][key].score) for t in comparable if _is_num(base_fb[t].get(key))]
        cur_nums = [float(cur_fb[t][key].score) for t in comparable if _is_num(cur_fb[t].get(key))]
        if base_nums or cur_nums:
            mean_deltas[key] = (_mean(base_nums), _mean(cur_nums))

        for t in comparable:
            b, c = base_fb[t].get(key), cur_fb[t].get(key)
            if b is None or c is None or not isinstance(b.score, bool) or not isinstance(c.score, bool):
                continue
            if b.score and not c.score:
                flipped_to_fail.append((key, t))
            elif not b.score and c.score:
                flipped_to_pass.append((key, t))

    return RunDiff(
        current_run_id=current.run_id,
        baseline_run_id=baseline.run_id,
        comparable_tasks=tuple(comparable),
        pass_rate_deltas=pass_rate_deltas,
        mean_deltas=mean_deltas,
        flipped_to_fail=tuple(flipped_to_fail),
        flipped_to_pass=tuple(flipped_to_pass),
        only_in_current=only_in_current,
        only_in_baseline=only_in_baseline,
        content_changed=content_changed,
        scorers_only_in_current=scorers_only_in_current,
        scorers_only_in_baseline=scorers_only_in_baseline,
    )


def _keys(tasks: tuple[TaskResult, ...]) -> set[str]:
    return {fb.key for tr in tasks for fb in tr.feedback}


def _by_key(tr: TaskResult) -> dict[str, Feedback]:
    return {fb.key: fb for fb in tr.feedback}


def _bool(fb: Feedback | None) -> bool | None:
    return fb.score if fb is not None and isinstance(fb.score, bool) else None


def _is_num(fb: Feedback | None) -> bool:
    return fb is not None and isinstance(fb.score, (int, float)) and not isinstance(fb.score, bool)


def _rate(scores: list[bool]) -> float:
    return sum(1 for s in scores if s) / len(scores) if scores else 0.0


def _mean(nums: list[float]) -> float:
    return sum(nums) / len(nums) if nums else 0.0


def _problems(
    only_in_current: tuple[str, ...],
    only_in_baseline: tuple[str, ...],
    content_changed: tuple[str, ...],
    scorers_only_in_current: tuple[str, ...],
    scorers_only_in_baseline: tuple[str, ...],
) -> list[str]:
    out: list[str] = []
    if content_changed:
        out.append(f"content changed for {len(content_changed)} task(s): {list(content_changed)}")
    if only_in_baseline:
        out.append(f"{len(only_in_baseline)} task(s) only in baseline: {list(only_in_baseline)}")
    if only_in_current:
        out.append(f"{len(only_in_current)} task(s) only in current: {list(only_in_current)}")
    if scorers_only_in_baseline:
        out.append(f"scorer(s) only in baseline: {list(scorers_only_in_baseline)}")
    if scorers_only_in_current:
        out.append(f"scorer(s) only in current: {list(scorers_only_in_current)}")
    return out


def _message(problems: list[str]) -> str:
    body = "\n".join(f"  • {p}" for p in problems)
    return (
        "these runs didn't grade the same eval —\n"
        f"{body}\n"
        "Pass strict=False to diff the overlap anyway (the mismatches are reported on the RunDiff)."
    )


def _excluded_lines(diff: RunDiff) -> list[str]:
    out: list[str] = []
    if diff.content_changed:
        out.append(f"  content changed: {list(diff.content_changed)}")
    if diff.only_in_current:
        out.append(f"  only in current: {list(diff.only_in_current)}")
    if diff.only_in_baseline:
        out.append(f"  only in baseline: {list(diff.only_in_baseline)}")
    if diff.scorers_only_in_current:
        out.append(f"  scorer only in current: {list(diff.scorers_only_in_current)}")
    if diff.scorers_only_in_baseline:
        out.append(f"  scorer only in baseline: {list(diff.scorers_only_in_baseline)}")
    return out
