# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Public-API tests for run comparison — ``load_run`` and ``RunResult.diff``."""

import pytest

from ag2.eval import (
    Feedback,
    RunResult,
    RunsNotComparableError,
    Suite,
    Task,
    TaskResult,
    Trace,
    load_run,
)


def _tr(
    task_id: str, feedback: tuple[Feedback, ...], *, inputs: dict | None = None, tags: tuple[str, ...] = ()
) -> TaskResult:
    return TaskResult(
        task=Task(task_id=task_id, inputs=inputs or {"input": "?"}, tags=tags),
        trace=Trace(events=[], exception=None, duration_ms=0),
        feedback=feedback,
    )


def _run(*task_results: TaskResult, run_id: str = "r", label: str | None = None) -> RunResult:
    suite = Suite(tuple(tr.task for tr in task_results), name="s", source="inline")
    return RunResult(
        run_id=run_id,
        tasks=tuple(task_results),
        suite=suite,
        target_path="m:f",
        concurrency=1,
        duration_ms=1,
        created_at="2026-01-01T00:00:00+00:00",
        label=label,
    )


class TestLoadRun:
    def test_round_trips_scores_and_identity(self, tmp_path) -> None:
        result = _run(
            _tr("t1", (Feedback(key="check", score=True),), tags=("easy",)),
            _tr("t2", (Feedback(key="check", score=False),)),
            run_id="run-1",
            label="nightly",
        )
        path = result.save(tmp_path)

        loaded = load_run(path)

        assert loaded.run_id == "run-1"
        assert loaded.label == "nightly"
        assert loaded.pass_rate("check") == 0.5
        assert loaded.pass_rate("check", tag="easy") == 1.0  # tags survive the round-trip
        assert {tr.task.task_id for tr in loaded.tasks} == {"t1", "t2"}

    def test_loaded_run_diffs_against_a_live_run(self, tmp_path) -> None:
        baseline = _run(_tr("t1", (Feedback(key="check", score=True),)), run_id="base")
        path = baseline.save(tmp_path)

        current = _run(_tr("t1", (Feedback(key="check", score=False),)), run_id="cur")
        delta = current.diff(load_run(path))

        assert delta.regressions == (("check", "t1"),)


class TestDiff:
    def test_identical_runs_have_no_regressions(self) -> None:
        diff = _run(_tr("t1", (Feedback(key="check", score=True),))).diff(
            _run(_tr("t1", (Feedback(key="check", score=True),)))
        )
        assert diff.regressions == ()
        assert diff.pass_rate_deltas["check"] == (1.0, 1.0)

    def test_flip_pass_to_fail_is_a_regression(self) -> None:
        baseline = _run(_tr("t1", (Feedback(key="check", score=True),)))
        current = _run(_tr("t1", (Feedback(key="check", score=False),)))

        diff = current.diff(baseline)

        assert diff.regressions == (("check", "t1"),)
        assert diff.pass_rate_deltas["check"] == (1.0, 0.0)

    def test_flip_fail_to_pass_is_not_a_regression(self) -> None:
        baseline = _run(_tr("t1", (Feedback(key="check", score=False),)))
        current = _run(_tr("t1", (Feedback(key="check", score=True),)))

        diff = current.diff(baseline)

        assert diff.regressions == ()
        assert diff.flipped_to_pass == (("check", "t1"),)

    def test_numeric_scorer_reports_mean_delta(self) -> None:
        baseline = _run(_tr("t1", (Feedback(key="cost", score=10),)), _tr("t2", (Feedback(key="cost", score=20),)))
        current = _run(_tr("t1", (Feedback(key="cost", score=5),)), _tr("t2", (Feedback(key="cost", score=5),)))

        diff = current.diff(baseline)

        assert diff.mean_deltas["cost"] == (15.0, 5.0)
        assert diff.regressions == ()  # numeric moves aren't pass/fail flips

    def test_strict_raises_on_extra_task(self) -> None:
        baseline = _run(_tr("t1", (Feedback(key="check", score=True),)))
        current = _run(
            _tr("t1", (Feedback(key="check", score=True),)),
            _tr("t2", (Feedback(key="check", score=True),)),
        )
        with pytest.raises(RunsNotComparableError, match="strict=False"):
            current.diff(baseline)

    def test_strict_false_diffs_overlap_and_reports_the_rest(self) -> None:
        baseline = _run(_tr("t1", (Feedback(key="check", score=True),)))
        current = _run(
            _tr("t1", (Feedback(key="check", score=False),)),
            _tr("t2", (Feedback(key="check", score=True),)),
        )

        diff = current.diff(baseline, strict=False)

        assert diff.comparable_tasks == ("t1",)
        assert diff.only_in_current == ("t2",)
        assert diff.regressions == (("check", "t1"),)

    def test_strict_raises_on_content_drift(self) -> None:
        baseline = _run(_tr("t1", (Feedback(key="check", score=True),), inputs={"input": "A"}))
        current = _run(_tr("t1", (Feedback(key="check", score=True),), inputs={"input": "B"}))
        with pytest.raises(RunsNotComparableError, match="content changed"):
            current.diff(baseline)

    def test_content_drift_is_excluded_under_non_strict(self) -> None:
        baseline = _run(_tr("t1", (Feedback(key="check", score=True),), inputs={"input": "A"}))
        current = _run(_tr("t1", (Feedback(key="check", score=False),), inputs={"input": "B"}))

        diff = current.diff(baseline, strict=False)

        assert diff.content_changed == ("t1",)
        assert diff.comparable_tasks == ()
        assert diff.regressions == ()  # a task whose question changed is not a regression

    def test_strict_raises_on_added_scorer(self) -> None:
        baseline = _run(_tr("t1", (Feedback(key="a", score=True),)))
        current = _run(_tr("t1", (Feedback(key="a", score=True), Feedback(key="b", score=True))))
        with pytest.raises(RunsNotComparableError, match="scorer"):
            current.diff(baseline)

        diff = current.diff(baseline, strict=False)
        assert diff.scorers_only_in_current == ("b",)
        assert "a" in diff.pass_rate_deltas
