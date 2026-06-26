# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Public-API tests for :class:`RunResult` — aggregation math, summary, save."""

import json
from pathlib import Path

import pytest

from ag2.eval import (
    Aggregates,
    Feedback,
    RunResult,
    ScoreStats,
    Suite,
    Task,
    TaskResult,
    Trace,
)


def _empty_trace(duration_ms: int = 0) -> Trace:
    return Trace(events=[], exception=None, duration_ms=duration_ms)


def _task_result(
    task_id: str,
    feedback: tuple[Feedback, ...],
    *,
    budget_violation: bool = False,
    exception: BaseException | None = None,
    tags: tuple[str, ...] = (),
) -> TaskResult:
    return TaskResult(
        task=Task(task_id=task_id, inputs={"input": "?"}, tags=tags),
        trace=Trace(events=[], exception=exception, duration_ms=0),
        feedback=feedback,
        budget_violation=budget_violation,
    )


def _result(*task_results: TaskResult, run_id: str = "test-run") -> RunResult:
    suite = Suite.from_list([{"task_id": tr.task.task_id, "inputs": tr.task.inputs} for tr in task_results])
    return RunResult(
        run_id=run_id,
        tasks=tuple(task_results),
        suite=suite,
        target_path="test_module:test_factory",
        concurrency=1,
        duration_ms=42,
        created_at="2026-05-11T00:00:00+00:00",
    )


class TestPassRate:
    def test_all_pass(self) -> None:
        result = _result(
            _task_result("t1", (Feedback(key="check", score=True),)),
            _task_result("t2", (Feedback(key="check", score=True),)),
        )
        assert result.pass_rate("check") == 1.0

    def test_partial_pass(self) -> None:
        result = _result(
            _task_result("t1", (Feedback(key="check", score=True),)),
            _task_result("t2", (Feedback(key="check", score=False),)),
            _task_result("t3", (Feedback(key="check", score=True),)),
            _task_result("t4", (Feedback(key="check", score=False),)),
        )
        assert result.pass_rate("check") == 0.5

    def test_zero_pass(self) -> None:
        result = _result(_task_result("t1", (Feedback(key="check", score=False),)))
        assert result.pass_rate("check") == 0.0

    def test_unknown_key_returns_zero(self) -> None:
        result = _result(_task_result("t1", (Feedback(key="check", score=True),)))
        assert result.pass_rate("does-not-exist") == 0.0

    def test_numeric_scores_do_not_contribute_to_pass_rate(self) -> None:
        """A scorer that returns 5 is not 'pass=True'. Pass rate stays empty for that key."""
        result = _result(_task_result("t1", (Feedback(key="counter", score=5),)))
        assert result.pass_rate("counter") == 0.0


class TestScoreStats:
    def test_basic_stats(self) -> None:
        result = _result(
            _task_result("t1", (Feedback(key="extra", score=0),)),
            _task_result("t2", (Feedback(key="extra", score=1),)),
            _task_result("t3", (Feedback(key="extra", score=2),)),
        )

        stats = result.score_stats("extra")

        assert stats.n == 3
        assert stats.mean == 1.0
        assert stats.p50 == 1.0

    def test_float_scores_supported(self) -> None:
        result = _result(
            _task_result("t1", (Feedback(key="ratio", score=0.25),)),
            _task_result("t2", (Feedback(key="ratio", score=0.75),)),
        )
        stats = result.score_stats("ratio")
        assert stats.n == 2
        assert stats.mean == pytest.approx(0.5)

    def test_unknown_key_returns_empty_stats(self) -> None:
        result = _result(_task_result("t1", (Feedback(key="x", score=1),)))
        stats = result.score_stats("not-there")
        assert stats == ScoreStats(mean=0.0, p50=0.0, p95=0.0, n=0)

    def test_bool_feedback_does_not_enter_score_stats(self) -> None:
        """bool is a subclass of int but should not become numeric score stats."""
        result = _result(_task_result("t1", (Feedback(key="b", score=True),)))
        assert result.score_stats("b").n == 0


class TestValueCounts:
    def test_counts_labels(self) -> None:
        result = _result(
            _task_result("t1", (Feedback(key="reason", value="completed"),)),
            _task_result("t2", (Feedback(key="reason", value="completed"),)),
            _task_result("t3", (Feedback(key="reason", value="halted"),)),
        )
        assert result.value_counts("reason") == {"completed": 2, "halted": 1}

    def test_unknown_key_returns_empty_dict(self) -> None:
        result = _result(_task_result("t1", (Feedback(key="x", value="a"),)))
        assert result.value_counts("not-there") == {}


class TestAggregates:
    def test_aggregates_property_exposes_everything(self) -> None:
        result = _result(
            _task_result("t1", (Feedback(key="ok", score=True), Feedback(key="why", value="completed"))),
            _task_result("t2", (Feedback(key="ok", score=False), Feedback(key="why", value="halted"))),
            _task_result("t3", (Feedback(key="ok", score=True), Feedback(key="why", value="completed"))),
        )

        aggs = result.aggregates

        assert isinstance(aggs, Aggregates)
        assert aggs.pass_rate == {"ok": pytest.approx(2 / 3)}
        assert aggs.value_counts == {"why": {"completed": 2, "halted": 1}}
        assert aggs.errors == 0
        assert aggs.budget_violations == 0

    def test_errors_counted(self) -> None:
        result = _result(
            _task_result("t1", (), exception=RuntimeError("boom")),
            _task_result("t2", (Feedback(key="ok", score=True),)),
        )
        assert result.aggregates.errors == 1

    def test_budget_violations_counted(self) -> None:
        result = _result(
            _task_result("t1", (Feedback(key="ok", score=True),), budget_violation=True),
            _task_result("t2", (Feedback(key="ok", score=True),), budget_violation=False),
        )
        assert result.aggregates.budget_violations == 1


class TestTagSlicing:
    def test_pass_rate_by_tag(self) -> None:
        result = _result(
            _task_result("t1", (Feedback(key="check", score=True),), tags=("easy",)),
            _task_result("t2", (Feedback(key="check", score=True),), tags=("easy",)),
            _task_result("t3", (Feedback(key="check", score=False),), tags=("hard",)),
            _task_result("t4", (Feedback(key="check", score=False),), tags=("hard",)),
        )

        assert result.pass_rate("check") == 0.5  # whole run
        assert result.pass_rate("check", tag="easy") == 1.0
        assert result.pass_rate("check", tag="hard") == 0.0

    def test_value_counts_by_tag(self) -> None:
        result = _result(
            _task_result("t1", (Feedback(key="reason", value="completed"),), tags=("easy",)),
            _task_result("t2", (Feedback(key="reason", value="error"),), tags=("hard",)),
        )

        assert result.value_counts("reason", tag="easy") == {"completed": 1}
        assert result.value_counts("reason", tag="hard") == {"error": 1}

    def test_tags_lists_all_present(self) -> None:
        result = _result(
            _task_result("t1", (Feedback(key="check", score=True),), tags=("easy", "smoke")),
            _task_result("t2", (Feedback(key="check", score=True),), tags=("hard",)),
        )

        assert result.tags == frozenset({"easy", "smoke", "hard"})

    def test_unknown_tag_is_an_empty_slice(self) -> None:
        result = _result(_task_result("t1", (Feedback(key="check", score=True),), tags=("easy",)))

        assert result.pass_rate("check", tag="nope") == 0.0


class TestSummary:
    def test_includes_run_id_duration_and_pass_rate(self) -> None:
        result = _result(
            _task_result("t1", (Feedback(key="ok", score=True),)),
            _task_result("t2", (Feedback(key="ok", score=False),)),
            run_id="abc123",
        )

        summary = result.summary()

        assert "abc123" in summary
        assert "42ms" in summary
        assert "ok" in summary
        assert "50.0% (1/2)" in summary  # pass-rate shows the (passed/total) denominator
        assert "Runs:" in summary  # number of runs executed is surfaced

    def test_renders_score_stats_and_value_counts(self) -> None:
        result = _result(
            _task_result("t1", (Feedback(key="count", score=2), Feedback(key="end", value="completed"))),
            _task_result("t2", (Feedback(key="count", score=4), Feedback(key="end", value="completed"))),
        )

        summary = result.summary()

        assert "count" in summary
        assert "mean=3.00" in summary
        assert "completed=2" in summary


class TestSave:
    def test_save_to_directory(self, tmp_path: Path) -> None:
        result = _result(
            _task_result("t1", (Feedback(key="ok", score=True),)),
            run_id="my-run-id",
        )

        written = result.save(tmp_path)

        assert written == tmp_path / "my-run-id.json"
        assert written.exists()
        data = json.loads(written.read_text(encoding="utf-8"))
        assert data["run_id"] == "my-run-id"
        assert data["schema_version"] == "0.1"

    def test_save_to_explicit_json_path(self, tmp_path: Path) -> None:
        result = _result(_task_result("t1", (Feedback(key="ok", score=True),)))
        target = tmp_path / "subdir" / "custom.json"

        written = result.save(target)

        assert written == target
        assert target.exists()

    def test_save_with_store_dir_set_at_construction(self, tmp_path: Path) -> None:
        suite = Suite.from_list([{"task_id": "t1", "inputs": {"input": "?"}}])
        result = RunResult(
            run_id="cfg-run",
            tasks=(_task_result("t1", (Feedback(key="ok", score=True),)),),
            suite=suite,
            target_path="x:y",
            concurrency=1,
            duration_ms=1,
            created_at="2026-05-11T00:00:00+00:00",
            store_dir=tmp_path,
        )

        written = result.save()  # no path argument

        assert written == tmp_path / "cfg-run.json"
        assert written.exists()

    def test_save_without_path_or_store_dir_raises(self) -> None:
        result = _result(_task_result("t1", (Feedback(key="ok", score=True),)))

        with pytest.raises(ValueError, match="no path given and no store_dir"):
            result.save()


class TestRunResultProperties:
    def test_run_id_and_schema(self) -> None:
        result = _result(_task_result("t1", ()), run_id="r1")
        assert result.run_id == "r1"
        assert result.schema_version == "0.1"

    def test_tasks_round_trip(self) -> None:
        tr = _task_result("t1", (Feedback(key="ok", score=True),))
        result = _result(tr)
        assert result.tasks == (tr,)
