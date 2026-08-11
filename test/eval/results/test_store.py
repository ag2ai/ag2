# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Public-API tests for store-layer serialization — schema 0.1 shape + JSON round-trip.

The store layer is what a future hosted dashboard will read, so the
schema shape is locked. Tests verify every documented field is present
and that the dict survives a ``json.dumps`` / ``json.loads`` round-trip
unchanged.
"""

import json
from pathlib import Path

from ag2.eval import (
    Feedback,
    RunResult,
    Suite,
    TaskResult,
    Trace,
    TraceRef,
)
from ag2.eval.results.store import load_run, to_dict
from ag2.events import HumanMessage, ModelResponse, ToolCallEvent, Usage


def _make_result(
    *,
    events: list = (),
    feedback: tuple[Feedback, ...] = (Feedback(key="ok", score=True),),
    exception: BaseException | None = None,
    budget_violation: bool = False,
    run_id: str = "test-run",
    trace_ref: TraceRef | None = None,
) -> RunResult:
    suite = Suite.from_list([
        {
            "task_id": "t1",
            "inputs": {"input": "Tokyo?"},
            "reference_outputs": {"city": "Tokyo"},
            "tags": ["happy-path"],
            "metadata": {"difficulty": "easy"},
        }
    ])
    trace = Trace(events=list(events), exception=exception, duration_ms=123)
    task_result = TaskResult(
        task=suite.tasks[0],
        trace=trace,
        feedback=feedback,
        budget_violation=budget_violation,
        trace_ref=trace_ref,
    )
    return RunResult(
        run_id=run_id,
        tasks=(task_result,),
        suite=suite,
        target_path="weather_agent.agent:build_weather_agent",
        concurrency=4,
        duration_ms=999,
        created_at="2026-05-11T14:23:00+00:00",
    )


class TestTopLevelSchema:
    def test_top_level_keys_present(self) -> None:
        data = to_dict(_make_result())
        assert set(data.keys()) == {
            "schema_version",
            "run_id",
            "label",
            "created_at",
            "duration_ms",
            "suite",
            "target",
            "concurrency",
            "tasks",
            "aggregates",
        }

    def test_schema_version_is_zero_one(self) -> None:
        data = to_dict(_make_result())
        assert data["schema_version"] == "0.1"

    def test_suite_block_shape(self) -> None:
        data = to_dict(_make_result())
        assert data["suite"] == {"name": "inline", "size": 1, "source": "inline"}


class TestTaskSerialization:
    def test_task_keys_present(self) -> None:
        data = to_dict(_make_result())
        [task] = data["tasks"]
        assert {
            "task_id",
            "inputs",
            "reference_outputs",
            "tags",
            "metadata",
            "duration_ms",
            "events",
            "exception",
            "tokens",
            "feedback",
        }.issubset(task.keys())

    def test_inputs_and_reference_outputs_passthrough(self) -> None:
        data = to_dict(_make_result())
        [task] = data["tasks"]
        assert task["inputs"] == {"input": "Tokyo?"}
        assert task["reference_outputs"] == {"city": "Tokyo"}
        assert task["tags"] == ["happy-path"]
        assert task["metadata"] == {"difficulty": "easy"}

    def test_exception_serialized_with_type_and_message(self) -> None:
        data = to_dict(_make_result(exception=ValueError("kaboom")))
        [task] = data["tasks"]
        assert task["exception"] == {"type": "ValueError", "message": "kaboom"}

    def test_no_exception_is_null(self) -> None:
        data = to_dict(_make_result())
        [task] = data["tasks"]
        assert task["exception"] is None

    def test_tokens_block_present_per_task(self) -> None:
        data = to_dict(_make_result())
        [task] = data["tasks"]
        assert task["tokens"] == {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0}

    def test_feedback_serialized_as_list_of_records(self) -> None:
        data = to_dict(
            _make_result(
                feedback=(
                    Feedback(key="check", score=True),
                    Feedback(key="reason", value="completed", comment="finished cleanly"),
                )
            )
        )
        [task] = data["tasks"]
        assert task["feedback"] == [
            {"key": "check", "score": True, "value": None, "comment": None, "detail": None},
            {"key": "reason", "score": None, "value": "completed", "comment": "finished cleanly", "detail": None},
        ]


class TestEventSerialization:
    def test_each_event_has_a_type_field(self) -> None:
        events = [
            ToolCallEvent(name="get_weather", arguments='{"city": "Tokyo"}'),
            ModelResponse(usage=Usage(prompt_tokens=10, completion_tokens=5)),
            HumanMessage("noted"),
        ]
        data = to_dict(_make_result(events=events))
        [task] = data["tasks"]
        types = [e["type"] for e in task["events"]]
        assert types == ["ToolCallEvent", "ModelResponse", "HumanMessage"]

    def test_event_order_preserved(self) -> None:
        first = ToolCallEvent(name="a", arguments="{}")
        second = ToolCallEvent(name="b", arguments="{}")
        data = to_dict(_make_result(events=[first, second]))
        [task] = data["tasks"]
        assert [e.get("name") for e in task["events"]] == ["a", "b"]


class TestAggregates:
    def test_aggregates_block_has_documented_keys(self) -> None:
        data = to_dict(_make_result())
        aggs = data["aggregates"]
        assert {
            "pass_rate",
            "score_stats",
            "value_counts",
            "tokens",
            "errors",
            "budget_violations",
        }.issubset(aggs.keys())

    def test_pass_rate_reflects_boolean_feedback(self) -> None:
        result = _make_result(feedback=(Feedback(key="ok", score=True),))
        data = to_dict(result)
        assert data["aggregates"]["pass_rate"] == {"ok": 1.0}

    def test_value_counts_reflects_categorical_feedback(self) -> None:
        result = _make_result(feedback=(Feedback(key="end", value="completed"),))
        data = to_dict(result)
        assert data["aggregates"]["value_counts"] == {"end": {"completed": 1}}

    def test_budget_violations_counted(self) -> None:
        result = _make_result(budget_violation=True)
        data = to_dict(result)
        assert data["aggregates"]["budget_violations"] == 1

    def test_aggregate_tokens_shape(self) -> None:
        data = to_dict(_make_result())
        assert set(data["aggregates"]["tokens"].keys()) == {"input", "output", "total"}


class TestRoundTrip:
    def test_dict_survives_json_round_trip(self) -> None:
        result = _make_result(
            events=[
                ToolCallEvent(name="get_weather", arguments='{"city": "Tokyo"}'),
                ModelResponse(usage=Usage(prompt_tokens=10, completion_tokens=5)),
            ],
            feedback=(
                Feedback(key="check", score=True),
                Feedback(key="end", value="completed"),
            ),
        )

        original = to_dict(result)
        roundtripped = json.loads(json.dumps(original, default=str))

        assert roundtripped == original

    def test_write_and_read_back_from_disk(self, tmp_path: Path) -> None:
        result = _make_result()
        target = tmp_path / "run.json"

        written = result.save(target)

        assert written == target
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["run_id"] == "test-run"
        assert data["schema_version"] == "0.1"


class TestTraceRefSerialization:
    def test_trace_ref_serialized_when_present(self) -> None:
        ref = TraceRef(trace_id="abc123", task_id="t1", metadata={"k": "v"})
        data = to_dict(_make_result(trace_ref=ref))
        [task] = data["tasks"]
        assert task["trace_ref"] == {"trace_id": "abc123", "task_id": "t1", "metadata": {"k": "v"}}

    def test_trace_ref_is_null_when_absent(self) -> None:
        data = to_dict(_make_result())
        [task] = data["tasks"]
        assert task["trace_ref"] is None

    def test_trace_ref_round_trips_through_disk(self, tmp_path: Path) -> None:
        """save then load_run restores trace_ref (trace_id / task_id / metadata)."""
        ref = TraceRef(trace_id="otel-deadbeef", task_id="t1", metadata={"region": "us"})
        target = tmp_path / "run.json"
        _make_result(trace_ref=ref).save(target)

        loaded = load_run(target)

        assert loaded.tasks[0].trace_ref == ref
