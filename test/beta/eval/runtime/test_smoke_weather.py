# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test — the v0 acceptance gate.

Runs five tasks via ``TestConfig`` cassettes — no network, no API key —
through the eval framework with five prebuilt scorers plus three custom
ones, and asserts:

* 100% pass rate on every boolean scorer
* score_stats / value_counts populated for the numeric and categorical
  custom scorers
* persisted JSON matches the schema-0.1 shape

The target factory and tool live inline — they're tiny.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("opentelemetry.sdk")

from autogen.beta import Agent, tool
from autogen.beta.eval import (
    BudgetThresholds,
    Suite,
    Trace,
    run_agent,
    scorer,
)
from autogen.beta.eval.scorers import (
    final_answer_matches,
    no_tool_errors,
    token_budget,
    tool_called,
)
from autogen.beta.events import ModelResponse, ToolCallEvent
from autogen.beta.testing import TestConfig

_DATASET_PATH = Path(__file__).parent / "weather_dataset.jsonl"


@tool
async def get_weather(city: str) -> str:
    """Stub weather tool — the eval cares about *how* the agent calls it, not its body."""
    return f"Sunny, 72F in {city}"


def _build_weather_agent(*, config: object = None) -> Agent:
    return Agent(
        "weather",
        prompt=(
            "You are a weather assistant. Use the get_weather tool to "
            "answer questions, and report the result back to the user."
        ),
        config=config,
        tools=[get_weather],
    )


@scorer
def called_get_weather_once(trace: Trace) -> bool:
    """Custom boolean scorer — exactly one call to get_weather."""
    return len(trace.events_of(ToolCallEvent, name="get_weather")) == 1


@scorer
def extra_tool_calls(trace: Trace) -> int:
    """Custom numeric scorer — how many tool calls beyond the expected one?"""
    return max(0, len(trace.events_of(ToolCallEvent)) - 1)


@scorer
def termination_reason(trace: Trace) -> str:
    """Custom categorical scorer — used for slicing, not pass/fail."""
    if trace.exception is not None:
        return "exception"
    responses = trace.events_of(ModelResponse)
    if responses and responses[-1].content:
        return "completed"
    return "empty"


_PREBUILT_SCORERS = [
    tool_called("get_weather"),
    no_tool_errors(),
    final_answer_matches(field="city", matcher="contains"),
    token_budget(2_000),
]

_CUSTOM_SCORERS = [
    called_get_weather_once,
    extra_tool_calls,
    termination_reason,
]


def _cassette(city: str) -> TestConfig:
    """Cassette for one task: model emits a tool call, then a final reply."""
    return TestConfig(
        ToolCallEvent(name="get_weather", arguments='{"city": "' + city + '"}'),
        f"{city} is sunny and 72F today.",
    )


_CASSETTES = {
    "weather-001": _cassette("Melbourne"),
    "weather-002": _cassette("Tokyo"),
    "weather-003": _cassette("Paris"),
    "weather-004": _cassette("Reykjavik"),
    "weather-005": _cassette("São Paulo"),
}


@pytest.mark.asyncio
async def test_smoke_weather_end_to_end(tmp_path: Path) -> None:
    """Acceptance gate — 5 tasks, 8 scorers, 100% pass, persisted JSON validates."""
    suite = Suite.from_jsonl(_DATASET_PATH)

    result = await run_agent(
        suite,
        agent=_build_weather_agent(),
        scorers=[*_PREBUILT_SCORERS, *_CUSTOM_SCORERS],
        model_config=_CASSETTES,
        budgets=BudgetThresholds(max_tokens_per_task=2_000, max_seconds_per_task=10.0),
        concurrency=4,
        store_dir=tmp_path,
    )

    # boolean scorers all pass on every task
    assert result.pass_rate("tool_called[get_weather]") == 1.0
    assert result.pass_rate("no_tool_errors") == 1.0
    assert result.pass_rate("final_answer_matches") == 1.0
    assert result.pass_rate("token_budget") == 1.0
    assert result.pass_rate("called_get_weather_once") == 1.0

    # numeric scorer surfaces in score_stats
    stats = result.score_stats("extra_tool_calls")
    assert stats.n == 5
    assert stats.mean == 0.0
    assert stats.p95 == 0.0

    # categorical scorer surfaces in value_counts
    assert result.value_counts("termination_reason") == {"completed": 5}

    # error / budget counters are zero
    assert result.aggregates.errors == 0
    assert result.aggregates.budget_violations == 0

    # the runner auto-persisted because store_dir was set
    expected_path = tmp_path / f"{result.run_id}.json"
    assert expected_path.exists()

    data = json.loads(expected_path.read_text(encoding="utf-8"))
    _assert_schema_0_1(data)
    assert data["suite"]["size"] == 5
    assert len(data["tasks"]) == 5
    assert data["aggregates"]["pass_rate"]["tool_called[get_weather]"] == 1.0


def _assert_schema_0_1(data: dict) -> None:
    """Verify the schema-0.1 top-level shape by key presence + types.

    Programmatic JSON-schema validation is deferred; for now we check
    every documented field is present and roughly the right type.
    """
    assert data["schema_version"] == "0.1"
    for key in (
        "run_id",
        "label",
        "created_at",
        "duration_ms",
        "suite",
        "target",
        "concurrency",
        "tasks",
        "aggregates",
    ):
        assert key in data, f"missing top-level key: {key}"

    assert {"name", "size", "source"} <= data["suite"].keys()

    for task in data["tasks"]:
        for key in (
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
        ):
            assert key in task, f"missing per-task key: {key}"
        assert {"input", "output", "cache_creation", "cache_read"} <= task["tokens"].keys()
        for fb in task["feedback"]:
            assert {"key", "score", "value", "comment"} <= fb.keys()
        for event in task["events"]:
            assert "type" in event

    for key in ("pass_rate", "score_stats", "value_counts", "tokens", "errors", "budget_violations"):
        assert key in data["aggregates"], f"missing aggregate: {key}"


@pytest.mark.asyncio
async def test_smoke_weather_summary_is_printable(tmp_path: Path) -> None:
    """The summary string contains the things you'd want in a CI log."""
    suite = Suite.from_jsonl(_DATASET_PATH)

    result = await run_agent(
        suite,
        agent=_build_weather_agent(),
        scorers=[*_PREBUILT_SCORERS, *_CUSTOM_SCORERS],
        store_dir=tmp_path,
        model_config=_CASSETTES,
        concurrency=4,
    )

    text = result.summary()

    assert result.run_id in text
    assert "weather_dataset" in text  # suite name (filename stem)
    assert "100.0%" in text  # at least one perfect pass rate rendered
    assert "completed=5" in text  # categorical surfacing
