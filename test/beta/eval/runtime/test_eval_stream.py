# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Eval observability — run / run_variants / run_pairwise publish lifecycle events on a stream."""

import pytest

pytest.importorskip("opentelemetry.sdk")

from autogen.beta import Agent
from autogen.beta.eval import Suite, Variants, console_reporter, run_agent, run_pairwise, run_variants, scorer
from autogen.beta.eval.pairwise import PairwiseOutcome
from autogen.beta.stream import MemoryStream
from autogen.beta.testing import TestConfig


@scorer
def _ok(outputs) -> bool:
    return bool(outputs.get("body"))


class _PickB:
    """A PairwiseComparator that always prefers variant B (fixed, for lifecycle tests)."""

    key = "quality"

    async def compare(self, *, task, trace_a, trace_b, reference_outputs) -> PairwiseOutcome:
        return PairwiseOutcome(winner="b")


def _collector() -> tuple[MemoryStream, list[str]]:
    stream = MemoryStream()
    seen: list[str] = []

    async def collect(event) -> None:
        seen.append(type(event).__name__)

    stream.subscribe(collect, sync_to_thread=False)
    return stream, seen


@pytest.mark.asyncio()
async def test_run_publishes_lifecycle_events(tmp_path) -> None:
    stream, seen = _collector()

    await run_agent(
        Suite.from_list([{"task_id": "t1", "inputs": {"input": "hi"}}]),
        agent=Agent("a", config=TestConfig("ok")),
        scorers=[_ok],
        store_dir=tmp_path,
        stream=stream,
        label="my-eval",
    )

    assert seen[0] == "EvalStarted"
    assert "TaskEvaluated" in seen
    assert seen[-1] == "EvalCompleted"


@pytest.mark.asyncio()
async def test_run_variants_publishes_variant_events(tmp_path) -> None:
    stream, seen = _collector()
    variants = Variants({
        "a": Agent("a", config=TestConfig("hi a")),
        "b": Agent("b", config=TestConfig("hi b")),
    })

    await run_variants(
        Suite.from_list([{"task_id": "t1", "inputs": {"input": "hi"}}]),
        variants=variants,
        scorers=[_ok],
        store_dir=tmp_path,
        stream=stream,
    )

    assert seen.count("VariantStarted") == 2
    assert seen.count("VariantCompleted") == 2


@pytest.mark.asyncio()
async def test_run_variants_tags_task_evaluated_with_variant(tmp_path) -> None:
    """Each variant's per-task events flow to the sweep stream, tagged with the variant name."""
    stream = MemoryStream()
    tagged: list[tuple[str, str | None]] = []

    async def collect(event) -> None:
        if type(event).__name__ == "TaskEvaluated":
            tagged.append((event.task_id, event.variant))

    stream.subscribe(collect, sync_to_thread=False)
    variants = Variants({
        "a": Agent("a", config=TestConfig("hi a")),
        "b": Agent("b", config=TestConfig("hi b")),
    })

    await run_variants(
        Suite.from_list([{"task_id": "t1", "inputs": {"input": "hi"}}]),
        variants=variants,
        scorers=[_ok],
        store_dir=tmp_path,
        stream=stream,
    )

    assert ("t1", "a") in tagged
    assert ("t1", "b") in tagged


@pytest.mark.asyncio()
async def test_run_pairwise_publishes_lifecycle_events(tmp_path) -> None:
    stream, seen = _collector()

    await run_pairwise(
        Suite.from_list([{"task_id": "t1", "inputs": {"input": "hi"}}]),
        variant_a=Agent("a", config=TestConfig("from a")),
        variant_b=Agent("b", config=TestConfig("from b")),
        comparators=[_PickB()],
        store_dir=tmp_path,
        stream=stream,
        label="bake-off",
    )

    assert seen[0] == "PairwiseStarted"
    assert "PairwiseCompared" in seen
    assert seen[-1] == "PairwiseCompleted"


@pytest.mark.asyncio()
async def test_console_reporter_prints_progress(tmp_path, capsys) -> None:
    """The built-in console_reporter, subscribed to a run's stream, prints progress."""
    stream = MemoryStream()
    stream.subscribe(console_reporter, sync_to_thread=False)

    await run_agent(
        Suite.from_list([{"task_id": "t1", "inputs": {"input": "hi"}}]),
        agent=Agent("a", config=TestConfig("ok")),
        scorers=[_ok],
        store_dir=tmp_path,
        stream=stream,
    )

    out = capsys.readouterr().out
    assert "task-run" in out  # EvalStarted line
    assert "t1" in out  # per-task line


@pytest.mark.asyncio()
async def test_console_reporter_prints_pairwise_progress(tmp_path, capsys) -> None:
    """The console_reporter renders the pairwise lifecycle events too."""
    stream = MemoryStream()
    stream.subscribe(console_reporter, sync_to_thread=False)

    await run_pairwise(
        Suite.from_list([{"task_id": "t1", "inputs": {"input": "hi"}}]),
        variant_a=Agent("a", config=TestConfig("from a")),
        variant_b=Agent("b", config=TestConfig("from b")),
        comparators=[_PickB()],
        store_dir=tmp_path,
        stream=stream,
    )

    out = capsys.readouterr().out
    assert "comparing" in out  # PairwiseStarted line
    assert "t1" in out  # PairwiseCompared line
