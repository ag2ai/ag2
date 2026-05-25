# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Eval observability — run / run_variants publish lifecycle events on a stream."""

import pytest

from autogen.beta import Agent
from autogen.beta.eval import Variants, console_reporter, run, run_variants, scorer
from autogen.beta.stream import MemoryStream
from autogen.beta.testing import TestConfig


@scorer
def _ok(outputs) -> bool:
    return bool(outputs.get("body"))


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

    await run(
        [{"task_id": "t1", "inputs": {"input": "hi"}}],
        target=Agent("a", config=TestConfig("ok")),
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
    variants = Variants.from_targets({
        "a": lambda: Agent("a", config=TestConfig("hi a")),
        "b": lambda: Agent("b", config=TestConfig("hi b")),
    })

    await run_variants(
        [{"task_id": "t1", "inputs": {"input": "hi"}}],
        variants=variants,
        scorers=[_ok],
        store_dir=tmp_path,
        stream=stream,
    )

    assert seen.count("VariantStarted") == 2
    assert seen.count("VariantCompleted") == 2


@pytest.mark.asyncio()
async def test_console_reporter_prints_progress(tmp_path, capsys) -> None:
    """The built-in console_reporter, subscribed to a run's stream, prints progress."""
    stream = MemoryStream()
    stream.subscribe(console_reporter, sync_to_thread=False)

    await run(
        [{"task_id": "t1", "inputs": {"input": "hi"}}],
        target=Agent("a", config=TestConfig("ok")),
        scorers=[_ok],
        store_dir=tmp_path,
        stream=stream,
    )

    out = capsys.readouterr().out
    assert "task-run" in out  # EvalStarted line
    assert "t1" in out  # per-task line
