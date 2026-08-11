# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Test run_pairwise: produce two variants over a suite, then compare them."""

import json

import pytest

pytest.importorskip("opentelemetry.sdk")

from ag2 import Agent
from ag2.eval import Suite, run_pairwise
from ag2.eval.pairwise import PairwiseOutcome
from ag2.testing import TestConfig


class _Scripted:
    """A PairwiseComparator returning a fixed winner per task_id."""

    def __init__(self, key: str, wins: dict[str, str]) -> None:
        self.key = key
        self._wins = wins

    async def compare(self, *, task, trace_a, trace_b, reference_outputs) -> PairwiseOutcome:
        return PairwiseOutcome(winner=self._wins.get(task.task_id, "tie"))


def _build_a(*, config=None) -> Agent:
    return Agent("variant-a", config=config or TestConfig("answer from A"))


def _build_b(*, config=None) -> Agent:
    return Agent("variant-b", config=config or TestConfig("answer from B"))


@pytest.mark.asyncio()
async def test_run_pairwise_produces_then_compares(tmp_path) -> None:
    suite = Suite.from_list([
        {"task_id": "t1", "inputs": {"input": "Q1?"}},
        {"task_id": "t2", "inputs": {"input": "Q2?"}},
    ])
    judge = _Scripted("quality", {"t1": "b", "t2": "b"})

    result = await run_pairwise(
        suite,
        variant_a=_build_a(),
        variant_b=_build_b(),
        comparators=[judge],
        variant_a_name="v1",
        variant_b_name="v2",
        store_dir=tmp_path,
    )

    assert result.tally("quality") == (0, 2, 0)  # B wins both produced pairs
    assert result.win_rate("quality").rate == 1.0
    assert (tmp_path / f"{result.run_id}.json").exists()


@pytest.mark.asyncio()
async def test_run_pairwise_accepts_agent_instances(tmp_path) -> None:
    """``variant_a`` / ``variant_b`` accept prebuilt Agent instances, not just factories."""
    suite = Suite.from_list([{"task_id": "t1", "inputs": {"input": "Q?"}}])
    judge = _Scripted("quality", {"t1": "b"})

    result = await run_pairwise(
        suite,
        variant_a=Agent("variant-a", config=TestConfig("answer from A")),
        variant_b=Agent("variant-b", config=TestConfig("answer from B")),
        comparators=[judge],
        store_dir=tmp_path,
    )

    assert result.tally("quality") == (0, 1, 0)
    assert result.win_rate("quality").rate == 1.0


@pytest.mark.asyncio()
async def test_run_pairwise_label_is_recorded_and_serialized(tmp_path) -> None:
    """A user-defined ``label`` is carried on the result and persisted to the run JSON."""
    suite = Suite.from_list([{"task_id": "t1", "inputs": {"input": "Q?"}}])
    judge = _Scripted("quality", {"t1": "b"})

    result = await run_pairwise(
        suite,
        variant_a=_build_a(),
        variant_b=_build_b(),
        comparators=[judge],
        store_dir=tmp_path,
        label="bake-off",
    )

    assert result.label == "bake-off"
    data = json.loads((tmp_path / f"{result.run_id}.json").read_text(encoding="utf-8"))
    assert data["label"] == "bake-off"
