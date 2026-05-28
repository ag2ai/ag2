# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for run_variants — typed single-axis variant comparison + leaderboard."""

import pytest

pytest.importorskip("opentelemetry.sdk")

from autogen.beta import Agent
from autogen.beta.eval import Variants, run_variants, scorer
from autogen.beta.eval.scorers import final_answer_matches
from autogen.beta.testing import TestConfig


def _build(*, config=None) -> Agent:
    return Agent("variant-agent", config=config)


@scorer
def got_reply(outputs) -> bool:
    return bool(outputs.get("body"))


@pytest.mark.asyncio()
async def test_from_configs_runs_each_variant_and_ranks(tmp_path) -> None:
    variants = Variants.from_configs(
        _build,
        {
            "right": TestConfig("The capital is Tokyo."),
            "wrong": TestConfig("The capital is Paris."),
        },
    )

    board = await run_variants(
        [{"task_id": "t1", "inputs": {"input": "Capital of Japan?"}, "reference_outputs": {"answer": "Tokyo"}}],
        variants=variants,
        scorers=[final_answer_matches(field="answer", matcher="contains")],
        store_dir=tmp_path,
        repeats=2,
        label="caps-eval",
    )

    assert board.axis == "config"
    assert set(board.results) == {"right", "wrong"}
    assert len(board.results["right"].tasks) == 2  # repeats threaded through to each variant
    assert board.results["right"].label == "caps-eval"  # label threaded through to each variant's run

    lb = board.leaderboard("final_answer_matches")
    assert [row.variant for row in lb] == ["right", "wrong"]
    assert lb[0].score == 1.0
    assert board.best("final_answer_matches") == "right"
    assert "ranked by final_answer_matches" in board.summary("final_answer_matches")


@pytest.mark.asyncio()
async def test_from_targets_accepts_factories_and_instances(tmp_path) -> None:
    variants = Variants.from_targets({
        "factory": lambda: Agent("a", config=TestConfig("hi from factory")),
        "instance": Agent("b", config=TestConfig("hi from instance")),  # reused as-is
    })

    board = await run_variants(
        [{"task_id": "t1", "inputs": {"input": "say hi"}}],
        variants=variants,
        scorers=[got_reply],
        store_dir=tmp_path,
    )

    assert board.axis == "target"
    assert set(board.results) == {"factory", "instance"}
    assert board.results["factory"].pass_rate("got_reply") == 1.0
    assert board.results["instance"].pass_rate("got_reply") == 1.0


@pytest.mark.asyncio()
async def test_errored_variant_does_not_abort_the_sweep(tmp_path) -> None:
    """A variant whose build raises is recorded (ranked last), not fatal to the others."""

    def boom(*, config=None) -> Agent:
        raise RuntimeError("bad build")

    variants = Variants.from_targets({"ok": lambda: Agent("ok", config=TestConfig("hi")), "broken": boom})

    board = await run_variants(
        [{"task_id": "t1", "inputs": {"input": "hi"}}],
        variants=variants,
        scorers=[got_reply],
        store_dir=tmp_path,
    )

    assert set(board.results) == {"ok", "broken"}
    assert board.best("got_reply") == "ok"
    assert board.results["broken"].aggregates.errors == 1


@pytest.mark.asyncio()
async def test_tie_shares_a_rank_and_has_no_unique_best(tmp_path) -> None:
    """Variants tied on the top score share rank 1, and ``best()`` returns None."""
    variants = Variants.from_configs(_build, {"a": TestConfig("Tokyo"), "b": TestConfig("Tokyo")})

    board = await run_variants(
        [{"task_id": "t1", "inputs": {"input": "capital?"}, "reference_outputs": {"answer": "Tokyo"}}],
        variants=variants,
        scorers=[final_answer_matches(field="answer", matcher="contains")],
        store_dir=tmp_path,
    )

    assert {row.rank for row in board.leaderboard("final_answer_matches")} == {1}  # both rank 1
    assert board.best("final_answer_matches") is None  # no unique winner
