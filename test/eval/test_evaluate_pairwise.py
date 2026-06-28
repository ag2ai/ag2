# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluate_pairwise + PairwiseRunResult (win-rate, Wilson CI, kappa).

Scripted comparators keep the aggregation math deterministic and independent of
any LLM; the dual-order swap is unit-tested separately in test_pairwise_judge.
"""

import pytest

from ag2.eval import InMemoryTraceSource, TraceRef, evaluate_pairwise
from ag2.eval.pairwise import PairwiseComparator, PairwiseOutcome
from ag2.eval.trace import Trace
from ag2.events import ModelMessage, ModelResponse

_TASKS = ("t1", "t2", "t3", "t4")


def _trace(answer: str) -> Trace:
    return Trace(events=[ModelResponse(message=ModelMessage(answer))], exception=None, duration_ms=0)


def _source(label: str) -> InMemoryTraceSource:
    return InMemoryTraceSource([(TraceRef(f"{label}-{t}", task_id=t), _trace(label)) for t in _TASKS])


class _Scripted:
    """A PairwiseComparator that returns a fixed winner per task_id."""

    def __init__(self, key: str, wins: dict[str, str]) -> None:
        self.key = key
        self._wins = wins

    async def compare(self, *, task, trace_a, trace_b, reference_outputs) -> PairwiseOutcome:
        return PairwiseOutcome(winner=self._wins.get(task.task_id, "tie"))


@pytest.mark.asyncio()
async def test_win_rate_tally_and_persist(tmp_path) -> None:
    judge = _Scripted("correctness", {"t1": "b", "t2": "b", "t3": "a", "t4": "tie"})
    assert isinstance(judge, PairwiseComparator)

    result = await evaluate_pairwise(
        _source("v1"), _source("v2"), comparators=[judge], variant_a="v1", variant_b="v2", store_dir=tmp_path
    )

    assert result.tally("correctness") == (1, 2, 1)  # (a_wins, b_wins, ties)
    wr = result.win_rate("correctness")
    assert (wr.wins, wr.losses, wr.ties, wr.n) == (2, 1, 1, 4)
    assert wr.rate == (2 + 0.5) / 4  # ties count 0.5 -> 0.625
    assert 0.0 <= wr.ci[0] <= wr.ci[1] <= 1.0
    assert (tmp_path / f"{result.run_id}.json").exists()


@pytest.mark.asyncio()
async def test_agreement_cohen_kappa(tmp_path) -> None:
    judge = _Scripted("c@judge", {"t1": "b", "t2": "b", "t3": "a", "t4": "tie"})
    human = _Scripted("c@human", {"t1": "b", "t2": "tie", "t3": "a", "t4": "tie"})

    result = await evaluate_pairwise(_source("v1"), _source("v2"), comparators=[judge, human], store_dir=tmp_path)

    ag = result.agreement("c@judge", "c@human")
    assert ag.n == 4
    assert ag.rate == 3 / 4  # agree on t1, t3, t4; differ on t2
    assert ag.disagreements == (("t2", "b", "tie"),)
    assert -1.0 <= ag.cohen_kappa <= 1.0


@pytest.mark.asyncio()
async def test_no_pairs_without_task_ids(tmp_path) -> None:
    a = InMemoryTraceSource([(TraceRef("a1"), _trace("v1"))])  # no task_id
    b = InMemoryTraceSource([(TraceRef("b1"), _trace("v2"))])

    result = await evaluate_pairwise(a, b, comparators=[_Scripted("k", {})], store_dir=tmp_path)

    assert result.tally("k") == (0, 0, 0)
