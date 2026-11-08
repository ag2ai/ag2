# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Public-API tests for the memory scorers and the offline retriever evaluation."""

from typing import Any

import pytest

from ag2.eval import Feedback, Scorer, Suite, Task, Trace
from ag2.eval.scorers import (
    MemoryEvalCase,
    build_memory_suite,
    evaluate_retriever,
    memory_precision,
    memory_recall,
    stale_fact_injection,
)
from ag2.events import MemoryInjected, MemoryItem
from ag2.knowledge.retrieval import RetrievedMemory


def _trace(*injections: MemoryInjected) -> Trace:
    return Trace(events=list(injections), exception=None, duration_ms=0)


def _injection(*paths: str, query: str = "q") -> MemoryInjected:
    return MemoryInjected(
        items=[
            MemoryItem(path=path, score=0.9, kind="fact" if "/memory/facts/" in path else "episode") for path in paths
        ],
        query=query,
    )


def _task(expected: list[str], forbidden: list[str] | None = None) -> Task:
    return Task(
        task_id="t1",
        inputs={"input": "q"},
        metadata={"expected_fact_ids": expected, "forbidden_fact_ids": forbidden or []},
    )


async def _run(scorer: Scorer, *, trace: Trace, task: Task) -> Feedback:
    [feedback] = await scorer(
        inputs={"input": "q"},
        outputs={},
        reference_outputs=None,
        trace=trace,
        task=task,
    )
    return feedback


@pytest.mark.asyncio
class TestMemoryRecall:
    async def test_full_recall(self) -> None:
        trace = _trace(_injection("/memory/facts/a.json", "/memory/facts/b.json"))

        feedback = await _run(memory_recall(), trace=trace, task=_task(["a", "b"]))

        assert (feedback.key, feedback.score) == ("memory_recall", 1.0)
        assert feedback.comment == "2/2 expected facts injected"
        assert feedback.detail["injected"] == ["a", "b"]

    async def test_partial_recall_counts_misses(self) -> None:
        trace = _trace(_injection("/memory/facts/a.json"))

        feedback = await _run(memory_recall(), trace=trace, task=_task(["a", "b", "c"]))

        assert feedback.score == pytest.approx(1 / 3)
        assert "b" in feedback.comment and "c" in feedback.comment

    async def test_recall_k_limits_counted_injections(self) -> None:
        trace = _trace(_injection("/memory/facts/a.json", "/memory/facts/b.json"))

        feedback = await _run(memory_recall(k=1), trace=trace, task=_task(["b"]))

        assert feedback.score == 0.0  # b was injected, but only after the k=1 cut
        assert feedback.detail["injected"] == ["a"]

    async def test_deduplicates_injections_across_model_calls(self) -> None:
        trace = _trace(_injection("/memory/facts/a.json"), _injection("/memory/facts/a.json", "/memory/facts/b.json"))

        feedback = await _run(memory_recall(), trace=trace, task=_task(["a", "b"]))

        assert feedback.score == 1.0
        assert feedback.detail["injected"] == ["a", "b"]  # first-seen order, no repeats

    async def test_episodes_do_not_count_as_facts(self) -> None:
        trace = _trace(_injection("/memory/conversations/one.md"))

        feedback = await _run(memory_recall(), trace=trace, task=_task(["a"]))

        assert feedback.score == 0.0
        assert feedback.detail["injected"] == []

    async def test_no_expected_facts_scores_one(self) -> None:
        feedback = await _run(memory_recall(), trace=_trace(), task=_task([]))

        assert feedback.score == 1.0
        assert feedback.comment == "no expected facts"


@pytest.mark.asyncio
class TestMemoryPrecision:
    async def test_all_injected_were_expected(self) -> None:
        trace = _trace(_injection("/memory/facts/a.json", "/memory/facts/b.json"))

        feedback = await _run(memory_precision(), trace=trace, task=_task(["a", "b"]))

        assert feedback.score == 1.0

    async def test_unexpected_injections_lower_precision(self) -> None:
        trace = _trace(_injection("/memory/facts/a.json", "/memory/facts/x.json"))

        feedback = await _run(memory_precision(), trace=trace, task=_task(["a"]))

        assert feedback.score == pytest.approx(0.5)

    async def test_nothing_injected_scores_zero(self) -> None:
        feedback = await _run(memory_precision(), trace=_trace(), task=_task(["a"]))

        assert feedback.score == 0.0
        assert feedback.comment == "no memory injected"

    async def test_precision_k(self) -> None:
        trace = _trace(_injection("/memory/facts/a.json", "/memory/facts/x.json"))

        feedback = await _run(memory_precision(k=1), trace=trace, task=_task(["a"]))

        assert feedback.score == 1.0


@pytest.mark.asyncio
class TestStaleFactInjection:
    async def test_no_stale_injection_scores_zero(self) -> None:
        trace = _trace(_injection("/memory/facts/a.json"))

        feedback = await _run(stale_fact_injection(), trace=trace, task=_task(["a"], forbidden=["stale"]))

        assert feedback.score == 0.0
        assert feedback.comment == "no stale facts injected"

    async def test_stale_injection_is_measured(self) -> None:
        trace = _trace(_injection("/memory/facts/a.json", "/memory/facts/stale.json"))

        feedback = await _run(stale_fact_injection(), trace=trace, task=_task(["a"], forbidden=["stale"]))

        assert feedback.score == pytest.approx(0.5)
        assert feedback.detail["injected"] == ["a", "stale"]

    async def test_nothing_injected_scores_zero(self) -> None:
        feedback = await _run(stale_fact_injection(), trace=_trace(), task=_task(["a"], forbidden=["stale"]))

        assert feedback.score == 0.0
        assert feedback.comment == "no memory injected"


class TestMemoryEvalCase:
    def test_requires_expected_facts(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            MemoryEvalCase(query="q", expected=())

    def test_requires_positive_k(self) -> None:
        with pytest.raises(ValueError, match="k"):
            MemoryEvalCase(query="q", expected=("a",), k=0)

    def test_sequences_are_coerced_to_tuples(self) -> None:
        case = MemoryEvalCase(query="q", expected=["a"], forbidden=["b"])  # type: ignore[arg-type]

        assert case.expected == ("a",)
        assert case.forbidden == ("b",)


class TestBuildMemorySuite:
    def test_cases_become_tasks_with_memory_metadata(self) -> None:
        suite = build_memory_suite([
            MemoryEvalCase(query="what tea?", expected=("a",), forbidden=("stale",)),
            MemoryEvalCase(query="what coffee?", expected=("b",)),
        ])

        assert isinstance(suite, Suite)
        assert len(suite) == 2
        first, second = suite.tasks
        assert first.inputs == {"input": "what tea?"}
        assert first.metadata == {"expected_fact_ids": ["a"], "forbidden_fact_ids": ["stale"]}
        assert second.metadata == {"expected_fact_ids": ["b"], "forbidden_fact_ids": []}


class _ScriptedRetriever:
    """Retriever double: maps each query to predetermined RetrievedMemory paths."""

    def __init__(self, by_query: dict[str, list[str]]) -> None:
        self._by_query = by_query

    async def retrieve(self, query: str, *, k: int = 5, **kwargs: Any) -> list[RetrievedMemory]:
        return [
            RetrievedMemory(path=path, chunk=0, text="t", score=1.0 - i / 10, metadata={})
            for i, path in enumerate(self._by_query.get(query, [])[:k])
        ]


@pytest.mark.asyncio
class TestEvaluateRetriever:
    async def test_perfect_retrieval(self) -> None:
        retriever = _ScriptedRetriever({
            "tea?": ["/memory/facts/a.json", "/memory/facts/b.json"],
        })

        report = await evaluate_retriever(
            retriever, [MemoryEvalCase(query="tea?", expected=("a", "b"), forbidden=("stale",))]
        )

        assert (report.recall, report.mrr, report.precision, report.stale_rate) == (1.0, 1.0, 1.0, 0.0)

    async def test_mrr_counts_the_first_expected_rank(self) -> None:
        retriever = _ScriptedRetriever({"q": ["/memory/facts/x.json", "/memory/facts/a.json", "/memory/facts/b.json"]})

        report = await evaluate_retriever(retriever, [MemoryEvalCase(query="q", expected=("a", "b"))])

        assert report.mrr == pytest.approx(0.5)  # first expected fact at rank 2
        assert report.recall == 1.0
        assert report.precision == pytest.approx(2 / 3)

    async def test_stale_rate_and_per_case_results(self) -> None:
        retriever = _ScriptedRetriever({"q": ["/memory/facts/stale.json", "/memory/facts/a.json"]})

        report = await evaluate_retriever(retriever, [MemoryEvalCase(query="q", expected=("a",), forbidden=("stale",))])

        assert report.stale_rate == pytest.approx(0.5)
        [case] = report.cases
        assert case.retrieved == ("stale", "a")
        assert case.reciprocal_rank == pytest.approx(0.5)  # the expected fact sits at rank 2

    async def test_non_fact_items_are_ignored(self) -> None:
        retriever = _ScriptedRetriever({"q": ["/memory/conversations/one.md", "/memory/facts/a.json"]})

        report = await evaluate_retriever(retriever, [MemoryEvalCase(query="q", expected=("a",))])

        assert (report.recall, report.precision, report.stale_rate) == (1.0, 1.0, 0.0)
        assert report.cases[0].retrieved == ("a",)

    async def test_k_bounds_each_case(self) -> None:
        retriever = _ScriptedRetriever({"q": ["/memory/facts/a.json", "/memory/facts/b.json"]})

        report = await evaluate_retriever(retriever, [MemoryEvalCase(query="q", expected=("a", "b"), k=1)])

        assert report.recall == pytest.approx(0.5)  # only the top-1 counted
        assert report.cases[0].retrieved == ("a",)

    async def test_nothing_retrieved_scores_zero(self) -> None:
        retriever = _ScriptedRetriever({"q": []})

        report = await evaluate_retriever(retriever, [MemoryEvalCase(query="q", expected=("a",))])

        assert (report.recall, report.mrr, report.precision, report.stale_rate) == (0.0, 0.0, 0.0, 0.0)

    async def test_averages_across_cases(self) -> None:
        retriever = _ScriptedRetriever({
            "hit": ["/memory/facts/a.json"],
            "miss": ["/memory/facts/x.json"],
        })

        report = await evaluate_retriever(
            retriever, [MemoryEvalCase(query="hit", expected=("a",)), MemoryEvalCase(query="miss", expected=("a",))]
        )

        assert report.recall == pytest.approx(0.5)
        assert report.mrr == pytest.approx(0.5)

    async def test_requires_cases(self) -> None:
        with pytest.raises(ValueError, match="case"):
            await evaluate_retriever(_ScriptedRetriever({}), [])
