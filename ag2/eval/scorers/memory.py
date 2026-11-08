# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Memory-quality scorers — did retrieval surface the right facts, and only those?

The graded signal is the :class:`~ag2.events.MemoryInjected` event that
:class:`~ag2.policies.RetrievalMemoryPolicy` publishes in transparent mode: one
event per assembly that injected anything, carrying every injected item with
its store path and retrieval score. The scorers parse the fact items out of
those events and compare them against the task's ``expected`` /
``forbidden_fact_ids`` metadata — which facts *should* have been recalled, and
which stale ones must not resurface.

Three prebuilts:

* :func:`memory_recall` — the fraction of expected facts that were injected
  (optionally limited to the first ``k`` injections, i.e. recall@k).
* :func:`memory_precision` — the fraction of injected facts that were expected.
* :func:`stale_fact_injection` — the fraction of injected facts that are
  forbidden (stale); ``0.0`` is the ideal.

The same expectations drive :class:`MemoryEvalCase`, :func:`build_memory_suite`
and :func:`evaluate_retriever`, which measure a retriever directly — no agent,
no LLM — and report recall@k, MRR, precision, and the stale-injection rate.

A note on traces: :class:`MemoryInjected` is a transient stream event, so it
appears in a Trace whenever the trace carries the run's stream events. A
trace rebuilt from spans alone does not include it; such traces grade as "no
memory injected".
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Protocol, runtime_checkable

from ag2.events import MemoryInjected
from ag2.knowledge import FACTS_PREFIX
from ag2.knowledge.retrieval import RetrievedMemory

from .._types import Feedback
from ..dataset import Suite, Task
from ..scorer import Scorer
from ..trace import Trace

__all__ = (
    "MemoryEvalCase",
    "MemoryEvalReport",
    "Retriever",
    "build_memory_suite",
    "evaluate_retriever",
    "memory_precision",
    "memory_recall",
    "stale_fact_injection",
)


@runtime_checkable
class Retriever(Protocol):
    """What :func:`evaluate_retriever` needs of a retriever — the
    :class:`~ag2.knowledge.retrieval.MemoryRetriever` signature's core."""

    async def retrieve(self, query: str, *, k: int = 5) -> Sequence[RetrievedMemory]:
        """Ranked results for the query."""
        ...


def _fact_id(path: str) -> str:
    """The fact id a ``.../<id>.json`` store path holds."""
    return PurePosixPath(path).stem


def _metadata_ids(task: Task, key: str) -> list[str]:
    value = task.metadata.get(key, [])
    return [str(item) for item in value] if isinstance(value, list) else []


def _injected_fact_ids(trace: Trace, k: int | None) -> list[str]:
    """Fact ids the trace's injections carried, first-seen order, limited to ``k``."""
    ids: list[str] = []
    for event in trace.events:
        if isinstance(event, MemoryInjected):
            for item in event.items:
                if item.kind == "fact" and item.path.startswith(FACTS_PREFIX):
                    fact_id = _fact_id(item.path)
                    if fact_id not in ids:
                        ids.append(fact_id)
    return ids[:k] if k is not None else ids


def _detail(expected: list[str], forbidden: list[str], injected: list[str], k: int | None) -> dict[str, Any]:
    return {"expected": expected, "forbidden": forbidden, "injected": injected, "k": k}


def memory_recall(key: str = "memory_recall", *, k: int | None = None) -> Scorer:
    """Fraction of the task's expected facts that were injected.

    Args:
        key: Feedback key.
        k: When set, only the first ``k`` injected facts count — recall@k.

    A task with no expected facts scores ``1.0``: nothing was required of the
    memory layer. Detail carries the expected, forbidden, and injected id lists.
    """

    async def _recall(trace: Trace, task: Task) -> Feedback:
        expected = _metadata_ids(task, "expected_fact_ids")
        injected = _injected_fact_ids(trace, k)
        if not expected:
            return Feedback(key=key, score=1.0, comment="no expected facts", detail=_detail(expected, [], injected, k))
        hits = [fact_id for fact_id in expected if fact_id in injected]
        comment = f"{len(hits)}/{len(expected)} expected facts injected"
        if misses := [fact_id for fact_id in expected if fact_id not in injected]:
            comment = f"{comment}; missed {', '.join(misses)}"
        return Feedback(
            key=key,
            score=len(hits) / len(expected),
            comment=comment,
            detail=_detail(expected, [], injected, k),
        )

    return Scorer(_recall, key=key)


def memory_precision(key: str = "memory_precision", *, k: int | None = None) -> Scorer:
    """Fraction of the injected facts that the task expected.

    Args:
        key: Feedback key.
        k: When set, only the first ``k`` injected facts count — precision@k.

    Nothing injected scores ``0.0`` — for a memory policy that is a failure to
    inject, not a vacuous pass.
    """

    async def _precision(trace: Trace, task: Task) -> Feedback:
        expected = _metadata_ids(task, "expected_fact_ids")
        injected = _injected_fact_ids(trace, k)
        if not injected:
            return Feedback(key=key, score=0.0, comment="no memory injected", detail=_detail(expected, [], injected, k))
        hits = [fact_id for fact_id in injected if fact_id in expected]
        return Feedback(
            key=key,
            score=len(hits) / len(injected),
            comment=f"{len(hits)}/{len(injected)} injected facts expected",
            detail=_detail(expected, [], injected, k),
        )

    return Scorer(_precision, key=key)


def stale_fact_injection(key: str = "stale_fact_injection") -> Scorer:
    """Fraction of injected facts that the task forbids — the stale-injection rate.

    ``0.0`` is the ideal. Nothing injected also scores ``0.0``: no stale fact
    can resurface if none was injected. Detail lists which forbidden ids came
    back.
    """

    async def _stale(trace: Trace, task: Task) -> Feedback:
        forbidden = _metadata_ids(task, "forbidden_fact_ids")
        injected = _injected_fact_ids(trace, None)
        if not injected:
            return Feedback(
                key=key, score=0.0, comment="no memory injected", detail=_detail([], forbidden, injected, None)
            )
        stale = [fact_id for fact_id in injected if fact_id in forbidden]
        comment = "no stale facts injected" if not stale else f"stale facts injected: {', '.join(stale)}"
        return Feedback(
            key=key,
            score=len(stale) / len(injected),
            comment=comment,
            detail=_detail([], forbidden, injected, None),
        )

    return Scorer(_stale, key=key)


@dataclass(frozen=True, slots=True)
class MemoryEvalCase:
    """One retrieval-evaluation case: a query and the facts that should — and
    must not — come back for it.

    Attributes:
        query: The text to retrieve for.
        expected: Fact ids the retriever should surface.
        forbidden: Fact ids (stale ones) the retriever must not surface.
        k: How many results to count for this case.
    """

    query: str
    expected: tuple[str, ...]
    forbidden: tuple[str, ...] = ()
    k: int = 5

    def __post_init__(self) -> None:
        if not self.expected:
            raise ValueError("MemoryEvalCase needs at least one expected fact id.")
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        object.__setattr__(self, "expected", tuple(self.expected))
        object.__setattr__(self, "forbidden", tuple(self.forbidden))


@dataclass(frozen=True, slots=True)
class MemoryCaseResult:
    """One case's retrieval outcome, as the report aggregates it."""

    query: str
    expected: tuple[str, ...]
    forbidden: tuple[str, ...]
    retrieved: tuple[str, ...]
    recall: float
    precision: float
    reciprocal_rank: float
    stale_rate: float


@dataclass(frozen=True, slots=True)
class MemoryEvalReport:
    """Aggregate retrieval quality over a set of :class:`MemoryEvalCase` items.

    All four rates are means over the cases. ``recall`` is recall@k as each
    case's own ``k`` defines it, ``mrr`` the mean reciprocal rank of the first
    expected fact, ``precision`` and ``stale_rate`` the per-case fractions.
    """

    recall: float
    mrr: float
    precision: float
    stale_rate: float
    cases: tuple[MemoryCaseResult, ...] = field(default_factory=tuple)


def build_memory_suite(cases: Sequence[MemoryEvalCase], *, name: str = "memory-retrieval") -> Suite:
    """Build a :class:`~ag2.eval.Suite` from memory cases.

    Each case becomes one task whose ``inputs["input"]`` is the query and whose
    metadata carries ``expected_fact_ids`` / ``forbidden_fact_ids`` — the shape
    the memory scorers read.
    """
    tasks = [
        Task(
            inputs={"input": case.query},
            task_id=f"memory-{index:04d}",
            metadata={
                "expected_fact_ids": list(case.expected),
                "forbidden_fact_ids": list(case.forbidden),
            },
        )
        for index, case in enumerate(cases)
    ]
    return Suite(tasks=tasks, name=name)


async def evaluate_retriever(retriever: Any, cases: Sequence[MemoryEvalCase]) -> MemoryEvalReport:
    """Measure a retriever against labelled cases — offline, no agent, no LLM.

    Runs each case's query through ``retriever.retrieve`` (a
    :class:`~ag2.knowledge.retrieval.MemoryRetriever` or anything with the same
    signature), keeps the fact items (paths under :data:`~ag2.knowledge.FACTS_PREFIX`),
    and scores recall, precision, reciprocal rank, and stale injection per case.

    Raises:
        ValueError: If ``cases`` is empty.
    """
    if not cases:
        raise ValueError("evaluate_retriever needs at least one case.")
    results = tuple([await _evaluate_case(retriever, case) for case in cases])
    count = len(results)
    return MemoryEvalReport(
        recall=sum(result.recall for result in results) / count,
        mrr=sum(result.reciprocal_rank for result in results) / count,
        precision=sum(result.precision for result in results) / count,
        stale_rate=sum(result.stale_rate for result in results) / count,
        cases=results,
    )


async def _evaluate_case(retriever: Any, case: MemoryEvalCase) -> MemoryCaseResult:
    retrieved = await retriever.retrieve(case.query, k=case.k)
    fact_ids = tuple(_fact_id(item.path) for item in retrieved if getattr(item, "path", "").startswith(FACTS_PREFIX))
    expected_set = set(case.expected)
    forbidden_set = set(case.forbidden)

    hit_ranks = [position for position, fact_id in enumerate(fact_ids) if fact_id in expected_set]
    stale_positions = [position for position, fact_id in enumerate(fact_ids) if fact_id in forbidden_set]

    return MemoryCaseResult(
        query=case.query,
        expected=case.expected,
        forbidden=case.forbidden,
        retrieved=fact_ids,
        recall=len(hit_ranks) / len(case.expected),
        precision=len(hit_ranks) / len(fact_ids) if fact_ids else 0.0,
        reciprocal_rank=1.0 / (hit_ranks[0] + 1) if hit_ranks else 0.0,
        stale_rate=len(stale_positions) / len(fact_ids) if fact_ids else 0.0,
    )
