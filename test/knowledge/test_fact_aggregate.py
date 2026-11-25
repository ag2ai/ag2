# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fact aggregate: extraction (static and model), decision
application, reports, matching through the shared index, and the
ModelReconciler fallback."""

import asyncio

import pytest

from ag2 import Context
from ag2.config import ModelProvider
from ag2.events import ModelRequest, TextInput
from ag2.knowledge import (
    CandidateMatch,
    ExtractedFact,
    Fact,
    FactDecision,
    FactExtractionAggregate,
    FactStore,
    HeuristicReconciler,
    MemoryKnowledgeStore,
    ModelReconciler,
    ScoredFact,
    StoreBackedIndex,
)
from ag2.knowledge.retrieval import HashingEmbedder, InMemoryIndex, KnowledgeIndexer
from ag2.stream import MemoryStream
from ag2.testing import TestConfig

EMBEDDER = HashingEmbedder(dimension=64)


def _events(*texts: str) -> list[ModelRequest]:
    return [ModelRequest([TextInput(text)]) for text in texts]


def _context() -> Context:
    return Context(stream=MemoryStream())


def _candidate(text: str, *, subject: str | None = "user", kind: str = "preference") -> ExtractedFact:
    return ExtractedFact(text=text, subject=subject, kind=kind)


class _StaticExtractor:
    """StaticFactExtractor double returning canned candidates and counting calls."""

    def __init__(self, candidates: list[ExtractedFact]) -> None:
        self._candidates = candidates
        self.calls = 0

    async def extract(self, events, store) -> list[ExtractedFact]:
        self.calls += 1
        return list(self._candidates)


class _ScriptedReconciler:
    """FactReconciler double handing back predetermined decisions in order."""

    def __init__(self, decisions: list[FactDecision]) -> None:
        self._decisions = list(decisions)
        self.candidates_seen: list[CandidateMatch] = []

    async def reconcile(self, candidates, context) -> list[FactDecision]:
        self.candidates_seen.extend(candidates)
        return [self._decisions[i] for i in range(len(candidates))]


class _YieldingHeuristicReconciler:
    """HeuristicReconciler behind one loop yield — the shape of a model-backed reconciler.

    The bare heuristic decides synchronously (no suspension point), so without
    this yield two concurrent aggregate runs cannot interleave at all and the
    double-add race the lock guards against never manifests in a test.
    """

    def __init__(self) -> None:
        self._inner = HeuristicReconciler()

    async def reconcile(self, candidates, context) -> list[FactDecision]:
        await asyncio.sleep(0)
        return await self._inner.reconcile(candidates, context)


def _match(candidate: ExtractedFact, *scored: ScoredFact) -> CandidateMatch:
    return CandidateMatch(candidate, tuple(scored))


async def _stored(fact_store: FactStore, fact: Fact) -> Fact:
    await fact_store.put(fact)
    return fact


@pytest.mark.asyncio
class TestHeuristicReconciler:
    async def test_duplicate_above_dup_threshold_is_noop(self) -> None:
        existing = Fact(text="user likes tea", subject="user", kind="preference")
        reconciler = HeuristicReconciler(dup_threshold=0.9)

        [decision] = await reconciler.reconcile(
            [_match(_candidate("user likes tea"), ScoredFact(existing, 0.95))], _context()
        )

        assert decision.action == "noop"
        assert decision.existing is existing

    async def test_same_subject_refinement_updates(self) -> None:
        existing = Fact(text="user likes tea", subject="user", kind="preference")
        reconciler = HeuristicReconciler(dup_threshold=0.99, update_threshold=0.5)

        [decision] = await reconciler.reconcile(
            [_match(_candidate("user likes green tea"), ScoredFact(existing, 0.7))], _context()
        )

        assert decision.action == "update"
        assert decision.existing is existing

    async def test_same_subject_low_similarity_supersedes(self) -> None:
        existing = Fact(text="user likes tea", subject="user", kind="preference")
        reconciler = HeuristicReconciler(dup_threshold=0.99, update_threshold=0.99)

        [decision] = await reconciler.reconcile(
            [_match(_candidate("user hates tea now"), ScoredFact(existing, 0.3))], _context()
        )

        assert decision.action == "supersede"
        assert decision.existing is existing

    async def test_different_subject_adds(self) -> None:
        existing = Fact(text="user likes tea", subject="user", kind="preference")
        reconciler = HeuristicReconciler()

        [decision] = await reconciler.reconcile(
            [_match(_candidate("deploy day is friday", subject="deploys"), ScoredFact(existing, 0.5))], _context()
        )

        assert decision.action == "add"
        assert decision.existing is None

    async def test_no_matches_adds(self) -> None:
        reconciler = HeuristicReconciler()

        [decision] = await reconciler.reconcile([_match(_candidate("user likes tea"))], _context())

        assert decision.action == "add"


@pytest.mark.asyncio
class TestModelReconciler:
    async def test_structured_decisions_are_applied(self) -> None:
        existing = Fact(text="user likes tea", subject="user", kind="preference")
        reconciler = ModelReconciler(
            TestConfig(
                '{"decisions": [{"index": 0, "action": "supersede", "existing_id": "'
                + existing.id
                + '", "reason": "contradicts"}]}',
                provider=ModelProvider.OPENAI,
                model="test",
            )
        )

        [decision] = await reconciler.reconcile(
            [_match(_candidate("user prefers coffee"), ScoredFact(existing, 0.4))], _context()
        )

        assert decision.action == "supersede"
        assert decision.existing is existing
        assert decision.reason == "contradicts"
        assert reconciler.failures == []

    async def test_bad_model_output_falls_back_and_records_the_failure(self) -> None:
        reconciler = ModelReconciler(TestConfig("not json at all", provider=ModelProvider.OPENAI, model="test"))

        [decision] = await reconciler.reconcile([_match(_candidate("user likes tea"))], _context())

        assert decision.action == "add"  # the heuristic fallback decided; the candidate was not lost
        assert len(reconciler.failures) == 1

    async def test_decision_naming_an_unmatched_fact_falls_back_per_candidate(self) -> None:
        existing = Fact(text="user likes tea", subject="user", kind="preference")
        reconciler = ModelReconciler(
            TestConfig(
                '{"decisions": [{"index": 0, "action": "noop", "existing_id": "no-such-fact", "reason": "x"}]}',
                provider=ModelProvider.OPENAI,
                model="test",
            )
        )

        [decision] = await reconciler.reconcile(
            [_match(_candidate("user prefers coffee"), ScoredFact(existing, 0.2))], _context()
        )

        assert decision.action == "supersede"  # heuristic decided: same subject, low similarity
        assert decision.existing is existing

    async def test_missing_index_in_the_plan_falls_back(self) -> None:
        reconciler = ModelReconciler(TestConfig('{"decisions": []}', provider=ModelProvider.OPENAI, model="test"))

        [decision] = await reconciler.reconcile([_match(_candidate("user likes tea"))], _context())

        assert decision.action == "add"

    async def test_decisions_cover_every_candidate_in_order(self) -> None:
        first = Fact(text="user likes tea", subject="user", kind="preference")
        second = Fact(text="deploy day is friday", subject="deploys", kind="fact")
        reconciler = ModelReconciler(
            TestConfig(
                '{"decisions": ['
                '{"index": 1, "action": "add"},'
                '{"index": 0, "action": "noop", "existing_id": "' + first.id + '"}'
                "]}",
                provider=ModelProvider.OPENAI,
                model="test",
            )
        )

        decisions = await reconciler.reconcile(
            [
                _match(_candidate("user likes tea"), ScoredFact(first, 0.99)),
                _match(_candidate("release tuesday"), ScoredFact(second, 0.1)),
            ],
            _context(),
        )

        assert [d.action for d in decisions] == ["noop", "add"]


@pytest.mark.asyncio
class TestFactExtractionAggregate:
    async def test_needs_a_config_or_an_extractor(self) -> None:
        with pytest.raises(ValueError, match="config"):
            FactExtractionAggregate()

    async def test_static_extractor_writes_facts_without_a_model(self) -> None:
        extractor = _StaticExtractor([_candidate("user likes tea")])
        aggregate = FactExtractionAggregate(extractor=extractor)
        store = MemoryKnowledgeStore()
        context = _context()

        await aggregate.aggregate(_events("i like tea"), context, store)

        fact_store = FactStore(store)
        [fact] = await fact_store.active()
        assert fact.text == "user likes tea"
        assert fact.source.stream_id == str(context.stream.id)
        assert aggregate.last_report.added == 1
        assert extractor.calls == 1

    async def test_empty_events_do_nothing(self) -> None:
        extractor = _StaticExtractor([_candidate("user likes tea")])
        aggregate = FactExtractionAggregate(extractor=extractor)

        await aggregate.aggregate([], _context(), MemoryKnowledgeStore())

        assert extractor.calls == 0

    async def test_max_facts_per_run_bounds_extraction(self) -> None:
        extractor = _StaticExtractor([_candidate(f"fact number {i}", subject=None, kind="fact") for i in range(5)])
        aggregate = FactExtractionAggregate(extractor=extractor, max_facts_per_run=2)
        store = MemoryKnowledgeStore()

        await aggregate.aggregate(_events("lots of facts"), _context(), store)

        assert aggregate.last_report.added == 2
        assert len(await FactStore(store).active()) == 2

    async def test_scripted_add_writes_a_new_fact(self) -> None:
        extractor = _StaticExtractor([_candidate("user likes tea")])
        reconciler = _ScriptedReconciler([FactDecision(_candidate("user likes tea"), "add")])
        aggregate = FactExtractionAggregate(extractor=extractor, reconciler=reconciler)
        store = MemoryKnowledgeStore()

        await aggregate.aggregate(_events("i like tea"), _context(), store)

        assert aggregate.last_report.added == 1
        assert len(await FactStore(store).active()) == 1

    async def test_scripted_update_rewrites_the_existing_fact(self) -> None:
        store = MemoryKnowledgeStore()
        fact_store = FactStore(store)
        existing = await _stored(fact_store, Fact(text="user likes tea", subject="user", kind="preference"))
        extractor = _StaticExtractor([_candidate("user likes green tea")])
        reconciler = _ScriptedReconciler([FactDecision(_candidate("user likes green tea"), "update", existing)])
        aggregate = FactExtractionAggregate(extractor=extractor, reconciler=reconciler)

        await aggregate.aggregate(_events("refined taste"), _context(), store)

        assert aggregate.last_report.updated == 1
        [fact] = await fact_store.active()
        assert fact.id == existing.id
        assert fact.text == "user likes green tea"

    async def test_scripted_supersede_links_and_retires_the_old_fact(self) -> None:
        store = MemoryKnowledgeStore()
        fact_store = FactStore(store)
        old = await _stored(fact_store, Fact(text="user likes tea", subject="user", kind="preference"))
        extractor = _StaticExtractor([_candidate("user hates tea now")])
        reconciler = _ScriptedReconciler([FactDecision(_candidate("user hates tea now"), "supersede", old)])
        aggregate = FactExtractionAggregate(extractor=extractor, reconciler=reconciler)

        await aggregate.aggregate(_events("changed my mind"), _context(), store)

        assert aggregate.last_report.superseded == 1
        active = await fact_store.active()
        assert len(active) == 1
        assert active[0].supersedes == [old.id]
        assert (await fact_store.get(old.id)).status == "superseded"

    async def test_scripted_noop_writes_nothing(self) -> None:
        store = MemoryKnowledgeStore()
        fact_store = FactStore(store)
        existing = await _stored(fact_store, Fact(text="user likes tea", subject="user", kind="preference"))
        extractor = _StaticExtractor([_candidate("user likes tea")])
        reconciler = _ScriptedReconciler([FactDecision(_candidate("user likes tea"), "noop", existing)])
        aggregate = FactExtractionAggregate(extractor=extractor, reconciler=reconciler)

        await aggregate.aggregate(_events("i like tea"), _context(), store)

        assert aggregate.last_report.noop == 1
        assert [fact.id for fact in await fact_store.active()] == [existing.id]

    async def test_failed_write_is_counted_not_fatal(self) -> None:
        class FailingWriteStore(MemoryKnowledgeStore):
            async def write(self, path: str, content: str) -> None:
                raise OSError("disk on fire")

        extractor = _StaticExtractor([_candidate("user likes tea")])
        aggregate = FactExtractionAggregate(extractor=extractor)

        await aggregate.aggregate(_events("i like tea"), _context(), FailingWriteStore())

        assert aggregate.last_report.failed == 1
        assert aggregate.last_report.added == 0

    async def test_matches_come_from_the_shared_index(self) -> None:
        store = MemoryKnowledgeStore()
        fact_store = FactStore(store)
        await _stored(fact_store, Fact(text="user likes tea", subject="user", kind="preference"))
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, EMBEDDER)
        await indexer.sync()

        extractor = _StaticExtractor([_candidate("user likes tea")])
        aggregate = FactExtractionAggregate(
            extractor=extractor, reconciler=HeuristicReconciler(dup_threshold=0.9), indexer=indexer
        )

        await aggregate.aggregate(_events("i like tea"), _context(), store)

        assert aggregate.last_report.noop == 1  # matched through the index, not by loading all facts
        assert len(await fact_store.active()) == 1

    async def test_indexer_syncs_after_writes(self) -> None:
        store = MemoryKnowledgeStore()
        index = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        indexer = KnowledgeIndexer(store, index, EMBEDDER)
        extractor = _StaticExtractor([_candidate("user likes tea")])
        aggregate = FactExtractionAggregate(extractor=extractor, indexer=indexer)

        await aggregate.aggregate(_events("i like tea"), _context(), store)

        [entry] = await index.entries()
        assert entry.metadata["text"] == "user likes tea"

    async def test_concurrent_runs_on_one_store_do_not_double_add(self) -> None:
        """Two concurrent aggregate runs must not both add the same candidate.

        Without per-store serialization both runs reconcile against the same
        pre-write snapshot — the index holds nothing yet — and both add their
        near-duplicate candidates.
        """
        store = MemoryKnowledgeStore()
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, EMBEDDER)
        extractor = _StaticExtractor([_candidate("user likes tea")])
        aggregate = FactExtractionAggregate(
            extractor=extractor, indexer=indexer, reconciler=_YieldingHeuristicReconciler()
        )

        await asyncio.gather(
            aggregate.aggregate(_events("i like tea"), _context(), store),
            aggregate.aggregate(_events("i like tea a lot"), _context(), store),
        )

        assert len(await FactStore(store).active()) == 1
        # one run added the fact; the other matched it through the synced index and noop'd
        assert (aggregate.last_report.added, aggregate.last_report.noop) in {(1, 0), (0, 1)}

    async def test_model_extraction_uses_a_structured_call(self) -> None:
        config = TestConfig(
            '{"facts": [{"text": "user likes tea", "subject": "user", "kind": "preference", "confidence": 0.9}]}',
            provider=ModelProvider.OPENAI,
            model="test",
        )
        aggregate = FactExtractionAggregate(config)
        store = MemoryKnowledgeStore()

        await aggregate.aggregate(_events("i like tea"), _context(), store)

        assert aggregate.last_report.added == 1
        [fact] = await FactStore(store).active()
        assert fact.text == "user likes tea"
        assert fact.confidence == 0.9

    async def test_extracted_expiry_is_parsed_onto_the_fact(self) -> None:
        extractor = _StaticExtractor([
            ExtractedFact(text="user is offline", subject="user", expires_at="2027-03-01T00:00:00+00:00")
        ])
        aggregate = FactExtractionAggregate(extractor=extractor)
        store = MemoryKnowledgeStore()

        await aggregate.aggregate(_events("brb"), _context(), store)

        [fact] = await FactStore(store).active()
        assert fact.expires_at is not None
        assert fact.expires_at.year == 2027
        assert fact.expires_at.tzinfo is not None

    async def test_unparseable_expiry_is_dropped_not_fatal(self) -> None:
        extractor = _StaticExtractor([ExtractedFact(text="user is offline", subject="user", expires_at="soon-ish")])
        aggregate = FactExtractionAggregate(extractor=extractor)
        store = MemoryKnowledgeStore()

        await aggregate.aggregate(_events("brb"), _context(), store)

        assert aggregate.last_report.added == 1
        [fact] = await FactStore(store).active()
        assert fact.expires_at is None

    async def test_model_extraction_failure_raises_into_aggregation(self) -> None:
        config = TestConfig(RuntimeError("boom"), provider=ModelProvider.OPENAI, model="test")
        aggregate = FactExtractionAggregate(config)
        store = MemoryKnowledgeStore()

        with pytest.raises(RuntimeError, match="boom"):
            await aggregate.aggregate(_events("i like tea"), _context(), store)

        assert await FactStore(store).active() == []
