# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for RetrievalMemoryPolicy and the MemoryRetriever shaping it relies on
(MMR, path dedup), including the no-store and failure no-ops."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ag2 import Context
from ag2.events import BaseEvent, MemoryInjected, ModelRequest, TextInput
from ag2.knowledge import (
    CONVERSATIONS_PREFIX,
    FACTS_PREFIX,
    KnowledgeStore,
    MemoryKnowledgeStore,
    MemoryRetriever,
)
from ag2.knowledge.facts import Fact, FactStore
from ag2.knowledge.retrieval import HashingEmbedder, InMemoryIndex, IndexEntry, KnowledgeIndexer, RetrievedMemory
from ag2.policies import RetrievalMemoryPolicy
from ag2.stream import MemoryStream

EMBEDDER = HashingEmbedder(dimension=64)


def _user_request(text: str, *, origin: str = "user") -> ModelRequest:
    return ModelRequest([TextInput(text)], origin=origin)


def _context(store: KnowledgeStore | None = None) -> Context:
    context = Context(stream=MemoryStream())
    if store is not None:
        context.dependencies[KnowledgeStore] = store
    return context


class _ExplodingRetriever:
    """A retriever double whose retrieve always raises — for the no-op-on-error contract."""

    async def retrieve(self, query: str, **kwargs: Any) -> list:
        raise OSError("index exploded")


class _ScriptedRetriever:
    """A retriever double returning predetermined results, recording the query."""

    def __init__(self, results: list[RetrievedMemory]) -> None:
        self._results = results
        self.queries: list[str] = []

    async def retrieve(self, query: str, *, k: int = 5, **kwargs: Any) -> list[RetrievedMemory]:
        self.queries.append(query)
        return list(self._results)


def _memory(path: str, score: float, text: str) -> RetrievedMemory:
    return RetrievedMemory(path=path, chunk=0, text=text, score=score, metadata={})


async def _store_with_facts(*texts: str, status: str = "active") -> MemoryKnowledgeStore:
    store = MemoryKnowledgeStore()
    fact_store = FactStore(store)
    for i, text in enumerate(texts):
        await fact_store.put(Fact(text=text, subject=f"s{i}", kind="fact", status=status))
    return store


async def _indexed_retriever(store: MemoryKnowledgeStore) -> MemoryRetriever:
    index = InMemoryIndex()
    indexer = KnowledgeIndexer(store, index, EMBEDDER)
    await indexer.sync()
    return MemoryRetriever(index, EMBEDDER, indexer)


@pytest.mark.asyncio
class TestPolicyInjection:
    async def test_injects_retrieved_facts_with_attributions(self) -> None:
        store = await _store_with_facts("user likes tea", "deploy day is friday")
        retriever = await _indexed_retriever(store)
        policy = RetrievalMemoryPolicy(retriever)
        events = [_user_request("what does the user like to drink?")]

        prompts, returned = await policy.apply([], events, _context(store))

        assert len(prompts) == 2
        block = prompts[0]
        assert block.startswith("## Retrieved Memory")
        assert "user likes tea" in block
        assert "(source: /memory/facts/" in block
        assert "score " in block
        assert returned == events  # the history is passed through unchanged

    async def test_query_is_the_latest_user_input(self) -> None:
        store = await _store_with_facts("user likes tea")
        retriever = _ScriptedRetriever([])
        policy = RetrievalMemoryPolicy(retriever)  # type: ignore[arg-type]

        await policy.apply(
            [], [_user_request("what does the user like?"), _user_request("ping", origin="service")], _context(store)
        )

        assert retriever.queries == ["what does the user like?"]

    async def test_episodes_and_facts_split_their_quotas_by_score(self) -> None:
        retriever = _ScriptedRetriever([
            _memory(f"{FACTS_PREFIX}b.json", 0.9, "best fact"),
            _memory(f"{CONVERSATIONS_PREFIX}one.md", 0.8, "best episode"),
            _memory(f"{FACTS_PREFIX}a.json", 0.7, "second fact"),
            _memory(f"{CONVERSATIONS_PREFIX}two.md", 0.6, "second episode"),
        ])
        policy = RetrievalMemoryPolicy(retriever, k_episodes=1, k_facts=1)  # type: ignore[arg-type]

        prompts, _ = await policy.apply([], [_user_request("tell me about tea")], _context(MemoryKnowledgeStore()))

        assert prompts[1] == "[retrieval_memory] Injected 2 memories (1 facts, 1 episodes)."
        lines = [line for line in prompts[0].splitlines() if line.startswith("- ")]
        assert "best fact" in lines[0] and "best episode" in lines[1]
        assert "second fact" not in prompts[0] and "second episode" not in prompts[0]

    async def test_transparent_event_is_published_with_items_and_query(self) -> None:
        retriever = _ScriptedRetriever([_memory(f"{FACTS_PREFIX}abc.json", 0.82, "user likes tea")])
        policy = RetrievalMemoryPolicy(retriever)  # type: ignore[arg-type]
        context = _context(MemoryKnowledgeStore())
        published: list[BaseEvent] = []
        context.stream.where(MemoryInjected).subscribe(lambda event: published.append(event))

        await policy.apply([], [_user_request("what does the user like to drink?")], context)

        [event] = published
        assert event.query == "what does the user like to drink?"
        [item] = event.items
        assert item.kind == "fact"
        assert item.path == f"{FACTS_PREFIX}abc.json"
        assert item.score == pytest.approx(0.82)

    async def test_non_transparent_has_no_attributions_or_event(self) -> None:
        store = await _store_with_facts("user likes tea")
        retriever = await _indexed_retriever(store)
        policy = RetrievalMemoryPolicy(retriever, transparent=False)
        context = _context(store)
        published: list[BaseEvent] = []
        context.stream.where(MemoryInjected).subscribe(lambda event: published.append(event))

        prompts, _ = await policy.apply([], [_user_request("what does the user like to drink?")], context)

        assert len(prompts) == 1
        assert "(source:" not in prompts[0]
        assert published == []

    async def test_superseded_and_forgotten_facts_are_never_injected(self) -> None:
        store = await _store_with_facts("user likes tea", status="superseded")
        retriever = await _indexed_retriever(store)
        policy = RetrievalMemoryPolicy(retriever)

        prompts, _ = await policy.apply([], [_user_request("what does the user like to drink?")], _context(store))

        assert prompts == []

    async def test_stale_index_metadata_is_not_trusted_for_facts(self) -> None:
        """The index says active, the file says superseded — the file wins.

        The index metadata can lag the store (a status change between syncs),
        so the policy re-reads every selected fact's file before injecting it.
        """
        store = await _store_with_facts("user likes tea")
        retriever = await _indexed_retriever(store)
        fact_store = FactStore(store)
        [fact] = await fact_store.active()
        fact.status = "superseded"
        await fact_store.put(fact)  # the file moved on; the index was not re-synced
        policy = RetrievalMemoryPolicy(retriever)

        prompts, _ = await policy.apply([], [_user_request("what does the user like to drink?")], _context(store))

        assert prompts == []

    async def test_expired_fact_file_is_not_injected_even_when_the_index_says_active(self) -> None:
        store = MemoryKnowledgeStore()
        await FactStore(store).put(
            Fact(text="user likes tea", subject="s0", kind="fact", expires_at=datetime.now(UTC) - timedelta(hours=1))
        )
        retriever = await _indexed_retriever(store)
        policy = RetrievalMemoryPolicy(retriever)

        prompts, _ = await policy.apply([], [_user_request("what does the user like to drink?")], _context(store))

        assert prompts == []

    async def test_active_unexpired_facts_survive_the_verification(self) -> None:
        store = MemoryKnowledgeStore()
        await FactStore(store).put(
            Fact(text="user likes tea", subject="s0", kind="fact", expires_at=datetime.now(UTC) + timedelta(days=1))
        )
        retriever = await _indexed_retriever(store)
        policy = RetrievalMemoryPolicy(retriever)

        prompts, _ = await policy.apply([], [_user_request("what does the user like to drink?")], _context(store))

        assert "user likes tea" in prompts[0]

    async def test_both_sides_of_a_crashed_supersede_are_never_injected(self) -> None:
        """A supersede that died between its two writes leaves both facts active.

        The policy must not present the agent with both sides of the
        contradiction: of two active facts on one subject, only the newest
        ``updated_at`` is injected.
        """
        store = MemoryKnowledgeStore()
        fact_store = FactStore(store)
        await fact_store.put(
            Fact(
                text="user likes tea",
                subject="drink",
                kind="preference",
                updated_at=datetime.now(UTC) - timedelta(hours=1),
            )
        )
        await fact_store.put(Fact(text="user prefers coffee", subject="drink", kind="preference"))
        retriever = await _indexed_retriever(store)
        policy = RetrievalMemoryPolicy(retriever)

        prompts, _ = await policy.apply([], [_user_request("what does the user like to drink?")], _context(store))

        assert "user prefers coffee" in prompts[0]
        assert "user likes tea" not in prompts[0]

    async def test_include_superseded_puts_them_back(self) -> None:
        store = await _store_with_facts("user likes tea", status="superseded")
        retriever = await _indexed_retriever(store)
        policy = RetrievalMemoryPolicy(retriever, include_superseded=True)

        prompts, _ = await policy.apply([], [_user_request("what does the user like to drink?")], _context(store))

        assert "user likes tea" in prompts[0]

    async def test_empty_index_is_a_noop(self) -> None:
        store = MemoryKnowledgeStore()
        retriever = await _indexed_retriever(store)
        policy = RetrievalMemoryPolicy(retriever)
        events = [_user_request("anything")]

        prompts, returned = await policy.apply([], events, _context(store))

        assert prompts == []
        assert returned == events


@pytest.mark.asyncio
class TestPolicyNoOps:
    async def test_no_store_dependency_is_a_noop(self) -> None:
        store = await _store_with_facts("user likes tea")
        retriever = await _indexed_retriever(store)
        policy = RetrievalMemoryPolicy(retriever)
        events = [_user_request("anything")]

        prompts, returned = await policy.apply([], events, _context())

        assert prompts == []
        assert returned == events

    async def test_no_user_request_is_a_noop(self) -> None:
        store = await _store_with_facts("user likes tea")
        retriever = await _indexed_retriever(store)
        policy = RetrievalMemoryPolicy(retriever)

        prompts, returned = await policy.apply([], [], _context(store))

        assert prompts == []
        assert returned == []

    async def test_service_only_window_is_a_noop(self) -> None:
        store = await _store_with_facts("user likes tea")
        retriever = await _indexed_retriever(store)
        policy = RetrievalMemoryPolicy(retriever)
        events = [_user_request("status update", origin="service")]

        prompts, returned = await policy.apply([], events, _context(store))

        assert prompts == []
        assert returned == events

    async def test_retriever_failure_returns_inputs_unchanged(self) -> None:
        store = await _store_with_facts("user likes tea")
        policy = RetrievalMemoryPolicy(_ExplodingRetriever())  # type: ignore[arg-type]
        events = [_user_request("anything")]

        prompts, returned = await policy.apply(["kept"], events, _context(store))

        assert prompts == ["kept"]
        assert returned == events

    async def test_retriever_failure_is_logged_not_raised(self, caplog: pytest.LogCaptureFixture) -> None:
        store = await _store_with_facts("user likes tea")
        policy = RetrievalMemoryPolicy(_ExplodingRetriever())  # type: ignore[arg-type]

        await policy.apply([], [_user_request("anything")], _context(store))

        assert any("RetrievalMemoryPolicy failed" in record.message for record in caplog.records)


@pytest.mark.asyncio
class TestPolicyConstruction:
    async def test_needs_a_retriever_or_embedder_plus_factory(self) -> None:
        with pytest.raises(ValueError, match="retriever"):
            RetrievalMemoryPolicy()

    async def test_rejects_both_shapes_at_once(self) -> None:
        store = await _store_with_facts("user likes tea")
        retriever = await _indexed_retriever(store)

        with pytest.raises(ValueError, match="either"):
            RetrievalMemoryPolicy(retriever, embedder=EMBEDDER, index_factory=lambda _store: InMemoryIndex())

    async def test_embedder_and_index_factory_shape_builds_one_index_per_store(self) -> None:
        store = await _store_with_facts("user likes tea")
        calls: list[KnowledgeStore] = []

        def factory(received: KnowledgeStore) -> InMemoryIndex:
            calls.append(received)
            return InMemoryIndex()

        policy = RetrievalMemoryPolicy(embedder=EMBEDDER, index_factory=factory)
        context = _context(store)

        await policy.apply([], [_user_request("first")], context)
        await policy.apply([], [_user_request("second")], context)

        assert len(calls) == 1  # the index is cached per store, not rebuilt per call

    async def test_sync_on_apply_runs_the_indexer(self) -> None:
        store = MemoryKnowledgeStore()
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, EMBEDDER)
        retriever = MemoryRetriever(index, EMBEDDER, indexer)
        policy = RetrievalMemoryPolicy(retriever, sync_on_apply=True)  # type: ignore[arg-type]
        await FactStore(store).put(Fact(text="user likes tea", subject="user", kind="fact"))

        prompts, _ = await policy.apply([], [_user_request("what does the user like to drink?")], _context(store))

        assert "user likes tea" in prompts[0]  # searchable within the same apply


@pytest.mark.asyncio
class TestTokenBudget:
    async def test_truncates_by_score_order(self) -> None:
        retriever = _ScriptedRetriever([
            _memory(f"{FACTS_PREFIX}one.json", 0.9, "one"),
            _memory(f"{FACTS_PREFIX}two.json", 0.8, "two"),
            _memory(f"{FACTS_PREFIX}three.json", 0.7, "three"),
        ])
        policy = RetrievalMemoryPolicy(retriever, token_budget=30)  # type: ignore[arg-type]

        prompts, _ = await policy.apply([], [_user_request("query")], _context(MemoryKnowledgeStore()))

        lines = [line for line in prompts[0].splitlines() if line.startswith("- ")]
        assert [line.split()[1] for line in lines] == ["one", "two"]  # best first, third did not fit

    async def test_best_item_is_kept_even_over_budget(self) -> None:
        retriever = _ScriptedRetriever([
            _memory(f"{FACTS_PREFIX}one.json", 0.9, "one"),
            _memory(f"{FACTS_PREFIX}two.json", 0.8, "two"),
        ])
        policy = RetrievalMemoryPolicy(retriever, token_budget=0)  # type: ignore[arg-type]

        prompts, _ = await policy.apply([], [_user_request("query")], _context(MemoryKnowledgeStore()))

        lines = [line for line in prompts[0].splitlines() if line.startswith("- ")]
        assert [line.split()[1] for line in lines] == ["one"]


class _TableEmbedder:
    """Embedder double mapping every text to one fixed vector."""

    def __init__(self, query: list[float], dimension: int) -> None:
        self._query = query
        self.dimension = dimension
        self.name = "table"

    async def embed(self, texts: Any) -> list[list[float]]:
        return [list(self._query) for _ in texts]


@pytest.mark.asyncio
class TestMemoryRetriever:
    def _entry(self, entry_id: str, path: str, vector: list[float], chunk: int = 0) -> IndexEntry:
        return IndexEntry(
            id=entry_id,
            path=path,
            chunk=chunk,
            text_hash=f"h-{entry_id}",
            vector=vector,
            metadata={},
            updated_at=1.0,
        )

    def _table_embedder(self, query: list[float]) -> _TableEmbedder:
        return _TableEmbedder(query, dimension=EMBEDDER.dimension)

    async def test_dedups_by_path_best_chunk_first(self) -> None:
        index = InMemoryIndex()
        await index.upsert([
            self._entry("p1c0", "/a.md", [1.0] + [0.0] * 63),
            self._entry("p1c1", "/a.md", [0.9, 0.1] + [0.0] * 62, chunk=1),
            self._entry("p2c0", "/b.md", [0.7, 0.7] + [0.0] * 62),
        ])
        retriever = MemoryRetriever(index, self._table_embedder([1.0] + [0.0] * 63))

        single = await retriever.retrieve("query", k=3)
        multi = await retriever.retrieve("query", k=3, per_path=2)

        assert [(r.path, r.chunk) for r in single] == [("/a.md", 0), ("/b.md", 0)]
        assert [(r.path, r.chunk) for r in multi] == [("/a.md", 0), ("/a.md", 1), ("/b.md", 0)]
        assert single[0].text == ""  # metadata carries no text here; empty, not a crash

    async def test_mmr_diversity_promotes_the_distinct_result(self) -> None:
        index = InMemoryIndex()
        await index.upsert([
            self._entry("a", "/a.md", [1.0] + [0.0] * 63),
            self._entry("b", "/b.md", [1.0] + [0.0] * 63),
            self._entry("c", "/c.md", [0.99, 0.1] + [0.0] * 62),
        ])
        retriever = MemoryRetriever(index, self._table_embedder([1.0] + [0.0] * 63))

        plain = await retriever.retrieve("query", k=3)
        diverse = await retriever.retrieve("query", k=3, diversity=0.8)

        assert [r.path for r in plain] == ["/a.md", "/b.md", "/c.md"]
        assert [r.path for r in diverse][:2] == ["/a.md", "/c.md"]  # the twin is demoted behind the distinct hit

    async def test_scores_in_results_stay_raw_similarities(self) -> None:
        index = InMemoryIndex()
        await index.upsert([
            self._entry("a", "/a.md", [1.0] + [0.0] * 63),
            self._entry("b", "/b.md", [0.0, 1.0] + [0.0] * 62),
        ])
        retriever = MemoryRetriever(index, self._table_embedder([1.0] + [0.0] * 63))

        plain = await retriever.retrieve("query", k=2)
        [diverse] = await retriever.retrieve("query", k=1, diversity=0.9)

        assert diverse.score == plain[0].score  # MMR re-orders; the reported score stays the raw similarity
        assert plain[1].score == pytest.approx(0.0)

    async def test_rejects_out_of_range_diversity(self) -> None:
        store = await _store_with_facts("user likes tea")
        retriever = await _indexed_retriever(store)

        with pytest.raises(ValueError, match="diversity"):
            await retriever.retrieve("query", diversity=1.5)

    async def test_blank_query_retrieves_nothing(self) -> None:
        store = await _store_with_facts("user likes tea")
        retriever = await _indexed_retriever(store)

        assert await retriever.retrieve("   ") == []
