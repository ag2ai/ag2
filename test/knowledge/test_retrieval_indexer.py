# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for KnowledgeIndexer: incremental sync, pruning, error tolerance,
limits, and the chunking it applies."""

from datetime import UTC, datetime, timedelta

import pytest

from ag2.knowledge import Fact, FactStore, MemoryKnowledgeStore
from ag2.knowledge.retrieval import (
    HashingEmbedder,
    InMemoryIndex,
    IndexEntry,
    KnowledgeIndexer,
    StoreBackedIndex,
)

EMBEDDER = HashingEmbedder(dimension=32)


class CountingEmbedder:
    """Embedder wrapper that counts embed calls — for the incremental-sync assertions."""

    def __init__(self, inner: HashingEmbedder) -> None:
        self._inner = inner
        self.calls: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def name(self) -> str:
        return self._inner.name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return await self._inner.embed(texts)


def _injectable(entry: IndexEntry) -> bool:
    """The policy's own injectability rule, restated: no dead fact status."""
    return entry.metadata.get("status") not in ("superseded", "forgotten")


def _embedded_texts(embedder: CountingEmbedder) -> list[str]:
    return [text for batch in embedder.calls for text in batch]


async def _indexed_fact(fact: Fact) -> tuple[FactStore, Fact, InMemoryIndex, KnowledgeIndexer, CountingEmbedder]:
    """A store holding one fact, indexed once by a counting embedder."""
    store = MemoryKnowledgeStore()
    facts = FactStore(store)
    await facts.put(fact)
    embedder = CountingEmbedder(EMBEDDER)
    index = InMemoryIndex()
    indexer = KnowledgeIndexer(store, index, embedder)
    await indexer.sync()
    return facts, fact, index, indexer, embedder


@pytest.mark.asyncio
class TestSync:
    async def test_indexes_files_under_the_roots(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/memory/conversations/one.md", "first conversation summary")
        await store.write("/memory/facts/f1.json", '{"id": "f1", "text": "user likes tea", "status": "active"}')
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, EMBEDDER)

        report = await indexer.sync()

        assert (report.added, report.updated, report.removed, report.unchanged) == (2, 0, 0, 0)
        assert report.errors == {}
        assert index.size == 2

    async def test_unchanged_content_is_not_re_embedded(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/memory/conversations/one.md", "stable summary")
        embedder = CountingEmbedder(EMBEDDER)
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, embedder)

        await indexer.sync()
        assert len(embedder.calls) == 1

        report = await indexer.sync()

        assert (report.added, report.updated, report.unchanged) == (0, 0, 1)
        assert len(embedder.calls) == 1  # no second embed call

    async def test_changed_content_is_re_embedded(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/memory/conversations/one.md", "before the edit")
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, EMBEDDER)
        await indexer.sync()

        await store.write("/memory/conversations/one.md", "after the edit")
        report = await indexer.sync()

        assert (report.added, report.updated, report.unchanged) == (0, 1, 0)
        [entry] = await index.entries()
        assert "after the edit" in entry.metadata["text"]

    async def test_deleted_file_is_removed_from_the_index(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/memory/conversations/gone.md", "about to vanish")
        await store.write("/memory/conversations/stays.md", "still here")
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, EMBEDDER)
        await indexer.sync()

        await store.delete("/memory/conversations/gone.md")
        report = await indexer.sync()

        assert report.removed == 1
        paths = {entry.path for entry in await index.entries()}
        assert paths == {"/memory/conversations/stays.md"}

    async def test_shrunken_file_drops_its_vanished_chunks(self) -> None:
        store = MemoryKnowledgeStore()
        long_text = "\n\n".join(f"paragraph {i}: {('content ' * 30)}" for i in range(10))
        await store.write("/memory/conversations/one.md", long_text)
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, EMBEDDER, max_bytes_per_file=100_000)
        await indexer.sync()
        chunks_before = index.size
        assert chunks_before > 1

        await store.write("/memory/conversations/one.md", "short now")
        report = await indexer.sync()

        # Every old chunk's id changed (ids hash the text), so all are removed;
        # the surviving chunk counts as updated, not added.
        assert (report.removed, report.updated, report.added) == (chunks_before, 1, 0)
        assert index.size == 1

    async def test_unreadable_file_is_reported_not_fatal(self) -> None:
        class FailingReadStore(MemoryKnowledgeStore):
            async def read(self, path: str) -> str | None:
                if path == "/memory/conversations/broken.md":
                    raise OSError("disk on fire")
                return await super().read(path)

        store = FailingReadStore()
        await store.write("/memory/conversations/broken.md", "x")
        await store.write("/memory/conversations/fine.md", "readable")
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, EMBEDDER)

        report = await indexer.sync()

        assert "broken.md" in next(iter(report.errors))
        assert "/memory/conversations/fine.md" in {entry.path for entry in await index.entries()}

    async def test_entries_of_an_unreadable_file_survive_the_pass(self) -> None:
        class FlakyStore(MemoryKnowledgeStore):
            def __init__(self) -> None:
                super().__init__()
                self.fail_reads = False

            async def read(self, path: str) -> str | None:
                if self.fail_reads and path == "/memory/conversations/one.md":
                    raise OSError("transient")
                return await super().read(path)

        store = FlakyStore()
        await store.write("/memory/conversations/one.md", "stable content")
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, EMBEDDER)
        await indexer.sync()

        store.fail_reads = True
        report = await indexer.sync()

        assert report.errors and index.size == 1  # not pruned on a transient failure

    async def test_oversized_file_is_skipped_and_reported(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/memory/conversations/big.md", "x" * 500)
        await store.write("/memory/conversations/small.md", "tiny")
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, EMBEDDER, max_bytes_per_file=100)

        report = await indexer.sync()

        assert any("big.md" in path for path in report.skipped)
        assert [entry.path for entry in await index.entries()] == ["/memory/conversations/small.md"]

    async def test_max_files_bounds_the_walk(self) -> None:
        store = MemoryKnowledgeStore()
        for i in range(4):
            await store.write(f"/memory/conversations/c{i}.md", f"summary {i}")
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, EMBEDDER, max_files=2)

        report = await indexer.sync()

        assert index.size == 2
        assert len(report.skipped) == 2

    async def test_exclude_keeps_the_index_sidecar_out(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/memory/conversations/one.md", "hello")
        index = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        indexer = KnowledgeIndexer(store, index, EMBEDDER)

        await indexer.sync()

        # Only the conversation is indexed — never the sidecar itself.
        assert {entry.path for entry in await index.entries()} == {"/memory/conversations/one.md"}

    async def test_include_predicate_filters_files(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/memory/conversations/keep.md", "keep me")
        await store.write("/memory/conversations/skip.md", "skip me")
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, EMBEDDER, include=lambda path: path.endswith("keep.md"))

        await indexer.sync()

        assert [entry.path for entry in await index.entries()] == ["/memory/conversations/keep.md"]

    async def test_custom_roots_are_walked(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/elsewhere/note.md", "outside the default roots")
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, EMBEDDER, roots=("/elsewhere/",))

        await indexer.sync()

        assert [entry.path for entry in await index.entries()] == ["/elsewhere/note.md"]

    async def test_entries_outside_the_roots_are_not_pruned(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/memory/conversations/one.md", "in scope")
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, EMBEDDER, roots=("/memory/conversations/",))
        foreign = IndexEntry(
            id="foreign", path="/user/notes.md", chunk=0, text_hash="h", vector=[0.0] * 32, metadata={}, updated_at=1.0
        )
        await index.upsert([foreign])

        report = await indexer.sync()

        assert report.removed == 0  # the indexer only prunes what it owns
        assert foreign.id in {entry.id for entry in await index.entries()}

    async def test_fact_json_is_indexed_by_its_text_field_with_status_metadata(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write(
            "/memory/facts/f1.json",
            '{"id": "f1", "text": "user likes tea", "subject": "user", "kind": "preference", "status": "active"}',
        )
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, EMBEDDER, roots=("/memory/facts/",))

        await indexer.sync()

        [entry] = await index.entries()
        assert entry.metadata["text"] == "user likes tea"
        assert entry.metadata["status"] == "active"
        assert entry.metadata["subject"] == "user"
        assert entry.metadata["kind"] == "preference"

    async def test_non_fact_json_is_indexed_as_raw_content(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/memory/facts/broken.json", "not json at all")
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, EMBEDDER, roots=("/memory/facts/",))

        await indexer.sync()

        [entry] = await index.entries()
        assert entry.metadata["text"] == "not json at all"

    async def test_long_markdown_is_chunked(self) -> None:
        store = MemoryKnowledgeStore()
        paragraphs = "\n\n".join(f"paragraph {i} with some content to chunk" for i in range(30))
        await store.write("/memory/conversations/one.md", paragraphs)
        index = InMemoryIndex()
        indexer = KnowledgeIndexer(store, index, EMBEDDER)

        report = await indexer.sync()

        assert report.added > 1
        chunks = sorted(entry.chunk for entry in await index.entries())
        assert chunks == list(range(len(chunks)))

    async def test_store_backed_index_survives_a_reopen_between_syncs(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/memory/conversations/one.md", "persist me")
        embedder = CountingEmbedder(EMBEDDER)
        first = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        await KnowledgeIndexer(store, first, embedder).sync()

        second = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        report = await KnowledgeIndexer(store, second, embedder).sync()

        assert report.unchanged == 1  # the persisted index, not a fresh one, was diffed
        assert len(embedder.calls) == 1


@pytest.mark.asyncio
class TestMetadataOnlySync:
    """A fact whose text is unchanged but whose lifecycle moved must be caught.

    ``text_hash`` covers only the fact's text, so a status change used to leave
    the stale metadata in the index and the dead fact kept being injected;
    ``meta_hash`` closes that hole without spending an embedding on it.
    """

    async def test_supersede_updates_the_index_metadata_without_re_embedding(self) -> None:
        facts, old, index, indexer, embedder = await _indexed_fact(
            Fact(text="user likes tea", subject="drink", kind="preference")
        )
        replacement = Fact(text="user prefers coffee", subject="drink", kind="preference")
        await facts.supersede(old.id, replacement)

        report = await indexer.sync()

        assert report.updated_metadata == 1
        # The old fact's text was embedded once, never again — only the new
        # fact's (genuinely new) text was embedded.
        assert _embedded_texts(embedder).count("user likes tea") == 1
        assert _embedded_texts(embedder)[-1] == "user prefers coffee"
        [entry] = [entry for entry in await index.entries() if entry.path == facts.path_for(old.id)]
        assert entry.metadata["status"] == "superseded"
        assert entry.metadata["superseded_by"] == replacement.id

    async def test_a_superseded_fact_is_no_longer_injectable(self) -> None:
        facts, old, index, indexer, embedder = await _indexed_fact(
            Fact(text="user likes tea", subject="drink", kind="preference")
        )
        replacement = await facts.supersede(
            old.id, Fact(text="user prefers coffee", subject="drink", kind="preference")
        )
        await indexer.sync()
        vector = (await EMBEDDER.embed(["user likes tea"]))[0]

        scored = await index.query(vector, k=5, filter=_injectable)

        # the dead fact is filtered out; the replacement that superseded it is not
        assert [s.entry.path for s in scored] == [facts.path_for(replacement.id)]

    async def test_forget_updates_the_index_metadata_without_re_embedding(self) -> None:
        facts, fact, index, indexer, embedder = await _indexed_fact(
            Fact(text="user likes tea", subject="drink", kind="preference")
        )

        await facts.forget(fact.id, "user asked to be forgotten")
        report = await indexer.sync()

        assert report.updated_metadata == 1
        assert len(embedder.calls) == 1  # no second embed call — text unchanged
        [entry] = await index.entries()
        assert entry.metadata["status"] == "forgotten"
        vector = (await EMBEDDER.embed(["user likes tea"]))[0]
        assert await index.query(vector, k=5, filter=_injectable) == []

    async def test_expire_updates_the_index_metadata_without_re_embedding(self) -> None:
        facts, fact, index, indexer, embedder = await _indexed_fact(
            Fact(
                text="user likes tea",
                subject="drink",
                kind="preference",
                expires_at=datetime.now(UTC) - timedelta(hours=1),
            )
        )

        await facts.expire(datetime.now(UTC))
        report = await indexer.sync()

        assert report.updated_metadata == 1
        assert len(embedder.calls) == 1  # no second embed call — text unchanged
        [entry] = await index.entries()
        assert entry.metadata["status"] == "forgotten"
        vector = (await EMBEDDER.embed(["user likes tea"]))[0]
        assert await index.query(vector, k=5, filter=_injectable) == []

    async def test_unchanged_metadata_is_not_counted_as_an_update(self) -> None:
        facts, fact, index, indexer, embedder = await _indexed_fact(
            Fact(text="user likes tea", subject="drink", kind="preference")
        )

        report = await indexer.sync()

        assert (report.unchanged, report.updated_metadata) == (1, 0)
        assert len(embedder.calls) == 1
