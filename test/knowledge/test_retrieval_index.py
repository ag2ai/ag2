# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the memory index: in-memory behavior, store-backed persistence,
header verification, and corruption handling."""

import hashlib
import json

import pytest

from ag2.knowledge import MemoryKnowledgeStore
from ag2.knowledge.retrieval import HashingEmbedder, InMemoryIndex, IndexEntry, StoreBackedIndex

EMBEDDER = HashingEmbedder(dimension=8)


def _entry(entry_id: str, *, path: str = "/memory/facts/a.json", vector: list[float] | None = None) -> IndexEntry:
    return IndexEntry(
        id=entry_id,
        path=path,
        chunk=0,
        text_hash=f"hash-{entry_id}",
        vector=vector if vector is not None else [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        metadata={"text": f"text of {entry_id}"},
        updated_at=1.0,
    )


@pytest.mark.asyncio
class TestInMemoryIndex:
    async def test_upsert_and_entries_round_trip(self) -> None:
        index = InMemoryIndex()

        await index.upsert([_entry("e1"), _entry("e2", path="/memory/facts/b.json")])

        entries = await index.entries()
        assert {entry.id for entry in entries} == {"e1", "e2"}
        assert index.size == 2

    async def test_upsert_replaces_by_id(self) -> None:
        index = InMemoryIndex()

        await index.upsert([_entry("e1")])
        await index.upsert([_entry("e1", vector=[0.0] * 8)])

        [entry] = await index.entries()
        assert entry.vector == [0.0] * 8
        assert index.size == 1

    async def test_delete_by_id_and_by_path(self) -> None:
        index = InMemoryIndex()
        await index.upsert([
            _entry("e1"),
            _entry("e2", path="/memory/facts/b.json"),
            _entry("e3", path="/memory/facts/b.json"),
        ])

        await index.delete(ids=["e1"], paths=["/memory/facts/b.json"])

        assert await index.entries() == []

    async def test_delete_missing_ids_are_ignored(self) -> None:
        index = InMemoryIndex()
        await index.upsert([_entry("e1")])

        await index.delete(ids=["ghost"], paths=["/nope/"])

        assert index.size == 1

    async def test_query_returns_top_k_by_similarity(self) -> None:
        index = InMemoryIndex()
        await index.upsert([
            _entry("x", vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _entry("y", vector=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _entry("z", vector=[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ])

        scored = await index.query([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], k=2)

        assert [s.entry.id for s in scored] == ["x", "z"]

    async def test_query_applies_min_score_and_filter(self) -> None:
        index = InMemoryIndex()
        await index.upsert([
            _entry("low", vector=[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _entry("high", vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _entry("filtered", vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], path="/log/other.json"),
        ])

        scored = await index.query(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            k=5,
            min_score=0.9,
            filter=lambda entry: not entry.path.startswith("/log/"),
        )

        assert [s.entry.id for s in scored] == ["high"]

    async def test_query_with_k_zero_returns_empty(self) -> None:
        index = InMemoryIndex()
        await index.upsert([_entry("e1")])

        assert await index.query([1.0] * 8, k=0) == []


@pytest.mark.asyncio
class TestStoreBackedIndexPersistence:
    async def test_round_trip_through_the_store(self) -> None:
        store = MemoryKnowledgeStore()
        index = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        await index.upsert([_entry("e1"), _entry("e2")])

        reloaded = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        entries = await reloaded.entries()

        assert {entry.id for entry in entries} == {"e1", "e2"}
        assert {entry.text_hash for entry in entries} == {"hash-e1", "hash-e2"}
        assert reloaded.size == 2
        assert reloaded.needs_rebuild is False

    async def test_delete_persists(self) -> None:
        store = MemoryKnowledgeStore()
        index = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        await index.upsert([_entry("e1"), _entry("e2")])

        await index.delete(ids=["e1"])
        reloaded = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)

        assert [entry.id for entry in await reloaded.entries()] == ["e2"]

    async def test_sidecar_lives_at_the_index_prefix(self) -> None:
        store = MemoryKnowledgeStore()
        index = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        await index.upsert([_entry("e1")])

        assert await store.exists("/memory/.index/index.jsonl")

    async def test_missing_file_is_empty_not_corrupt(self) -> None:
        store = MemoryKnowledgeStore()
        index = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)

        entries = await index.entries()

        assert entries == []
        assert index.needs_rebuild is False

    async def test_corrupt_file_reports_needs_rebuild_and_serves_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = MemoryKnowledgeStore()
        index = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        await index.upsert([_entry("e1")])
        await store.write("/memory/.index/index.jsonl", "this is not json\nat all")

        reloaded = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        entries = await reloaded.entries()

        assert entries == []
        assert reloaded.size == 0
        assert reloaded.needs_rebuild is True
        assert any("needs_rebuild" in record.message for record in caplog.records)

    async def test_header_that_is_not_an_object_reports_needs_rebuild(self) -> None:
        store = MemoryKnowledgeStore()
        index = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        await index.upsert([_entry("e1")])
        await store.write("/memory/.index/index.jsonl", "[1, 2, 3]\nnot entries")

        reloaded = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)

        assert await reloaded.entries() == []
        assert reloaded.needs_rebuild is True

    async def test_checksum_tampering_reports_needs_rebuild(self) -> None:
        store = MemoryKnowledgeStore()
        index = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        await index.upsert([_entry("e1")])
        content = await store.read("/memory/.index/index.jsonl")
        header, _, body = content.partition("\n")
        header_json = json.loads(header)
        header_json["count"] = 99  # tamper without recomputing the checksum
        await store.write("/memory/.index/index.jsonl", f"{json.dumps(header_json)}\n{body}")

        reloaded = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        entries = await reloaded.entries()

        assert entries == []
        assert reloaded.needs_rebuild is True

    async def test_embedder_mismatch_reports_needs_rebuild(self) -> None:
        store = MemoryKnowledgeStore()
        index = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        await index.upsert([_entry("e1")])

        reloaded = StoreBackedIndex(store, embedder_name="other-embedder", dimension=EMBEDDER.dimension)
        entries = await reloaded.entries()

        assert entries == []
        assert reloaded.needs_rebuild is True

    async def test_dimension_mismatch_reports_needs_rebuild(self) -> None:
        store = MemoryKnowledgeStore()
        index = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        await index.upsert([_entry("e1")])

        reloaded = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension * 2)

        assert await reloaded.entries() == []
        assert reloaded.needs_rebuild is True

    async def test_rebuild_overwrites_the_rejected_file(self) -> None:
        store = MemoryKnowledgeStore()
        index = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        await index.upsert([_entry("e1")])
        await store.write("/memory/.index/index.jsonl", "garbage")

        reloaded = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        await reloaded.refresh()
        await reloaded.upsert([_entry("fresh")])

        recheck = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        assert [entry.id for entry in await recheck.entries()] == ["fresh"]
        assert recheck.needs_rebuild is False

    async def test_query_after_reload_ranks_loaded_entries(self) -> None:
        store = MemoryKnowledgeStore()
        index = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        await index.upsert([_entry("match", vector=[1.0] * 8), _entry("other", vector=[-1.0] + [0.0] * 7)])

        reloaded = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        scored = await reloaded.query([1.0] * 8, k=1)

        assert [s.entry.id for s in scored] == ["match"]

    async def test_meta_hash_round_trips_through_the_store(self) -> None:
        store = MemoryKnowledgeStore()
        index = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        entry = IndexEntry(
            id="e1",
            path="/memory/facts/a.json",
            chunk=0,
            text_hash="hash-e1",
            vector=[1.0] + [0.0] * 7,
            metadata={"text": "text of e1"},
            updated_at=1.0,
            meta_hash="mh-e1",
        )
        await index.upsert([entry, _entry("e2")])

        reloaded = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        entries = {loaded.id: loaded for loaded in await reloaded.entries()}

        assert entries["e1"].meta_hash == "mh-e1"

    async def test_version_1_file_without_meta_hash_reports_needs_rebuild(self) -> None:
        """An index written before ``meta_hash`` existed must be rebuilt, not served.

        Its entries carry no metadata hash, so the indexer could not tell a
        status change from an unchanged entry — the version bump forces a
        rebuild instead of silently keeping stale metadata.
        """
        store = MemoryKnowledgeStore()
        body = json.dumps({
            "id": "e1",
            "path": "/memory/facts/a.json",
            "chunk": 0,
            "text_hash": "hash-e1",
            "vector": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "metadata": {"text": "text of e1"},
            "updated_at": 1.0,
        })
        checksum = f"sha1:{hashlib.sha1(body.encode('utf-8')).hexdigest()}"
        header = json.dumps({
            "version": 1,
            "embedder": EMBEDDER.name,
            "dimension": EMBEDDER.dimension,
            "count": 1,
            "checksum": checksum,
        })
        await store.write("/memory/.index/index.jsonl", f"{header}\n{body}")

        reloaded = StoreBackedIndex(store, embedder_name=EMBEDDER.name, dimension=EMBEDDER.dimension)
        entries = await reloaded.entries()

        assert entries == []
        assert reloaded.needs_rebuild is True
