# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Memory index — stores chunk vectors and ranks them against a query vector.

Two implementations of one :class:`MemoryIndex` protocol:

* :class:`InMemoryIndex` — a dict guarded by an ``asyncio.Lock``, exact top-k
  over ``heapq``. For tests and short-lived processes.
* :class:`StoreBackedIndex` — persists the whole index through a
  :class:`~ag2.knowledge.KnowledgeStore` as one JSONL sidecar, so every store
  backend (memory, disk, sqlite, redis) becomes a durable vector index with no
  new infrastructure (ADR 0036).

Entries carry their chunk text in ``metadata["text"]`` — retrieval returns
injectable text without a second store read — plus the two change signals the
indexer diffs against: a ``text_hash`` for the chunk text and a ``meta_hash``
for the metadata dict, so a fact whose status flipped is caught without
re-embedding its (unchanged) text.
"""

import asyncio
import hashlib
import heapq
import json
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from operator import attrgetter
from typing import Any, Protocol, runtime_checkable

from ag2.knowledge.base import KnowledgeStore

from ._vector import cosine_similarity

__all__ = (
    "InMemoryIndex",
    "IndexEntry",
    "MemoryIndex",
    "ScoredEntry",
    "StoreBackedIndex",
)

logger = logging.getLogger(__name__)

_INDEX_VERSION = 2


@dataclass(frozen=True, slots=True)
class IndexEntry:
    """One indexed chunk: where it came from, what it contains, its vector.

    Attributes:
        id: Stable chunk id (see :func:`ag2.knowledge.retrieval.chunking.chunk_id`).
        path: Store path of the file the chunk came from.
        chunk: Position of the chunk within that file.
        text_hash: SHA-1 of the chunk text — the content signal the indexer diffs.
        meta_hash: SHA-1 of the metadata dict's canonical JSON — the metadata
            signal the indexer diffs, separate from ``text_hash`` so a status
            flip is caught while the (unchanged) text keeps its vector.
        vector: Embedding of the chunk text, in the index's vector space.
        metadata: Payload carried with the chunk. ``"text"`` holds the chunk
            text itself; JSON files also surface their parsed scalar fields
            (e.g. a fact's ``status``) so queries can filter on them.
        updated_at: Unix timestamp of the last write of this entry.
    """

    id: str
    path: str
    chunk: int
    text_hash: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: float = 0.0
    meta_hash: str = ""


@dataclass(frozen=True, slots=True)
class ScoredEntry:
    """An index entry paired with its similarity to the query vector."""

    entry: IndexEntry
    score: float


@runtime_checkable
class MemoryIndex(Protocol):
    """Stores chunk vectors and returns the entries closest to a query vector."""

    @property
    def size(self) -> int:
        """Number of entries currently held."""
        ...

    async def upsert(self, entries: Iterable[IndexEntry]) -> None:
        """Insert or replace entries by id."""
        ...

    async def delete(self, ids: Sequence[str] = (), paths: Sequence[str] = ()) -> None:
        """Remove entries by id, by store path, or both. Missing ids are ignored."""
        ...

    async def query(
        self,
        vector: Sequence[float],
        *,
        k: int = 5,
        filter: Callable[[IndexEntry], bool] | None = None,
        min_score: float = 0.0,
    ) -> list[ScoredEntry]:
        """Return at most ``k`` entries by descending cosine similarity.

        Only entries passing ``filter`` are considered, and only those scoring
        at least ``min_score`` (inclusive) are returned.
        """
        ...

    async def entries(self) -> list[IndexEntry]:
        """Every stored entry, in unspecified order."""
        ...


class InMemoryIndex:
    """:class:`MemoryIndex` over a plain dict, exact top-k via ``heapq``.

    Mutations and queries serialize on one ``asyncio.Lock`` — sufficient for
    the single-event-loop concurrency ``ag2`` runs under.
    """

    def __init__(self) -> None:
        self._entries: dict[str, IndexEntry] = {}
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        return len(self._entries)

    async def upsert(self, entries: Iterable[IndexEntry]) -> None:
        async with self._lock:
            for entry in entries:
                self._entries[entry.id] = entry

    async def delete(self, ids: Sequence[str] = (), paths: Sequence[str] = ()) -> None:
        id_set = set(ids)
        path_set = set(paths)
        async with self._lock:
            doomed = [eid for eid, entry in self._entries.items() if eid in id_set or entry.path in path_set]
            for eid in doomed:
                del self._entries[eid]

    async def query(
        self,
        vector: Sequence[float],
        *,
        k: int = 5,
        filter: Callable[[IndexEntry], bool] | None = None,
        min_score: float = 0.0,
    ) -> list[ScoredEntry]:
        if k <= 0:
            return []
        async with self._lock:
            candidates = [entry for entry in self._entries.values() if filter is None or filter(entry)]
        scored = [ScoredEntry(entry, cosine_similarity(vector, entry.vector)) for entry in candidates]
        scored = [scored_entry for scored_entry in scored if scored_entry.score >= min_score]
        return heapq.nlargest(k, scored, key=attrgetter("score"))

    async def entries(self) -> list[IndexEntry]:
        async with self._lock:
            return list(self._entries.values())


def _entry_to_dict(entry: IndexEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "path": entry.path,
        "chunk": entry.chunk,
        "text_hash": entry.text_hash,
        "meta_hash": entry.meta_hash,
        "vector": list(entry.vector),
        "metadata": entry.metadata,
        "updated_at": entry.updated_at,
    }


def _entry_from_dict(data: dict[str, Any]) -> IndexEntry:
    return IndexEntry(
        id=str(data["id"]),
        path=str(data["path"]),
        chunk=int(data["chunk"]),
        text_hash=str(data["text_hash"]),
        vector=[float(v) for v in data["vector"]],
        metadata=dict(data.get("metadata") or {}),
        updated_at=float(data.get("updated_at") or 0.0),
        meta_hash=str(data.get("meta_hash") or ""),
    )


class StoreBackedIndex:
    """:class:`MemoryIndex` persisted through a :class:`~ag2.knowledge.KnowledgeStore`.

    The whole index lives in one JSONL sidecar file: a header line followed by
    one line per entry::

        {"version": 2, "embedder": "...", "dimension": 256, "count": 3, "checksum": "sha1:..."}
        {"id": "...", "path": "...", "chunk": 0, "text_hash": "...", "meta_hash": "...", "vector": [...], ...}

    Saving rewrites the file in full via ``store.write``. Among the built-in
    backends that write is atomic (``MemoryKnowledgeStore`` assigns under a
    lock; ``SqliteKnowledgeStore`` is one INSERT OR REPLACE under its lock);
    ``DiskKnowledgeStore`` writes in place, so a crash mid-write can leave a
    torn file — which :meth:`load` treats as corruption and reports via
    ``needs_rebuild`` rather than serving a partial index (ADR 0036). An index
    written by an older format version is rejected the same way: its entries
    predate a field the current indexer diffs on, so it is rebuilt, not served.

    The entries are mirrored in memory, loaded from the store on the first
    awaited operation. ``size`` reads that mirror: it is ``0`` until an await
    (or :meth:`refresh`) has loaded the file.

    Args:
        store: The knowledge store the sidecar persists through.
        path: Sidecar path inside the store. Keep it under
            :data:`~ag2.knowledge.INDEX_PREFIX`, which the indexer excludes.
        embedder_name: Identity of the embedder the vectors came from; stored
            in the header.
        dimension: Vector length; stored in the header.
    """

    def __init__(
        self,
        store: KnowledgeStore,
        path: str = "/memory/.index/index.jsonl",
        *,
        embedder_name: str,
        dimension: int,
    ) -> None:
        self._store = store
        self._path = path
        self._embedder_name = embedder_name
        self._dimension = dimension
        self._entries: dict[str, IndexEntry] = {}
        self._loaded = False
        self.needs_rebuild = False
        """Set by :meth:`refresh` when the stored index was unusable — a header
        mismatch or a corrupt/unverifiable file. The caller should re-index."""

    @property
    def size(self) -> int:
        """Entries in the in-memory mirror (loaded on the first awaited operation)."""
        return len(self._entries)

    async def refresh(self) -> None:
        """Reload the index file from the store, replacing the in-memory mirror."""
        await self._load()

    async def upsert(self, entries: Iterable[IndexEntry]) -> None:
        await self._ensure_loaded()
        for entry in entries:
            self._entries[entry.id] = entry
        await self._save()

    async def delete(self, ids: Sequence[str] = (), paths: Sequence[str] = ()) -> None:
        await self._ensure_loaded()
        id_set = set(ids)
        path_set = set(paths)
        doomed = [eid for eid, entry in self._entries.items() if eid in id_set or entry.path in path_set]
        if not doomed:
            return
        for eid in doomed:
            del self._entries[eid]
        await self._save()

    async def query(
        self,
        vector: Sequence[float],
        *,
        k: int = 5,
        filter: Callable[[IndexEntry], bool] | None = None,
        min_score: float = 0.0,
    ) -> list[ScoredEntry]:
        if k <= 0:
            return []
        await self._ensure_loaded()
        candidates = [entry for entry in self._entries.values() if filter is None or filter(entry)]
        scored = [ScoredEntry(entry, cosine_similarity(vector, entry.vector)) for entry in candidates]
        scored = [scored_entry for scored_entry in scored if scored_entry.score >= min_score]
        return heapq.nlargest(k, scored, key=attrgetter("score"))

    async def entries(self) -> list[IndexEntry]:
        await self._ensure_loaded()
        return list(self._entries.values())

    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            await self._load()

    async def _load(self) -> None:
        content = await self._store.read(self._path)
        self._loaded = True
        self._entries = {}
        self.needs_rebuild = False
        if content is None:
            return  # no index yet — an empty one is the truth, nothing to rebuild
        try:
            header_line, _, body = content.partition("\n")
            header = json.loads(header_line)
            if not isinstance(header, dict):
                raise ValueError("first line is not a header object")
            if header.get("version") != _INDEX_VERSION:
                raise ValueError(f"unsupported index version {header.get('version')!r}")
            if header.get("embedder") != self._embedder_name or header.get("dimension") != self._dimension:
                raise ValueError(
                    f"index was built by {header.get('embedder')!r} at dimension {header.get('dimension')!r}, "
                    f"not {self._embedder_name!r} at {self._dimension}"
                )
            count = int(header.get("count", -1))
            if header.get("checksum") != _checksum(body):
                raise ValueError("checksum mismatch")
            entries: dict[str, IndexEntry] = {}
            for line in body.splitlines():
                if not line.strip():
                    continue
                entry = _entry_from_dict(json.loads(line))
                entries[entry.id] = entry
            if count != len(entries):
                raise ValueError(f"header claims {count} entries, file holds {len(entries)}")
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Discarding unusable memory index at %s (%s); set needs_rebuild", self._path, exc)
            self._entries = {}
            self.needs_rebuild = True
            return
        self._entries = entries

    async def _save(self) -> None:
        lines = [json.dumps(_entry_to_dict(entry)) for entry in self._entries.values()]
        body = "\n".join(lines)
        header = json.dumps({
            "version": _INDEX_VERSION,
            "embedder": self._embedder_name,
            "dimension": self._dimension,
            "count": len(lines),
            "checksum": _checksum(body),
        })
        await self._store.write(self._path, f"{header}\n{body}" if body else f"{header}\n")


def _checksum(body: str) -> str:
    return f"sha1:{hashlib.sha1(body.encode('utf-8')).hexdigest()}"
