# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""MemoryRetriever — query the memory index and shape the results.

Retrieval is one embed call plus one index query. On top of raw similarity the
retriever applies the two shapings prompt injection needs:

* **MMR** (maximal marginal relevance) when ``diversity > 0`` — trades a little
  raw relevance for coverage, so five chunks of the same summary do not crowd
  out everything else.
* **Path dedup** — at most ``per_path`` chunks per source file (best first),
  the default being the single best chunk of each file.

The chunk text comes from the entry's own metadata, so a retrieval is two async
calls total — no per-hit store reads.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._vector import cosine_similarity
from .embedder import Embedder
from .index import IndexEntry, MemoryIndex, ScoredEntry
from .indexer import IndexSyncReport, KnowledgeIndexer

__all__ = ("MemoryRetriever", "RetrievedMemory")

_POOL_FACTOR = 4
"""How much larger than the requested k the candidate pool is when a shaping
(MMR or multi-chunk-per-path) needs room to work on."""


@dataclass(frozen=True, slots=True)
class RetrievedMemory:
    """One retrieved chunk, ready to inject.

    ``score`` is the raw similarity to the query — not an MMR-adjusted value —
    so results stay comparable across configurations.
    """

    path: str
    chunk: int
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryRetriever:
    """Queries a memory index for the chunks closest to a text query.

    Args:
        index: The index to query.
        embedder: Embeds the query text; must be the embedder that built the
            index (a persisted index enforces this via its header).
        indexer: Optional indexer this retriever can run via :meth:`sync`,
            for callers that refresh the index lazily (e.g. an assembly
            policy's ``sync_on_apply``).
    """

    def __init__(self, index: MemoryIndex, embedder: Embedder, indexer: KnowledgeIndexer | None = None) -> None:
        self._index = index
        self._embedder = embedder
        self.indexer = indexer

    async def sync(self) -> IndexSyncReport | None:
        """Run the indexer when one was provided; ``None`` means there is nothing to sync."""
        if self.indexer is None:
            return None
        return await self.indexer.sync()

    async def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        min_score: float = 0.0,
        filter: Callable[[IndexEntry], bool] | None = None,
        diversity: float = 0.0,
        per_path: int = 1,
    ) -> list[RetrievedMemory]:
        """Return the top chunks for ``query``, shaped and deduplicated.

        Args:
            query: The text to search for.
            k: Maximum results returned.
            min_score: Drop results scoring below this (inclusive keep).
            filter: Optional predicate on index entries — e.g. exclude
                superseded facts by their metadata.
            diversity: MMR strength in ``[0, 1]``. ``0`` ranks purely by
                similarity; higher values penalize candidates too similar to
                the already-selected ones.
            per_path: How many chunks of one source file to return at most,
                best first. The default ``1`` makes results one-per-file.

        Results carry each file's best-ranked chunks in presentation order —
        similarity order when ``diversity == 0``, MMR order otherwise — and
        ``score`` always remains the raw similarity.
        """
        if k <= 0 or not query.strip():
            return []
        if not 0.0 <= diversity <= 1.0:
            raise ValueError(f"diversity must be within [0, 1], got {diversity}")
        if per_path <= 0:
            raise ValueError(f"per_path must be positive, got {per_path}")

        [query_vector] = await self._embedder.embed([query])
        pool = k if (diversity == 0.0 and per_path == 1) else max(k * per_path * _POOL_FACTOR, k)
        scored = await self._index.query(query_vector, k=pool, filter=filter, min_score=min_score)
        ranked = _mmr(query_vector, scored, diversity) if diversity > 0.0 else scored
        selected = _dedup_by_path(ranked, k, per_path)
        return [
            RetrievedMemory(
                path=scored_entry.entry.path,
                chunk=scored_entry.entry.chunk,
                text=str(scored_entry.entry.metadata.get("text", "")),
                score=scored_entry.score,
                metadata=scored_entry.entry.metadata,
            )
            for scored_entry in selected
        ]


def _mmr(query_vector: Sequence[float], scored: list[ScoredEntry], diversity: float) -> list[ScoredEntry]:
    """Greedily re-order candidates by maximal marginal relevance.

    Picks, each round, the candidate maximizing
    ``(1 - diversity) * similarity(query) - diversity * max similarity to the
    already-picked``. Ties keep the more relevant candidate, so the order is
    deterministic.
    """
    remaining = list(scored)
    selected: list[ScoredEntry] = []
    while remaining:
        best_index = 0
        best_value = -math.inf
        for position, candidate in enumerate(remaining):
            redundancy = max(
                (cosine_similarity(candidate.entry.vector, chosen.entry.vector) for chosen in selected),
                default=0.0,
            )
            value = (1.0 - diversity) * candidate.score - diversity * redundancy
            if value > best_value:
                best_index = position
                best_value = value
        selected.append(remaining.pop(best_index))
    return selected


def _dedup_by_path(ranked: list[ScoredEntry], k: int, per_path: int) -> list[ScoredEntry]:
    """First ``k`` entries, at most ``per_path`` from any one source file."""
    selected: list[ScoredEntry] = []
    counts: dict[str, int] = {}
    for scored_entry in ranked:
        if len(selected) >= k:
            break
        path = scored_entry.entry.path
        if counts.get(path, 0) >= per_path:
            continue
        counts[path] = counts.get(path, 0) + 1
        selected.append(scored_entry)
    return selected
