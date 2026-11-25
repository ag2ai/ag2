# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Relevance-based retrieval over the knowledge store.

Everything here is optional and additive: the store keeps its filesystem
semantics, and these pieces add a content index on top of it —

* :class:`Embedder` turns text into vectors (:class:`HashingEmbedder` is the
  dependency-free default; :class:`CallableEmbedder` wraps any model call).
* :class:`MemoryIndex` stores chunk vectors — in memory
  (:class:`InMemoryIndex`) or persisted through the store itself
  (:class:`StoreBackedIndex`, see ADR 0036).
* :class:`KnowledgeIndexer` keeps the index in step with the store's files.
* :class:`MemoryRetriever` answers a text query with ranked, deduplicated
  chunks.

:class:`~ag2.policies.RetrievalMemoryPolicy` wires a retriever into context
assembly; :class:`~ag2.knowledge.FactExtractionAggregate` feeds it fact records
with a lifecycle.
"""

from .chunking import chunk_id, chunk_markdown, meta_hash, text_hash
from .embedder import CallableEmbedder, Embedder, HashingEmbedder
from .index import InMemoryIndex, IndexEntry, MemoryIndex, ScoredEntry, StoreBackedIndex
from .indexer import IndexSyncReport, KnowledgeIndexer
from .retriever import MemoryRetriever, RetrievedMemory

__all__ = (
    "CallableEmbedder",
    "Embedder",
    "HashingEmbedder",
    "InMemoryIndex",
    "IndexEntry",
    "IndexSyncReport",
    "KnowledgeIndexer",
    "MemoryIndex",
    "MemoryRetriever",
    "RetrievedMemory",
    "ScoredEntry",
    "StoreBackedIndex",
    "chunk_id",
    "chunk_markdown",
    "meta_hash",
    "text_hash",
)
