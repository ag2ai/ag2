# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""KnowledgeIndexer — keeps a memory index in step with the store's files.

Sync walks the configured roots through the store (so any backend works),
chunks what it finds, and re-embeds only content whose hash changed. A chunk
whose text is unchanged but whose metadata moved — a fact's status flipping to
``superseded``, an expiry appearing — is refreshed in place instead: the vector
describes the text, and the text did not change, so no embedding call is spent
on it. Files that disappeared — and chunks that no longer exist within a file —
are removed from the index, so a deleted fact stops serving stale vectors. One
unreadable file is reported and skipped; it never aborts the pass.

The walk is bounded: ``max_files`` caps how many files a sync touches and
``max_bytes_per_file`` skips oversized files (both reported in the returned
:class:`IndexSyncReport`), so a runaway store cannot turn one sync into an
unbounded job. Files skipped for either limit have their entries removed — the
index only serves content the pass could verify. Entries for a file whose read
*errored* are left in place and re-verified on a later successful pass.

After a successful sync, every entry corresponds to content this pass verified:
either its hash matched the stored entry, or the entry was written from the file
just read.
"""

import json
import logging
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from ag2.knowledge.base import KnowledgeStore
from ag2.knowledge.constants import CONVERSATIONS_PREFIX, FACTS_PREFIX, INDEX_PREFIX, LOG_PREFIX

from .chunking import chunk_id, chunk_markdown, meta_hash, text_hash
from .embedder import Embedder
from .index import IndexEntry, MemoryIndex

__all__ = ("IndexSyncReport", "KnowledgeIndexer")

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IndexSyncReport:
    """What one :meth:`KnowledgeIndexer.sync` pass did.

    Attributes:
        added: Entries written for chunks that had no index entry before.
        updated: Entries re-embedded because their chunk's content changed.
        updated_metadata: Entries whose metadata was refreshed in place — text
            unchanged (no re-embed), metadata changed (e.g. a fact's status).
        removed: Entries deleted — their file is gone, the chunk vanished, or
            the file was skipped by a limit.
        unchanged: Chunks whose text and metadata both matched the stored
            entry, not rewritten.
        skipped: ``{path: reason}`` for files the pass declined to index.
        errors: ``{path: message}`` for files that could not be read or whose
            embedding failed; the pass continued past each one.
    """

    added: int = 0
    updated: int = 0
    updated_metadata: int = 0
    removed: int = 0
    unchanged: int = 0
    skipped: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        """Whether the pass wrote anything."""
        return bool(self.added or self.updated or self.updated_metadata or self.removed)


@dataclass(frozen=True, slots=True)
class _Chunk:
    """One chunk of one file, ready to compare against and write into the index."""

    path: str
    index: int
    text: str
    hash: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Pending:
    """A changed chunk awaiting embedding, remembering whether it replaces an entry."""

    chunk: _Chunk
    existed_before: bool


class KnowledgeIndexer:
    """Walks the store's memory roots and maintains the index over their chunks.

    Args:
        store: The knowledge store to walk and read.
        index: The index to upsert into and prune.
        embedder: Embeds chunk text; its ``name``/``dimension`` must match the
            index's header — a :class:`StoreBackedIndex` loaded under a mismatch
            reports ``needs_rebuild`` and the next sync rebuilds it.
        roots: Store prefixes to walk recursively.
        include: Optional predicate on file paths; only files passing it are
            indexed. ``None`` includes everything under the roots.
        exclude: Prefixes never indexed, whatever the roots say — by default
            the index sidecar itself (``/memory/.index/``) and the event logs.
        max_files: Files per sync; beyond this many are skipped and reported.
        max_bytes_per_file: Files whose content exceeds this are skipped and
            reported.
        batch_size: Texts per ``embed`` call.
    """

    def __init__(
        self,
        store: KnowledgeStore,
        index: MemoryIndex,
        embedder: Embedder,
        *,
        roots: Sequence[str] = (CONVERSATIONS_PREFIX, FACTS_PREFIX),
        include: Callable[[str], bool] | None = None,
        exclude: Sequence[str] = (INDEX_PREFIX, LOG_PREFIX),
        max_files: int = 1000,
        max_bytes_per_file: int = 200_000,
        batch_size: int = 32,
    ) -> None:
        self.store = store
        self.index = index
        self.embedder = embedder
        self._roots = tuple(roots)
        self._include = include
        self._exclude = tuple(exclude)
        self._max_files = max_files
        self._max_bytes_per_file = max_bytes_per_file
        self._batch_size = max(1, batch_size)

    async def sync(self) -> IndexSyncReport:
        """Bring the index in line with the store.

        Text changes re-embed; metadata-only changes (a fact's status, expiry)
        refresh the entry in place; a bad file is reported, not fatal.
        """
        report = IndexSyncReport()
        files = await self._walk(report)
        existing = {(entry.path, entry.chunk): entry for entry in await self.index.entries()}

        keep: dict[tuple[str, int], IndexEntry] = {}
        pending: list[_Pending] = []
        for path in files:
            try:
                content = await self.store.read(path)
                if content is None:
                    continue  # vanished between the walk and the read — pruned below
                if len(content.encode("utf-8")) > self._max_bytes_per_file:
                    report.skipped[path] = f"file exceeds max_bytes_per_file ({self._max_bytes_per_file})"
                    continue
                for chunk in _chunks_for(path, content):
                    prior = existing.get((chunk.path, chunk.index))
                    if prior is not None and prior.text_hash == chunk.hash:
                        current_meta = meta_hash(chunk.metadata)
                        if prior.meta_hash == current_meta:
                            keep[(chunk.path, chunk.index)] = prior
                            report.unchanged += 1
                        else:
                            # Metadata-only change — a fact's status flipped, an
                            # expiry appeared. Same text, same vector, same id:
                            # refresh in place, no embedding call.
                            keep[(chunk.path, chunk.index)] = IndexEntry(
                                id=prior.id,
                                path=prior.path,
                                chunk=prior.chunk,
                                text_hash=prior.text_hash,
                                vector=prior.vector,
                                metadata=chunk.metadata,
                                updated_at=time.time(),
                                meta_hash=current_meta,
                            )
                            report.updated_metadata += 1
                    else:
                        pending.append(_Pending(chunk, prior is not None))
            except Exception as exc:
                report.errors[path] = f"{type(exc).__name__}: {exc}"
                logger.warning("Skipping unreadable file %s during index sync: %s", path, exc)
                for (entry_path, chunk_index), entry in existing.items():
                    if entry_path == path:
                        keep[(entry_path, chunk_index)] = entry

        await self._embed(pending, keep, report)
        kept_ids = {entry.id for entry in keep.values()}
        removed_ids = [
            entry.id
            for (path, _chunk_index), entry in existing.items()
            if _under_any(path, self._roots) and entry.id not in kept_ids
        ]
        if removed_ids:
            await self.index.delete(ids=removed_ids)
            report.removed = len(removed_ids)
        writes = [entry for key, entry in keep.items() if existing.get(key) is not entry]
        if writes:
            await self.index.upsert(writes)
        return report

    async def _embed(
        self, pending: list[_Pending], keep: dict[tuple[str, int], IndexEntry], report: IndexSyncReport
    ) -> None:
        """Embed pending chunks in batches; count and keep only what embedded."""
        for start in range(0, len(pending), self._batch_size):
            batch = pending[start : start + self._batch_size]
            try:
                vectors = await self.embedder.embed([item.chunk.text for item in batch])
                if len(vectors) != len(batch):
                    raise ValueError(f"embedder returned {len(vectors)} vectors for {len(batch)} texts")
            except Exception as exc:
                logger.warning("Embedding failed for %d chunk(s): %s", len(batch), exc)
                for item in batch:
                    report.errors.setdefault(item.chunk.path, f"embedding failed: {type(exc).__name__}: {exc}")
                continue
            now = time.time()
            for item, vector in zip(batch, vectors):
                keep[(item.chunk.path, item.chunk.index)] = IndexEntry(
                    id=chunk_id(item.chunk.path, item.chunk.index, item.chunk.hash),
                    path=item.chunk.path,
                    chunk=item.chunk.index,
                    text_hash=item.chunk.hash,
                    vector=vector,
                    metadata=item.chunk.metadata,
                    updated_at=now,
                    meta_hash=meta_hash(item.chunk.metadata),
                )
                if item.existed_before:
                    report.updated += 1
                else:
                    report.added += 1

    async def _walk(self, report: IndexSyncReport) -> list[str]:
        """All indexable file paths under the roots, sorted, bounded by ``max_files``."""
        found: set[str] = set()
        for root in self._roots:
            frontier = [root if root.endswith("/") else f"{root}/"]
            while frontier:
                prefix = frontier.pop()
                for entry in await self.store.list(prefix):
                    path = f"{prefix}{entry}"
                    if path.endswith("/"):
                        frontier.append(path)
                    elif not _under_any(path, self._exclude) and (self._include is None or self._include(path)):
                        found.add(path)
        ordered = sorted(found)
        for path in ordered[self._max_files :]:
            report.skipped[path] = f"max_files ({self._max_files}) exceeded"
        return ordered[: self._max_files]


def _chunks_for(path: str, content: str) -> list[_Chunk]:
    """Chunk one file's content, choosing text and metadata by file type.

    Markdown goes through :func:`chunk_markdown`. A JSON object carrying a
    string ``text`` field — a fact record — is indexed as one chunk of that
    text (the content the fact states), with the record's other scalar fields
    surfaced as entry metadata so queries can filter on them (e.g. by
    ``status``). Any other JSON, and any other file type, indexes as one chunk
    of the raw content.
    """
    if path.endswith(".json"):
        text, metadata = _json_chunk(content)
        return [_Chunk(path, 0, text, text_hash(text), metadata)]
    return [
        _Chunk(path, index, chunk, text_hash(chunk), {"text": chunk})
        for index, chunk in enumerate(chunk_markdown(content))
    ]


def _json_chunk(content: str) -> tuple[str, dict[str, Any]]:
    """``(chunk text, metadata)`` for one JSON file."""
    metadata: dict[str, Any] = {}
    with suppress(ValueError):
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if value is None or isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
    text = metadata.get("text")
    if not isinstance(text, str) or not text:
        text = content
    metadata["text"] = text
    return text, metadata


def _under_any(path: str, prefixes: Sequence[str]) -> bool:
    """Whether ``path`` equals one of the prefixes or lives underneath it."""
    return any(
        path == prefix.rstrip("/") or path.startswith(prefix if prefix.endswith("/") else f"{prefix}/")
        for prefix in prefixes
    )
