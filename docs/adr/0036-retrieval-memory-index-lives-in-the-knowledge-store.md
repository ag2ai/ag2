---
status: proposed
date: 2026-11-08
---

# 0036. The retrieval memory index lives in the knowledge store

## Context

Retrieval over the knowledge layer needs an index of chunk vectors. The obvious
shapes — a dedicated vector database, or an in-process index persisted to its
own file — both add a second durable home for memory state. The knowledge layer
already has one: `KnowledgeStore` is a path-based virtual filesystem with
memory, disk, sqlite, and redis backends, and every agent already carries one.
Index state is memory state; keeping it anywhere else would fork persistence,
bootstrap, and backend choice for no capability the store fails to provide.

An index that persists through the store must also never mix vector spaces:
embedders differ in dimension and geometry, and an index loaded under a
different embedder produces scores that look plausible and mean nothing.

## Decision

The retrieval index is persisted *through* the `KnowledgeStore` as a sidecar
file under `/memory/.index/` (`StoreBackedIndex`), not to a separate service or
file location. `KnowledgeIndexer` walks the store's memory roots, so the same
walk serves every backend; no new backend, bootstrap, or dependency is
introduced.

The index file's header pins `version`, the `embedder` name, and the vector
`dimension`, and carries a checksum over the body. On load, a version,
embedder, or dimension mismatch — or any corruption the checksum catches —
yields an empty index with `needs_rebuild=True` (a logged warning), never a
mixed vector space and never a crash. The next indexer sync rebuilds the index
from the store's content, which is the source of truth the index only mirrors.
Bumping the version rejects indexes written by older formats too: an index that
predates a field the current indexer diffs on is rebuilt, not served with
silently stale metadata.

Entries carry two change signals: a `text_hash` over the chunk text and a
`meta_hash` over the metadata dict (canonical JSON). The indexer re-embeds only
when the text changed; a metadata-only change — a fact's `status` flipping to
`superseded`, an expiry appearing — refreshes the entry in place at no
embedding cost, because the vector describes the text and the text did not
change.

Superseded and forgotten fact records stay on disk (provenance, audit) and are
excluded at retrieval time by their record's `status` metadata, rather than
being deleted or moved aside. The index metadata is an optimization, not the
authority: the assembly policy re-reads each selected fact's file through the
`FactStore` before injecting it and drops anything that is no longer `active`
or is past its `expires_at`, so an index that lags a status change cannot
inject a dead fact. Among the still-active facts sharing a subject and kind,
only the newest `updated_at` is injected — the readable half of a supersede
that crashed between its two writes must not reach the model as both sides of
a contradiction.

## Consequences

Any `KnowledgeStore` backend becomes a durable vector index with no new
infrastructure; agents with a store already have everything retrieval needs.
The cost is the sidecar rewrite: `StoreBackedIndex` saves the whole file on each
write, which is proportional to index size — acceptable for the memory-shaped
volumes this serves, and the reason an external vector store remains the right
answer past them.

The sidecar path is excluded from indexing by the indexer's default exclude
list, so the index never indexes itself. Atomicity follows the backend:
memory and sqlite writes are atomic, the disk backend writes in place — a torn
disk write lands in the checksum-mismatch path above, which is the designed
recovery rather than a failure mode. `needs_rebuild` is reported, not acted on
automatically, so a caller can decide when a rebuild's embedding cost runs.
