---
status: proposed
date: 2026-11-07
---

# Working memory uses conditional store updates

## Context

WorkingMemoryAggregate reads a document, asks a model to merge new facts, and
writes a replacement. Independent strategies and streams can read the same
snapshot. Locking only the final write silently loses one stream's facts;
an instance-local asyncio lock cannot coordinate independent store handles.

## Decision

Add the optional structural CompareAndSwapStore capability, without extending
the required KnowledgeStore protocol. Its conditional replacement distinguishes
a missing file from an empty file. Built-in stores implement it: Memory commits
without yielding before mutation, SQLite uses a short write transaction, Redis
uses Lua, and Disk coordinates cooperating POSIX store handles with an advisory
lock on the root directory inode. Disk keeps that inode when clearing `/`;
mutations run off the event loop and use the same lock. External filesystem
writers that bypass this protocol are outside the guarantee.

WorkingMemoryAggregate merges outside storage locks and commits only if its
snapshot is still current. A conflict rereads and remerges; max_attempts bounds
the model calls (default 3). Exhaustion raises rather than overwriting new data.
Each paid attempt emits its source UsageEvent. AggregationCompleted counts and
sums those records rather than reporting only the final attempt.

Custom stores must implement the optional capability or use LockedKnowledgeStore
with an appropriate shared lock. The adapter holds that lock only for the short
conditional replacement, not during model inference. All participating writers
must use the same adapter/lock; operations must complete within its existing
30-second lease. A missing capability fails before spending model tokens.

## Consequences

No new dependencies or global lock registry. Under no contention there is one
model call. Contention can spend additional tokens; bounded retries can fail,
and the existing plugin emits AggregationFailed. Cancellation is not retried.
For SQLite, cancelling a conditional update waits for its executor transaction
to finish before releasing that handle's connection lock.

This is content-based optimistic concurrency, not a revision-history API. An
ABA change back to identical content is treated as the same snapshot. Disk
requires a local POSIX filesystem with working directory flock semantics;
non-cooperating external editors and filesystem-specific distributed locking
are not covered. A model can still omit facts in its answer; conditional writes
prevent storage races, not model summarization errors.
