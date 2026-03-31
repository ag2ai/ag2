# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""KnowledgeStore — virtual filesystem for actor knowledge.

A KnowledgeStore provides filesystem semantics over any storage backend.
It stores everything an actor is associated with throughout its lifetime:
operational logs, external artifacts, summaries, and working memory.

Filesystem semantics are used because:
1. LLMs are trained on filesystem operations (read, write, list, delete)
2. Hierarchical paths give free semantic grouping without schema design
3. Any backend (memory, disk, S3, Redis) can implement path-based key-value
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import UUID

from autogen.beta.events._serialization import (
    import_event_class,
    qualified_name,
)

if TYPE_CHECKING:
    from autogen.beta.events import BaseEvent

    from .network.primitives.envelope import EventRegistry


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class KnowledgeStore(Protocol):
    """Virtual path-based store for actor knowledge.

    Provides filesystem semantics over any storage backend.
    Paths use Unix conventions: /dir/subdir/file.txt
    Directories are implicit -- writing /a/b/c.txt implies /a/ and /a/b/ exist.
    Listing returns immediate children. Directory entries end with '/'.
    """

    async def read(self, path: str) -> str | None:
        """Read content at path. Returns None if not found."""
        ...

    async def write(self, path: str, content: str) -> None:
        """Write content to path. Creates parent directories implicitly."""
        ...

    async def list(self, path: str = "/") -> list[str]:
        """List immediate children of a directory path.

        Returns relative names. Directories end with '/'.
        Example: list("/log/") might return ["stream-abc.jsonl", "stream-def.jsonl"]
        """
        ...

    async def delete(self, path: str) -> None:
        """Delete entry at path. No-op if not found."""
        ...

    async def exists(self, path: str) -> bool:
        """Check if path exists."""
        ...


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------


def _normalize(path: str) -> str:
    """Normalize path: ensure leading /, collapse //, strip trailing /."""
    if not path.startswith("/"):
        path = "/" + path
    while "//" in path:
        path = path.replace("//", "/")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return path


class MemoryKnowledgeStore:
    """In-memory KnowledgeStore. Development default.

    Backed by a flat dict. Paths are keys. Directories are inferred
    from stored paths via prefix matching.
    """

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def read(self, path: str) -> str | None:
        return self._data.get(_normalize(path))

    async def write(self, path: str, content: str) -> None:
        self._data[_normalize(path)] = content

    async def list(self, path: str = "/") -> list[str]:
        prefix = _normalize(path).rstrip("/") + "/"

        children: set[str] = set()
        for key in self._data:
            if not key.startswith(prefix):
                continue
            remainder = key[len(prefix) :]
            if "/" in remainder:
                children.add(remainder.split("/")[0] + "/")
            else:
                children.add(remainder)
        return sorted(children)

    async def delete(self, path: str) -> None:
        normalized = _normalize(path)
        # Delete exact match
        self._data.pop(normalized, None)
        # Delete children (if deleting a directory)
        prefix = normalized.rstrip("/") + "/"
        to_delete = [k for k in self._data if k.startswith(prefix)]
        for k in to_delete:
            del self._data[k]

    async def exists(self, path: str) -> bool:
        normalized = _normalize(path)
        if normalized in self._data:
            return True
        # Check if it's a directory (any children exist)
        prefix = normalized.rstrip("/") + "/"
        return any(k.startswith(prefix) for k in self._data)


# ---------------------------------------------------------------------------
# Locked wrapper
# ---------------------------------------------------------------------------


class LockedKnowledgeStore:
    """Wraps a KnowledgeStore with a Lock for concurrent access safety.

    Reads are not locked (safe for concurrent access on all backends).
    Writes and deletes acquire the lock.
    """

    def __init__(self, store: KnowledgeStore, lock: Any) -> None:
        self._store = store
        self._lock = lock

    async def read(self, path: str) -> str | None:
        return await self._store.read(path)

    async def write(self, path: str, content: str) -> None:
        acquired = await self._lock.acquire(f"store:write:{path}", ttl=30.0)
        if not acquired:
            raise RuntimeError(f"Failed to acquire write lock for {path}")
        try:
            await self._store.write(path, content)
        finally:
            await self._lock.release(f"store:write:{path}")

    async def list(self, path: str = "/") -> list[str]:
        return await self._store.list(path)

    async def delete(self, path: str) -> None:
        acquired = await self._lock.acquire(f"store:write:{path}", ttl=30.0)
        if not acquired:
            raise RuntimeError(f"Failed to acquire delete lock for {path}")
        try:
            await self._store.delete(path)
        finally:
            await self._lock.release(f"store:write:{path}")

    async def exists(self, path: str) -> bool:
        return await self._store.exists(path)


# ---------------------------------------------------------------------------
# EventLogWriter — WAL persistence for stream events
# ---------------------------------------------------------------------------

StreamId = UUID


class EventLogWriter:
    """Persists stream events to the knowledge store as WAL entries.

    Each event is serialized as a JSON line with a type tag for
    deserialization. Uses append-segmented writes: dropped events from
    compaction go to numbered segment files, final events go to the
    main log file.
    """

    def __init__(self, store: KnowledgeStore) -> None:
        self._store = store

    async def persist(self, stream_id: StreamId, events: Iterable[BaseEvent]) -> None:
        """Write final events to /log/{stream_id}.jsonl."""
        path = f"/log/{stream_id}.jsonl"
        lines = self._serialize_events(events)
        await self._store.write(path, "\n".join(lines))

    async def persist_dropped(self, stream_id: StreamId, events: Iterable[BaseEvent]) -> None:
        """Write compaction-dropped events to /log/{stream_id}.dropped-{n}.jsonl.

        Discovers existing segments in the store to avoid overwriting.
        """
        prefix = f"{stream_id}.dropped-"
        entries = await self._store.list("/log/")
        existing = [e for e in entries if e.startswith(prefix) and e.endswith(".jsonl")]
        n = len(existing) + 1
        path = f"/log/{stream_id}.dropped-{n}.jsonl"
        lines = self._serialize_events(events)
        await self._store.write(path, "\n".join(lines))

    async def load(
        self,
        stream_id: StreamId,
        registry: EventRegistry | None = None,
    ) -> list[BaseEvent]:
        """Load events from WAL files: all dropped segments in order, then final.

        Returns typed BaseEvent instances. Unknown types become UnknownEvent.
        """

        all_events: list[BaseEvent] = []

        # Read dropped segments in order
        entries = await self._store.list("/log/")
        prefix = f"{stream_id}.dropped-"
        segments = sorted(
            [e for e in entries if e.startswith(prefix) and e.endswith(".jsonl")],
            key=lambda e: int(e[len(prefix) : -len(".jsonl")]),
        )
        for segment in segments:
            events = await self._load_file(f"/log/{segment}", registry)
            all_events.extend(events)

        # Read final log
        final = await self._load_file(f"/log/{stream_id}.jsonl", registry)
        all_events.extend(final)

        return all_events

    def _serialize_events(self, events: Iterable[BaseEvent]) -> list[str]:
        lines: list[str] = []
        for event in events:
            record = {
                "type": qualified_name(event),
                "data": event.to_dict(),
            }
            lines.append(json.dumps(record, default=str))
        return lines

    async def _load_file(
        self,
        path: str,
        registry: EventRegistry | None = None,
    ) -> list[BaseEvent]:

        from .network.events import UnknownEvent

        content = await self._store.read(path)
        if not content:
            return []

        events: list[BaseEvent] = []
        for line in content.strip().split("\n"):
            if not line:
                continue
            record = json.loads(line)
            event_type = record["type"]
            event_data = record["data"]

            # Resolve type — try registry, then import
            cls: type[BaseEvent] | None = None
            if registry is not None:
                cls = registry.resolve(event_type)
            if cls is None:
                cls = import_event_class(event_type)

            if cls is not None:
                try:
                    events.append(cls.from_dict(event_data))
                except Exception:
                    events.append(UnknownEvent(type_name=event_type, data=event_data))
            else:
                events.append(UnknownEvent(type_name=event_type, data=event_data))

        return events


# ---------------------------------------------------------------------------
# Store bootstrapping
# ---------------------------------------------------------------------------


@runtime_checkable
class StoreBootstrap(Protocol):
    """Initializes a knowledge store with a starting structure.

    Called once when an actor first runs with a store. Subsequent
    runs skip bootstrapping (detected via a sentinel file).
    """

    async def bootstrap(self, store: KnowledgeStore, actor_name: str) -> None:
        """Create initial store structure."""
        ...


class DefaultBootstrap:
    """Creates the standard knowledge store layout with SKILL.md files."""

    async def bootstrap(self, store: KnowledgeStore, actor_name: str) -> None:
        # Sentinel is written by Actor._execute() before calling bootstrap,
        # so we don't write it here to avoid a double-write.

        await store.write(
            "/SKILL.md",
            f"# {actor_name} Knowledge Store\n\n"
            "This is your persistent knowledge store. Use the `knowledge` tool to manage it.\n\n"
            "## Directories\n"
            "- `/log/` -- Conversation history (auto-managed)\n"
            "- `/artifacts/` -- External files and data\n"
            "- `/memory/` -- Working memory and summaries (auto-managed)\n",
        )

        await store.write(
            "/log/SKILL.md",
            "Conversation logs. Each file is a JSONL record of one conversation's events. "
            "Auto-populated by the framework after each conversation.",
        )

        await store.write(
            "/artifacts/SKILL.md",
            "External data: uploaded files, downloaded content, reference materials. "
            "Write here to store data you want to reference later.",
        )

        await store.write(
            "/memory/SKILL.md",
            "Working memory and conversation summaries. "
            "`working.md` contains your current persistent state. "
            "`conversations/` contains per-conversation summaries. "
            "Both are auto-updated by aggregation strategies.",
        )
