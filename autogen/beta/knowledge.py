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

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import UUID

from autogen.beta.events._serialization import (
    import_event_class,
    qualified_name,
)

if TYPE_CHECKING:
    from autogen.beta.events import BaseEvent


ChangeCallback = Callable[[str], Awaitable[None]]


class ChangeSubscription(Protocol):
    """Handle returned by :meth:`KnowledgeStore.on_change`.

    Closing the subscription stops delivery of change notifications. This
    is filesystem-level reactivity for the backing store — not to be
    confused with ``autogen.beta.watch.Watch``, which is the event- and
    time-pattern trigger system used by the framework-core ``Scheduler``.
    """

    async def close(self) -> None:
        """Stop receiving change notifications."""
        ...


class NoopChangeSubscription:
    """Sentinel returned by backends that cannot observe changes efficiently.

    The hub falls back to polling when it sees a
    :class:`NoopChangeSubscription`.
    """

    async def close(self) -> None:
        return None


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

    The network layer uses the same protocol to back the hub's virtual file
    system; see :mod:`autogen.beta.network`. Three methods beyond the basic
    CRUD are required for WAL-backed sessions: ``append`` and ``read_range``
    are mandatory, while ``on_change`` is optional (backends that cannot
    observe changes efficiently return a :class:`NoopChangeSubscription`
    and callers fall back to polling).
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

    async def append(self, path: str, content: str) -> int:
        """Atomically append ``content`` to the file at ``path``.

        Creates the file (and its parents) if it does not exist. Returns the
        byte offset at which ``content`` was written, so callers can record a
        cursor for later ``read_range`` calls.
        """
        ...

    async def read_range(self, path: str, start: int, end: int | None = None) -> str:
        """Read the byte slice ``[start, end)`` of the file at ``path``.

        ``end`` of ``None`` means "up to the current end of file". Returns an
        empty string if the file does not exist. Slices are returned as UTF-8
        text; callers that append multi-byte content must align offsets to
        character boundaries themselves.
        """
        ...

    async def on_change(self, path: str, callback: ChangeCallback) -> ChangeSubscription:
        """Subscribe to change notifications at ``path``.

        Backends that can observe changes efficiently invoke
        ``callback(path)`` whenever a file under ``path`` changes. Backends
        that cannot must return :class:`NoopChangeSubscription`; the hub
        then polls on a short interval instead.
        """
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
        self._append_lock = asyncio.Lock()
        self._subscribers: dict[str, list[ChangeCallback]] = {}

    async def read(self, path: str) -> str | None:
        return self._data.get(_normalize(path))

    async def write(self, path: str, content: str) -> None:
        normalized = _normalize(path)
        self._data[normalized] = content
        await self._notify(normalized)

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
        affected: list[str] = []
        if normalized in self._data:
            del self._data[normalized]
            affected.append(normalized)
        prefix = normalized.rstrip("/") + "/"
        for key in [k for k in self._data if k.startswith(prefix)]:
            del self._data[key]
            affected.append(key)
        for key in affected:
            await self._notify(key)

    async def exists(self, path: str) -> bool:
        normalized = _normalize(path)
        if normalized in self._data:
            return True
        prefix = normalized.rstrip("/") + "/"
        return any(k.startswith(prefix) for k in self._data)

    async def append(self, path: str, content: str) -> int:
        normalized = _normalize(path)
        async with self._append_lock:
            existing = self._data.get(normalized, "")
            offset = len(existing.encode("utf-8"))
            self._data[normalized] = existing + content
        await self._notify(normalized)
        return offset

    async def read_range(self, path: str, start: int, end: int | None = None) -> str:
        normalized = _normalize(path)
        existing = self._data.get(normalized)
        if existing is None:
            return ""
        data = existing.encode("utf-8")
        stop = len(data) if end is None else min(end, len(data))
        if start >= stop:
            return ""
        return data[start:stop].decode("utf-8", errors="strict")

    async def on_change(self, path: str, callback: ChangeCallback) -> ChangeSubscription:
        normalized = _normalize(path)
        self._subscribers.setdefault(normalized, []).append(callback)

        subscribers = self._subscribers
        key = normalized

        class _Sub:
            async def close(self) -> None:
                bucket = subscribers.get(key)
                if not bucket:
                    return
                with contextlib.suppress(ValueError):
                    bucket.remove(callback)
                if not bucket:
                    subscribers.pop(key, None)

        return _Sub()

    async def _notify(self, changed_path: str) -> None:
        for subscribed_path, callbacks in list(self._subscribers.items()):
            if changed_path == subscribed_path or changed_path.startswith(subscribed_path.rstrip("/") + "/"):
                for callback in list(callbacks):
                    await callback(changed_path)


# ---------------------------------------------------------------------------
# Disk change-notification helpers (watchdog bridge)
# ---------------------------------------------------------------------------


class _DiskChangeHandler:
    """watchdog ``FileSystemEventHandler`` that bridges to an async callback.

    The watchdog observer runs in a background thread and calls this
    handler synchronously on file events. We translate the physical
    path back into its store-relative virtual form and schedule the
    async ``callback`` on the main event loop via
    :func:`asyncio.run_coroutine_threadsafe`.

    Directory events are ignored — only file-level writes / creations /
    deletions / moves are delivered, matching the ``MemoryKnowledgeStore``
    contract where every change is observed as a file path.
    """

    def __init__(
        self,
        *,
        root: Path,
        virtual_prefix: str,
        loop: asyncio.AbstractEventLoop,
        callback: ChangeCallback,
    ) -> None:
        self._root = root
        self._virtual_prefix = virtual_prefix
        self._loop = loop
        self._callback = callback

    def _virtual_path_for(self, src_path: str) -> str | None:
        try:
            rel = Path(src_path).resolve().relative_to(self._root)
        except ValueError:
            return None
        virtual = "/" + str(rel).replace("\\", "/")
        # Filter events that bubble up from sibling subtrees (recursive
        # observer on a broader path than we subscribed to).
        if self._virtual_prefix != "/":
            prefix = self._virtual_prefix.rstrip("/") + "/"
            if virtual != self._virtual_prefix and not virtual.startswith(prefix):
                return None
        return virtual

    def _dispatch(self, src_path: str) -> None:
        virtual = self._virtual_path_for(src_path)
        if virtual is None:
            return
        coro = self._callback(virtual)
        try:
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        except RuntimeError:
            # Loop is closed — the subscription outlived its owner.
            # Close the orphan coroutine to suppress "was never awaited".
            coro.close()

    # watchdog event dispatch hooks ---------------------------------------
    # watchdog instantiates a ``FileSystemEventHandler`` subclass; we
    # duck-type the five callback names it looks for. Keeping this as a
    # plain class avoids a hard import of ``watchdog.events`` at module
    # load time — the import is lazy in ``DiskKnowledgeStore.on_change``.

    def on_modified(self, event: Any) -> None:
        if getattr(event, "is_directory", False):
            return
        self._dispatch(event.src_path)

    def on_created(self, event: Any) -> None:
        if getattr(event, "is_directory", False):
            return
        self._dispatch(event.src_path)

    def on_deleted(self, event: Any) -> None:
        if getattr(event, "is_directory", False):
            return
        self._dispatch(event.src_path)

    def on_moved(self, event: Any) -> None:
        if getattr(event, "is_directory", False):
            return
        dest = getattr(event, "dest_path", None) or event.src_path
        self._dispatch(dest)

    def dispatch(self, event: Any) -> None:
        """watchdog's entry point. Delegates to the per-type hooks above."""

        event_type = getattr(event, "event_type", "")
        if event_type == "modified":
            self.on_modified(event)
        elif event_type == "created":
            self.on_created(event)
        elif event_type == "deleted":
            self.on_deleted(event)
        elif event_type == "moved":
            self.on_moved(event)


class _DiskChangeSubscription:
    """Handle returned by :meth:`DiskKnowledgeStore.on_change`.

    Wraps the live watchdog ``Observer`` and ensures ``close()`` stops
    and joins the background thread so the subscription never leaks a
    daemon thread past the caller's lifetime.
    """

    def __init__(self, observer: Any) -> None:
        self._observer = observer
        self._closed = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Observer.stop() + join() are blocking — run in the default
        # executor so we don't pin the event loop while the background
        # thread wraps up.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._shutdown)

    def _shutdown(self) -> None:
        with contextlib.suppress(Exception):  # pragma: no cover — watchdog internals
            self._observer.unschedule_all()
        with contextlib.suppress(Exception):  # pragma: no cover
            self._observer.stop()
        with contextlib.suppress(Exception):  # pragma: no cover
            self._observer.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Disk implementation
# ---------------------------------------------------------------------------


class DiskKnowledgeStore:
    """Persistent KnowledgeStore backed by the local filesystem.

    Maps virtual paths directly to real files under a root directory.
    Directories are created on write. Supports macOS and Linux.
    Not supported on Windows (filenames may contain characters that
    are illegal on NTFS such as ``:``, ``?``, ``*``, ``<``, ``>``).

    Example::

        store = DiskKnowledgeStore("/tmp/my-agent")
        await store.write("/artifacts/report.md", "# Report")
        # Creates /tmp/my-agent/artifacts/report.md on disk
    """

    def __init__(self, root: str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        """Map virtual path to real filesystem path."""
        normalized = _normalize(path).lstrip("/")
        resolved = (self._root / normalized).resolve() if normalized else self._root.resolve()
        # Prevent path traversal
        if not str(resolved).startswith(str(self._root.resolve())):
            raise ValueError(f"Path traversal blocked: {path}")
        return resolved

    async def read(self, path: str) -> str | None:
        target = self._resolve(path)
        if not target.is_file():
            return None
        return target.read_text(encoding="utf-8")

    async def write(self, path: str, content: str) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    async def list(self, path: str = "/") -> list[str]:
        target = self._resolve(path)
        if not target.is_dir():
            return []
        children: list[str] = []
        for entry in sorted(target.iterdir()):
            if entry.is_dir():
                children.append(entry.name + "/")
            else:
                children.append(entry.name)
        return children

    async def delete(self, path: str) -> None:
        target = self._resolve(path)
        if target.is_file():
            target.unlink()
        elif target.is_dir():
            import shutil

            shutil.rmtree(target)

    async def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    async def append(self, path: str, content: str) -> int:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = content.encode("utf-8")
        # Open in append-binary so POSIX guarantees atomicity per write().
        with target.open("ab") as fh:
            offset = fh.tell()
            fh.write(payload)
        return offset

    async def read_range(self, path: str, start: int, end: int | None = None) -> str:
        target = self._resolve(path)
        if not target.is_file():
            return ""
        with target.open("rb") as fh:
            fh.seek(start)
            if end is None:
                data = fh.read()
            else:
                span = max(0, end - start)
                data = fh.read(span)
        return data.decode("utf-8", errors="strict")

    async def on_change(self, path: str, callback: ChangeCallback) -> ChangeSubscription:
        """Subscribe to filesystem change notifications under ``path``.

        Uses the ``watchdog`` library to dispatch platform-native events
        (inotify on Linux, FSEvents on macOS, ReadDirectoryChangesW on
        Windows). Falls back to :class:`PollingObserver` if the native
        backend cannot be initialized, and to :class:`NoopChangeSubscription`
        if ``watchdog`` is not installed at all.

        The ``path`` argument is the virtual (store-relative) directory
        to watch. The watcher is recursive: modifications to any file
        beneath ``path`` deliver the file's virtual path to ``callback``.
        If ``path`` does not exist on disk yet, the directory is created
        so watchdog has something to attach to — this matches the
        ``MemoryKnowledgeStore`` contract where "subscribe first, then
        write" is legal.
        """

        try:
            from watchdog.observers import Observer  # type: ignore[import-not-found]
            from watchdog.observers.polling import PollingObserver  # type: ignore[import-not-found]
        except ImportError:
            return NoopChangeSubscription()

        virtual_path = _normalize(path)
        physical_target = self._resolve(virtual_path)
        physical_target.mkdir(parents=True, exist_ok=True)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return NoopChangeSubscription()

        handler = _DiskChangeHandler(
            root=self._root.resolve(),
            virtual_prefix=virtual_path,
            loop=loop,
            callback=callback,
        )

        observer: Any
        try:
            observer = Observer()
            observer.schedule(handler, str(physical_target), recursive=True)
            observer.start()
        except Exception:
            # Native backend unavailable (e.g. inside a container with no
            # inotify caps). Fall back to polling — slower but correct.
            observer = PollingObserver()
            observer.schedule(handler, str(physical_target), recursive=True)
            observer.start()

        return _DiskChangeSubscription(observer)


# ---------------------------------------------------------------------------
# Polling change watcher (shared by Sqlite + Redis backends)
# ---------------------------------------------------------------------------


class _PollingChangeWatcher:
    """Polls a store for changes and dispatches to an async callback.

    Used by backends that cannot observe changes natively (Sqlite,
    Redis). The hub-facing semantics mirror the Memory + Disk stores:

    * Subscribe to a path prefix.
    * Receive ``callback(changed_path)`` asynchronously when the
      backend reports a newer version for any key under that prefix.
    * Close the subscription via :meth:`close` to stop delivery.

    Two interfaces are required from the backend:

    * ``list_versions_under(prefix)`` — returns ``dict[path, version]``
      for every key beneath ``prefix``. The version can be an mtime
      float, a monotonic int, or anything comparable for equality.
    * Normal polling fires at ``interval_s`` (default 0.5s). The
      watcher diffs the current snapshot against its last-seen
      snapshot and fires the callback once per changed key.
    """

    def __init__(
        self,
        *,
        backend: Any,
        prefix: str,
        callback: ChangeCallback,
        interval_s: float = 0.5,
    ) -> None:
        self._backend = backend
        self._prefix = _normalize(prefix)
        self._callback = callback
        self._interval_s = max(0.05, float(interval_s))
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._last_snapshot: dict[str, Any] = {}

    async def start(self) -> None:
        # Seed the baseline so the first tick does not announce every
        # existing file as a change.
        self._last_snapshot = await self._backend.list_versions_under(self._prefix)
        self._task = asyncio.create_task(self._run(), name="store-poll-watcher")

    async def _run(self) -> None:
        while not self._closed:
            try:
                await asyncio.sleep(self._interval_s)
            except asyncio.CancelledError:
                return
            if self._closed:
                return
            try:
                current = await self._backend.list_versions_under(self._prefix)
            except Exception:  # pragma: no cover
                continue
            # Diff: new keys, changed versions, deleted keys.
            for path, version in current.items():
                prior = self._last_snapshot.get(path)
                if prior is None or prior != version:
                    await self._safe_fire(path)
            for path in self._last_snapshot:
                if path not in current:
                    await self._safe_fire(path)
            self._last_snapshot = current

    async def _safe_fire(self, path: str) -> None:
        try:  # noqa: SIM105
            await self._callback(path)
        except Exception:  # pragma: no cover
            pass  # callbacks must never crash the watcher

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None


# ---------------------------------------------------------------------------
# Sqlite implementation
# ---------------------------------------------------------------------------


class SqliteKnowledgeStore:
    """SQLite-backed :class:`KnowledgeStore`.

    Stores every virtual path as a row ``(path, content, version)`` in a
    single ``entries`` table. ``version`` is a monotonic integer bumped
    on every write or append and is the key the polling change watcher
    uses to detect mutations.

    SQLite does not have native change notifications, so ``on_change``
    uses :class:`_PollingChangeWatcher` at a default 500ms interval.
    Callers that need lower latency pass ``poll_interval_s`` to the
    constructor.

    All store operations run on the asyncio loop's default executor so
    the blocking sqlite3 API does not stall the event loop. A single
    connection is serialized with an ``asyncio.Lock`` — SQLite's own
    file locking handles concurrent Python processes if multiple hubs
    share a database.

    This is the same protocol surface as Memory and Disk, so every
    test that parameterizes over stores picks up Sqlite automatically.
    """

    def __init__(
        self,
        path: str,
        *,
        poll_interval_s: float = 0.5,
    ) -> None:
        import sqlite3  # stdlib — no optional install needed.

        self._db_path = path
        self._poll_interval_s = poll_interval_s
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                path TEXT PRIMARY KEY,
                content BLOB NOT NULL,
                version INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.commit()
        self._lock = asyncio.Lock()
        self._version_counter = self._load_max_version()

    def _load_max_version(self) -> int:
        cur = self._conn.execute("SELECT COALESCE(MAX(version), 0) FROM entries")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def _next_version(self) -> int:
        self._version_counter += 1
        return self._version_counter

    async def _run(self, func: Callable[[], Any]) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func)

    async def read(self, path: str) -> str | None:
        normalized = _normalize(path)

        def _op() -> str | None:
            cur = self._conn.execute("SELECT content FROM entries WHERE path = ?", (normalized,))
            row = cur.fetchone()
            if row is None:
                return None
            return row[0].decode("utf-8")

        return await self._run(_op)

    async def write(self, path: str, content: str) -> None:
        normalized = _normalize(path)
        payload = content.encode("utf-8")

        async with self._lock:
            version = self._next_version()

            def _op() -> None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO entries (path, content, version) VALUES (?, ?, ?)",
                    (normalized, payload, version),
                )
                self._conn.commit()

            await self._run(_op)

    async def list(self, path: str = "/") -> list[str]:
        prefix = _normalize(path).rstrip("/") + "/"

        def _op() -> list[str]:
            cur = self._conn.execute(
                "SELECT path FROM entries WHERE path LIKE ?",
                (prefix + "%",),
            )
            children: set[str] = set()
            for (p,) in cur.fetchall():
                remainder = p[len(prefix) :]
                if "/" in remainder:
                    children.add(remainder.split("/")[0] + "/")
                else:
                    children.add(remainder)
            return sorted(children)

        return await self._run(_op)

    async def delete(self, path: str) -> None:
        normalized = _normalize(path)
        prefix = normalized.rstrip("/") + "/"

        async with self._lock:

            def _op() -> None:
                self._conn.execute("DELETE FROM entries WHERE path = ?", (normalized,))
                self._conn.execute("DELETE FROM entries WHERE path LIKE ?", (prefix + "%",))
                self._conn.commit()

            await self._run(_op)

    async def exists(self, path: str) -> bool:
        normalized = _normalize(path)
        prefix = normalized.rstrip("/") + "/"

        def _op() -> bool:
            cur = self._conn.execute("SELECT 1 FROM entries WHERE path = ? LIMIT 1", (normalized,))
            if cur.fetchone() is not None:
                return True
            cur = self._conn.execute(
                "SELECT 1 FROM entries WHERE path LIKE ? LIMIT 1",
                (prefix + "%",),
            )
            return cur.fetchone() is not None

        return await self._run(_op)

    async def append(self, path: str, content: str) -> int:
        normalized = _normalize(path)
        payload = content.encode("utf-8")

        async with self._lock:
            version = self._next_version()

            def _op() -> int:
                cur = self._conn.execute("SELECT content FROM entries WHERE path = ?", (normalized,))
                row = cur.fetchone()
                existing = row[0] if row else b""
                offset = len(existing)
                combined = existing + payload
                self._conn.execute(
                    "INSERT OR REPLACE INTO entries (path, content, version) VALUES (?, ?, ?)",
                    (normalized, combined, version),
                )
                self._conn.commit()
                return offset

            return await self._run(_op)

    async def read_range(self, path: str, start: int, end: int | None = None) -> str:
        normalized = _normalize(path)

        def _op() -> str:
            cur = self._conn.execute("SELECT content FROM entries WHERE path = ?", (normalized,))
            row = cur.fetchone()
            if row is None:
                return ""
            data: bytes = row[0]
            stop = len(data) if end is None else min(end, len(data))
            if start >= stop:
                return ""
            return data[start:stop].decode("utf-8", errors="strict")

        return await self._run(_op)

    async def list_versions_under(self, prefix: str) -> dict[str, int]:
        """Return ``{path: version}`` for every key under ``prefix``.

        Used by :class:`_PollingChangeWatcher` to diff snapshots. The
        raw version numbers are per-write monotonic ints so any change
        under ``prefix`` bumps exactly one key's version, making the
        diff loop O(keys-under-prefix).
        """

        normalized = _normalize(prefix).rstrip("/")

        def _op() -> dict[str, int]:
            if normalized in ("", "/"):
                cur = self._conn.execute("SELECT path, version FROM entries")
            else:
                cur = self._conn.execute(
                    "SELECT path, version FROM entries WHERE path = ? OR path LIKE ?",
                    (normalized, normalized + "/%"),
                )
            return {row[0]: int(row[1]) for row in cur.fetchall()}

        return await self._run(_op)

    async def on_change(self, path: str, callback: ChangeCallback) -> ChangeSubscription:
        watcher = _PollingChangeWatcher(
            backend=self,
            prefix=path,
            callback=callback,
            interval_s=self._poll_interval_s,
        )
        await watcher.start()
        return watcher

    def close(self) -> None:
        """Close the SQLite connection. Idempotent."""

        with contextlib.suppress(Exception):
            self._conn.close()


# ---------------------------------------------------------------------------
# Redis implementation
# ---------------------------------------------------------------------------


class RedisKnowledgeStore:
    """Redis-backed :class:`KnowledgeStore`.

    Each virtual path becomes a Redis string under a configurable key
    prefix (default ``ag2:knowledge:``). A companion sorted set at
    ``{prefix}__index`` keeps a (path, version) ranking so
    :meth:`list_versions_under` returns a snapshot in one round trip
    and the polling watcher can diff without per-key ``GET``s.

    ``on_change`` uses :class:`_PollingChangeWatcher` (no Redis
    keyspace notifications — see §14 Phase 3b "Deferred"). The default
    poll interval is 500 ms and is configurable via ``poll_interval_s``.

    Dependencies: ``redis.asyncio`` is an optional install. The import
    is lazy so a project that never constructs a
    :class:`RedisKnowledgeStore` does not force the dependency on
    every install.
    """

    def __init__(
        self,
        url_or_client: Any,
        *,
        key_prefix: str = "ag2:knowledge",
        poll_interval_s: float = 0.5,
    ) -> None:
        if isinstance(url_or_client, str):
            try:
                from redis import asyncio as aioredis  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "RedisKnowledgeStore requires the 'redis' package. Install with: pip install redis"
                ) from exc
            self._client = aioredis.from_url(url_or_client, decode_responses=False)
            self._owns_client = True
        else:
            self._client = url_or_client
            self._owns_client = False
        self._key_prefix = key_prefix.rstrip(":")
        self._index_key = f"{self._key_prefix}:__index"
        self._poll_interval_s = poll_interval_s
        self._lock = asyncio.Lock()

    def _key(self, path: str) -> str:
        return f"{self._key_prefix}:{_normalize(path)}"

    async def _index_add(self, normalized: str, version: int) -> None:
        await self._client.zadd(self._index_key, {normalized: float(version)})

    async def _index_remove(self, *normalized_paths: str) -> None:
        if normalized_paths:
            await self._client.zrem(self._index_key, *normalized_paths)

    async def _index_scan(self) -> dict[str, int]:
        raw = await self._client.zrange(self._index_key, 0, -1, withscores=True)
        result: dict[str, int] = {}
        for entry in raw:
            path, score = entry
            if isinstance(path, bytes):
                path = path.decode("utf-8")
            result[path] = int(score)
        return result

    async def read(self, path: str) -> str | None:
        value = await self._client.get(self._key(path))
        if value is None:
            return None
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    async def write(self, path: str, content: str) -> None:
        normalized = _normalize(path)
        async with self._lock:
            version = int(await self._client.incr(f"{self._key_prefix}:__version_counter"))
            await self._client.set(self._key(path), content.encode("utf-8"))
            await self._index_add(normalized, version)

    async def list(self, path: str = "/") -> list[str]:
        prefix = _normalize(path).rstrip("/") + "/"
        snapshot = await self._index_scan()
        children: set[str] = set()
        for p in snapshot:
            if not p.startswith(prefix):
                continue
            remainder = p[len(prefix) :]
            if "/" in remainder:
                children.add(remainder.split("/")[0] + "/")
            else:
                children.add(remainder)
        return sorted(children)

    async def delete(self, path: str) -> None:
        normalized = _normalize(path)
        prefix = normalized.rstrip("/") + "/"

        async with self._lock:
            snapshot = await self._index_scan()
            to_delete = [p for p in snapshot if p == normalized or p.startswith(prefix)]
            if not to_delete:
                return
            keys = [self._key(p) for p in to_delete]
            await self._client.delete(*keys)
            await self._index_remove(*to_delete)

    async def exists(self, path: str) -> bool:
        normalized = _normalize(path)
        if await self._client.exists(self._key(normalized)):
            return True
        prefix = normalized.rstrip("/") + "/"
        snapshot = await self._index_scan()
        return any(p.startswith(prefix) for p in snapshot)

    async def append(self, path: str, content: str) -> int:
        normalized = _normalize(path)
        payload = content.encode("utf-8")
        async with self._lock:
            existing = await self._client.get(self._key(path))
            existing_bytes = (
                existing if isinstance(existing, bytes) else b"" if existing is None else str(existing).encode("utf-8")
            )
            offset = len(existing_bytes)
            combined = existing_bytes + payload
            version = int(await self._client.incr(f"{self._key_prefix}:__version_counter"))
            await self._client.set(self._key(path), combined)
            await self._index_add(normalized, version)
        return offset

    async def read_range(self, path: str, start: int, end: int | None = None) -> str:
        existing = await self._client.get(self._key(path))
        if existing is None:
            return ""
        data = existing if isinstance(existing, bytes) else str(existing).encode("utf-8")
        stop = len(data) if end is None else min(end, len(data))
        if start >= stop:
            return ""
        return data[start:stop].decode("utf-8", errors="strict")

    async def list_versions_under(self, prefix: str) -> dict[str, int]:
        snapshot = await self._index_scan()
        normalized = _normalize(prefix).rstrip("/")
        if normalized in ("", "/"):
            return snapshot
        scope = normalized + "/"
        return {p: v for p, v in snapshot.items() if p == normalized or p.startswith(scope)}

    async def on_change(self, path: str, callback: ChangeCallback) -> ChangeSubscription:
        watcher = _PollingChangeWatcher(
            backend=self,
            prefix=path,
            callback=callback,
            interval_s=self._poll_interval_s,
        )
        await watcher.start()
        return watcher

    async def close(self) -> None:
        """Close the underlying Redis client (if we own it)."""

        if self._owns_client:
            with contextlib.suppress(Exception):
                await self._client.aclose()


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

    async def append(self, path: str, content: str) -> int:
        acquired = await self._lock.acquire(f"store:write:{path}", ttl=30.0)
        if not acquired:
            raise RuntimeError(f"Failed to acquire append lock for {path}")
        try:
            return await self._store.append(path, content)
        finally:
            await self._lock.release(f"store:write:{path}")

    async def read_range(self, path: str, start: int, end: int | None = None) -> str:
        return await self._store.read_range(path, start, end)

    async def on_change(self, path: str, callback: ChangeCallback) -> ChangeSubscription:
        return await self._store.on_change(path, callback)


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

    async def load(self, stream_id: StreamId) -> list[BaseEvent]:
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
            events = await self._load_file(f"/log/{segment}")
            all_events.extend(events)

        # Read final log
        final = await self._load_file(f"/log/{stream_id}.jsonl")
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

    async def _load_file(self, path: str) -> list[BaseEvent]:

        from .events.lifecycle import UnknownEvent

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

            cls: type[BaseEvent] | None = import_event_class(event_type)

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
