# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import functools
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .base import ChangeCallback, ChangeSubscription, _normalize
from .polling import PollingChangeWatcher


class SqliteKnowledgeStore:
    """SQLite-backed :class:`KnowledgeStore`.

    Stores every virtual path as a row ``(path, content, version)`` in a
    single ``entries`` table. ``version`` is a monotonic integer bumped
    on every write or append and is the key the polling change watcher
    uses to detect mutations.

    SQLite does not have native change notifications, so ``on_change``
    uses :class:`PollingChangeWatcher` at a default 500ms interval.
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
        path: str | os.PathLike[str],
        *,
        poll_interval_s: float = 0.5,
    ) -> None:
        self._db_path: Path = Path(path)
        self._poll_interval_s = poll_interval_s
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._version_counter = 0

    def _ensure_connected(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                path TEXT PRIMARY KEY,
                content BLOB NOT NULL,
                version INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.commit()
        self._conn = conn
        cur = conn.execute("SELECT COALESCE(MAX(version), 0) FROM entries")
        row = cur.fetchone()
        self._version_counter = int(row[0]) if row else 0
        return conn

    def _next_version(self) -> int:
        self._version_counter += 1
        return self._version_counter

    async def _run(self, func: Callable[[], Any]) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func)

    def _sync_read(self, normalized: str) -> str | None:
        conn = self._ensure_connected()
        cur = conn.execute("SELECT content FROM entries WHERE path = ?", (normalized,))
        row = cur.fetchone()
        if row is None:
            return None
        return row[0].decode("utf-8")

    def _sync_write(self, normalized: str, payload: bytes, version: int) -> None:
        conn = self._ensure_connected()
        conn.execute(
            "INSERT OR REPLACE INTO entries (path, content, version) VALUES (?, ?, ?)",
            (normalized, payload, version),
        )
        conn.commit()

    def _sync_list(self, prefix: str) -> list[str]:
        conn = self._ensure_connected()
        cur = conn.execute(
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

    def _sync_delete(self, normalized: str, prefix: str) -> None:
        conn = self._ensure_connected()
        conn.execute("DELETE FROM entries WHERE path = ?", (normalized,))
        conn.execute("DELETE FROM entries WHERE path LIKE ?", (prefix + "%",))
        conn.commit()

    def _sync_exists(self, normalized: str, prefix: str) -> bool:
        conn = self._ensure_connected()
        cur = conn.execute("SELECT 1 FROM entries WHERE path = ? LIMIT 1", (normalized,))
        if cur.fetchone() is not None:
            return True
        cur = conn.execute(
            "SELECT 1 FROM entries WHERE path LIKE ? LIMIT 1",
            (prefix + "%",),
        )
        return cur.fetchone() is not None

    def _sync_append(self, normalized: str, payload: bytes, version: int) -> int:
        conn = self._ensure_connected()
        cur = conn.execute("SELECT content FROM entries WHERE path = ?", (normalized,))
        row = cur.fetchone()
        existing = row[0] if row else b""
        offset = len(existing)
        combined = existing + payload
        conn.execute(
            "INSERT OR REPLACE INTO entries (path, content, version) VALUES (?, ?, ?)",
            (normalized, combined, version),
        )
        conn.commit()
        return offset

    def _sync_read_range(self, normalized: str, start: int, end: int | None) -> str:
        conn = self._ensure_connected()
        cur = conn.execute("SELECT content FROM entries WHERE path = ?", (normalized,))
        row = cur.fetchone()
        if row is None:
            return ""
        data: bytes = row[0]
        stop = len(data) if end is None else min(end, len(data))
        if start >= stop:
            return ""
        return data[start:stop].decode("utf-8", errors="strict")

    def _sync_list_versions(self, normalized: str) -> dict[str, int]:
        conn = self._ensure_connected()
        if normalized in ("", "/"):
            cur = conn.execute("SELECT path, version FROM entries")
        else:
            cur = conn.execute(
                "SELECT path, version FROM entries WHERE path = ? OR path LIKE ?",
                (normalized, normalized + "/%"),
            )
        return {row[0]: int(row[1]) for row in cur.fetchall()}

    async def read(self, path: str) -> str | None:
        normalized = _normalize(path)
        return await self._run(functools.partial(self._sync_read, normalized))

    async def write(self, path: str, content: str) -> None:
        normalized = _normalize(path)
        payload = content.encode("utf-8")
        async with self._lock:
            version = self._next_version()
            await self._run(functools.partial(self._sync_write, normalized, payload, version))

    async def list(self, path: str = "/") -> list[str]:
        prefix = _normalize(path).rstrip("/") + "/"
        return await self._run(functools.partial(self._sync_list, prefix))

    async def delete(self, path: str) -> None:
        normalized = _normalize(path)
        prefix = normalized.rstrip("/") + "/"
        async with self._lock:
            await self._run(functools.partial(self._sync_delete, normalized, prefix))

    async def exists(self, path: str) -> bool:
        normalized = _normalize(path)
        prefix = normalized.rstrip("/") + "/"
        return await self._run(functools.partial(self._sync_exists, normalized, prefix))

    async def append(self, path: str, content: str) -> int:
        normalized = _normalize(path)
        payload = content.encode("utf-8")
        async with self._lock:
            version = self._next_version()
            return await self._run(functools.partial(self._sync_append, normalized, payload, version))

    async def read_range(self, path: str, start: int, end: int | None = None) -> str:
        normalized = _normalize(path)
        return await self._run(functools.partial(self._sync_read_range, normalized, start, end))

    async def list_versions_under(self, prefix: str) -> dict[str, int]:
        """Return ``{path: version}`` for every key under ``prefix``.

        Used by :class:`PollingChangeWatcher` to diff snapshots. The
        raw version numbers are per-write monotonic ints so any change
        under ``prefix`` bumps exactly one key's version, making the
        diff loop O(keys-under-prefix).
        """
        normalized = _normalize(prefix).rstrip("/")
        return await self._run(functools.partial(self._sync_list_versions, normalized))

    async def on_change(self, path: str, callback: ChangeCallback) -> ChangeSubscription:
        watcher = PollingChangeWatcher(
            backend=self,
            prefix=path,
            callback=callback,
            interval_s=self._poll_interval_s,
        )
        await watcher.start()
        return watcher

    def close(self) -> None:
        """Close the SQLite connection. Idempotent."""
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
