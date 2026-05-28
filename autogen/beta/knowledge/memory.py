# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib

from .base import ChangeCallback, ChangeSubscription, _normalize


class _MemoryChangeSubscription:
    """Subscription handle for :class:`MemoryKnowledgeStore`."""

    def __init__(self, subscribers: dict[str, list[ChangeCallback]], key: str, callback: ChangeCallback) -> None:
        self._subscribers = subscribers
        self._key = key
        self._callback = callback

    async def close(self) -> None:
        bucket = self._subscribers.get(self._key)
        if not bucket:
            return
        with contextlib.suppress(ValueError):
            bucket.remove(self._callback)
        if not bucket:
            self._subscribers.pop(self._key, None)


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
        return _MemoryChangeSubscription(self._subscribers, normalized, callback)

    async def _notify(self, changed_path: str) -> None:
        for subscribed_path, callbacks in list(self._subscribers.items()):
            if changed_path == subscribed_path or changed_path.startswith(subscribed_path.rstrip("/") + "/"):
                for callback in list(callbacks):
                    await callback(changed_path)
