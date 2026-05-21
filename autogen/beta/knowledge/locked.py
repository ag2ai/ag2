# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from .base import ChangeCallback, ChangeSubscription, KnowledgeStore


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
