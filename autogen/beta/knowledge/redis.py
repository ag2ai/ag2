# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
from typing import Any

from .base import ChangeCallback, ChangeSubscription, _normalize
from .polling import PollingChangeWatcher

try:
    from redis import asyncio as _aioredis  # type: ignore[import-not-found]
except ImportError:
    _aioredis = None  # type: ignore[assignment]


class RedisKnowledgeStore:
    """Redis-backed :class:`KnowledgeStore`.

    Each virtual path becomes a Redis string under a configurable key
    prefix (default ``ag2:knowledge:``). A companion sorted set at
    ``{prefix}__index`` keeps a (path, version) ranking so
    :meth:`list_versions_under` returns a snapshot in one round trip
    and the polling watcher can diff without per-key ``GET``s.

    ``on_change`` uses :class:`PollingChangeWatcher` (no Redis
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
            if _aioredis is None:  # pragma: no cover
                raise ImportError("RedisKnowledgeStore requires the 'redis' package. Install with: pip install redis")
            self._client = _aioredis.from_url(url_or_client, decode_responses=False)
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
        watcher = PollingChangeWatcher(
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
