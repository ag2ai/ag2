# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Infrastructure protocols — backend contracts for production deployment.

Each protocol has an in-memory default implementation. AG2 Cloud (or any
deployment) swaps in persistent, distributed backends — same application
code, different infrastructure.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# ActorInfo — metadata about a registered actor
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ActorInfo:
    """Metadata about a registered actor in the network."""

    name: str
    capabilities: list[str] = field(default_factory=list)
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# StateStore — persistent key-value state
# ---------------------------------------------------------------------------


@runtime_checkable
class StateStore(Protocol):
    """Persistent key-value state for actors and plugins.

    Enables crash recovery, checkpointing, and distributed state.
    """

    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl: float | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...


class MemoryStateStore:
    """In-memory StateStore. Development default."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}

    async def get(self, key: str) -> Any | None:
        if key in self._expiry and time.monotonic() > self._expiry[key]:
            del self._store[key]
            del self._expiry[key]
            return None
        return self._store.get(key)

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        self._store[key] = value
        if ttl is not None:
            self._expiry[key] = time.monotonic() + ttl
        elif key in self._expiry:
            del self._expiry[key]

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._expiry.pop(key, None)

    async def exists(self, key: str) -> bool:
        if key in self._expiry and time.monotonic() > self._expiry[key]:
            del self._store[key]
            del self._expiry[key]
            return False
        return key in self._store


# ---------------------------------------------------------------------------
# Cache — result caching for reducing LLM calls and tool invocations
# ---------------------------------------------------------------------------


@runtime_checkable
class Cache(Protocol):
    """Result caching for reducing LLM calls and tool invocations."""

    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl: float | None = None) -> None: ...
    async def invalidate(self, pattern: str) -> None: ...


class MemoryCache:
    """In-memory Cache with TTL. Development default."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[Any, float | None]] = {}  # key -> (value, expiry_time)

    async def get(self, key: str) -> Any | None:
        if key not in self._cache:
            return None
        value, expiry = self._cache[key]
        if expiry is not None and time.monotonic() > expiry:
            del self._cache[key]
            return None
        return value

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        expiry = (time.monotonic() + ttl) if ttl is not None else None
        self._cache[key] = (value, expiry)

    async def invalidate(self, pattern: str) -> None:
        """Invalidate cache entries matching a prefix pattern.

        Pattern matching: ``"user:*"`` matches ``"user:123"``, ``"user:abc"``.
        A pattern without ``*`` matches exactly.
        """
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            keys = [k for k in self._cache if k.startswith(prefix)]
        else:
            keys = [pattern] if pattern in self._cache else []
        for key in keys:
            del self._cache[key]


# ---------------------------------------------------------------------------
# Lock — distributed coordination for exclusive access
# ---------------------------------------------------------------------------


@runtime_checkable
class Lock(Protocol):
    """Distributed coordination for exclusive access."""

    async def acquire(self, key: str, ttl: float = 30.0) -> bool: ...
    async def release(self, key: str) -> None: ...


class LocalLock:
    """In-memory Lock. Development default (single-process only).

    TTL is enforced via asyncio — a background task releases the lock
    after the TTL expires, preventing deadlocks from crashed holders.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._ttl_tasks: dict[str, asyncio.Task[None]] = {}
        self._gate = asyncio.Lock()  # Serializes acquire attempts

    async def acquire(self, key: str, ttl: float = 30.0) -> bool:
        async with self._gate:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            lock = self._locks[key]
            if lock.locked():
                return False
            # Gate ensures no other coroutine can interleave between
            # the locked() check and acquire() — eliminates TOCTOU race.
            await lock.acquire()
        # Schedule auto-release after TTL
        self._cancel_ttl(key)
        self._ttl_tasks[key] = asyncio.ensure_future(self._auto_release(key, ttl))
        return True

    async def release(self, key: str) -> None:
        self._cancel_ttl(key)
        if key in self._locks and self._locks[key].locked():
            self._locks[key].release()

    def _cancel_ttl(self, key: str) -> None:
        task = self._ttl_tasks.pop(key, None)
        if task and not task.done():
            task.cancel()

    async def _auto_release(self, key: str, ttl: float) -> None:
        await asyncio.sleep(ttl)
        if key in self._locks and self._locks[key].locked():
            self._locks[key].release()

    @asynccontextmanager
    async def held(self, key: str, ttl: float = 30.0):  # type: ignore[override]
        """Context manager for exclusive access with TTL."""
        acquired = await self.acquire(key, ttl=ttl)
        if not acquired:
            raise RuntimeError(f"Failed to acquire lock: {key!r}")
        try:
            yield
        finally:
            await self.release(key)


# ---------------------------------------------------------------------------
# Registry — service registry for actor discovery
# ---------------------------------------------------------------------------


@runtime_checkable
class Registry(Protocol):
    """Service registry for actor discovery. Decoupled from Hub for swappability."""

    async def register(self, name: str, info: ActorInfo) -> None: ...
    async def unregister(self, name: str) -> None: ...
    async def discover(self, capability: str = "") -> list[ActorInfo]: ...
    async def heartbeat(self, name: str) -> None: ...


class LocalRegistry:
    """In-memory Registry. Development default."""

    def __init__(self) -> None:
        self._actors: dict[str, ActorInfo] = {}

    async def register(self, name: str, info: ActorInfo) -> None:
        self._actors[name] = info

    async def unregister(self, name: str) -> None:
        self._actors.pop(name, None)

    async def discover(self, capability: str = "") -> list[ActorInfo]:
        if not capability:
            return list(self._actors.values())
        return [info for info in self._actors.values() if capability in info.capabilities]

    async def heartbeat(self, name: str) -> None:
        if name in self._actors:
            self._actors[name].last_heartbeat = time.monotonic()
