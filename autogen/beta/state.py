# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""StateStore — persistent key-value state for operational concerns.

StateStore provides ephemeral, TTL-capable key-value storage for Hub
and plugin coordination state. It is distinct from KnowledgeStore,
which provides filesystem semantics for persistent actor knowledge.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable


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
