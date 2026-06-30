# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import UUID, uuid4

from ag2.history import MemoryStorage, Storage
from ag2.stream import MemoryStream

# Sentinel session id for stdio: that transport carries no ``mcp-session-id`` and
# serves a single client per process, so all turns share one accumulating stream.
STDIO_SESSION = "stdio"


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Tunables for session-keyed multi-turn history on :class:`MCPServer`.

    Each MCP session (keyed by the transport's ``mcp-session-id``) gets its own
    conversation history that accumulates across ``tools/call`` invocations. The
    registry is bounded so a long-lived server cannot leak memory:

    * ``max_sessions`` — LRU cap; the least-recently-used session's history is
      dropped once the cap is exceeded.
    * ``ttl`` — optional idle expiry in seconds; a session untouched for longer
      than this has its history dropped on the next access (``None`` = no expiry).
    * ``storage`` — pluggable history backend shared across sessions (each keyed
      by its own stream id). Defaults to an in-memory :class:`MemoryStorage`;
      pass e.g. a Redis-backed :class:`Storage` for cross-replica continuity.
    """

    max_sessions: int = 1024
    ttl: float | None = None
    storage: Storage | None = None


class _Entry:
    __slots__ = ("stream_id", "last", "turn_lock")

    def __init__(self, stream_id: UUID, last: float) -> None:
        self.stream_id = stream_id
        self.last = last
        # Serializes turns of one session: a fresh MemoryStream is handed out per
        # call, so the agent's per-stream turn lock can't serialize same-session
        # concurrency — this entry-scoped lock does.
        self.turn_lock = asyncio.Lock()


class SessionStore:
    """Bounded LRU registry mapping an ``mcp-session-id`` to a persistent stream.

    Each session is bound to a stable :class:`~uuid.UUID` stream id over a shared
    :class:`Storage`; :meth:`acquire` returns a *fresh* :class:`MemoryStream`
    object on every call (so per-call progress subscribers never accumulate) that
    reads prior turns back from storage — mirroring the subagents'
    ``persistent_stream`` pattern. Eviction (LRU overflow + idle TTL) drops the
    evicted session's stored history so memory stays bounded.
    """

    __slots__ = ("_storage", "_max", "_ttl", "_entries", "_lock", "_clock")

    def __init__(
        self,
        *,
        max_sessions: int = 1024,
        ttl: float | None = None,
        storage: Storage | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_sessions < 1:
            raise ValueError(f"max_sessions must be >= 1, got {max_sessions}.")
        if ttl is not None and ttl <= 0:
            raise ValueError(f"ttl must be > 0 when set, got {ttl}.")
        self._storage = storage or MemoryStorage()
        self._max = max_sessions
        self._ttl = ttl
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._clock = clock

    @asynccontextmanager
    async def session(self, session_id: str) -> AsyncIterator[MemoryStream]:
        """Yield ``session_id``'s stream while holding its per-session turn lock.

        Holding the lock for the duration of the turn serializes concurrent calls
        on the same session, so their accumulated history can't interleave.
        """
        entry = await self._entry(session_id)
        async with entry.turn_lock:
            yield MemoryStream(storage=self._storage, id=entry.stream_id)

    async def acquire(self, session_id: str) -> MemoryStream:
        """Return a stream carrying ``session_id``'s accumulated history.

        Does not hold the turn lock — prefer :meth:`session` on the serving path.
        """
        entry = await self._entry(session_id)
        return MemoryStream(storage=self._storage, id=entry.stream_id)

    async def _entry(self, session_id: str) -> _Entry:
        async with self._lock:
            now = self._clock()
            await self._evict_expired(now)
            entry = self._entries.get(session_id)
            if entry is None:
                entry = _Entry(stream_id=uuid4(), last=now)
                self._entries[session_id] = entry
            else:
                entry.last = now
                self._entries.move_to_end(session_id)
            await self._evict_overflow()
            return entry

    async def _evict_expired(self, now: float) -> None:
        if self._ttl is None:
            return
        expired = [sid for sid, e in self._entries.items() if now - e.last > self._ttl]
        for sid in expired:
            entry = self._entries.pop(sid)
            await self._storage.drop_history(entry.stream_id)

    async def _evict_overflow(self) -> None:
        while len(self._entries) > self._max:
            _sid, entry = self._entries.popitem(last=False)
            await self._storage.drop_history(entry.stream_id)


__all__ = (
    "STDIO_SESSION",
    "SessionConfig",
    "SessionStore",
)
