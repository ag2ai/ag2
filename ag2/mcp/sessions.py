# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import UUID, uuid4

from ag2.history import MemoryStorage, Storage
from ag2.stream import MemoryStream

from .errors import UnknownConversationError

# Sentinel MCP session id for stdio: that transport carries no ``mcp-session-id``
# and serves a single client per process, so all handshake-era turns share one
# accumulating conversation. Withdrawn from the modern era, whose revision
# forbids establishing context from connection or process identity.
STDIO_SESSION = "stdio"

# Where a conversation handle travels back in a result's ``_meta``. Reverse-DNS
# as the ``_meta`` key rules require; it lives here rather than in the executor
# so reading it needs no ``ag2[mcp]`` install.
CONVERSATION_META_KEY = "ai.ag2/conversation"


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Tunables for multi-turn conversation history on :class:`MCPServer`.

    Attributes:
        max_sessions: LRU cap on conversations held at once. Every call naming
            none, with no MCP session to fall back on, mints one — so one-shot
            traffic occupies slots too; size for the call rate and set a ``ttl``.
        ttl: Idle expiry in seconds; ``None`` means no expiry.
        storage: History backend shared across conversations. The registry
            mapping a conversation's *name* to its history is per-process either
            way, so a shared backend does not make a handle portable.
    """

    max_sessions: int = 1024
    ttl: float | None = None
    storage: Storage | None = None


@dataclass(frozen=True, slots=True)
class ConversationBounds:
    """How long a conversation lives in a :class:`SessionStore`.

    Reported as data, not prose: the protocol requires the lifetime to appear in
    the tool description, and the descriptor is what words it.
    """

    max_conversations: int
    ttl: float | None = None


@dataclass(frozen=True, slots=True)
class Conversation:
    """One conversation as the serving path sees it: its stream and its handle.

    ``handle`` is ``None`` only for a stateless call, which has none to continue.
    """

    stream: MemoryStream
    handle: str | None = None


class _Entry:
    __slots__ = ("stream_id", "handle", "principal", "last", "turn_lock")

    def __init__(self, stream_id: UUID, handle: str, principal: str | None, last: float) -> None:
        self.stream_id = stream_id
        self.handle = handle
        # The principal that created this conversation, revalidated on every
        # handle lookup. ``None`` when no authentication is configured, in which
        # case the handle is the sole credential.
        self.principal = principal
        self.last = last
        # Serializes turns of one conversation at *this* tier, for the whole
        # scope rather than only the window ``Agent.ask`` is inside.
        #
        # Not the only lock in play, and the other one matters: every call gets a
        # fresh ``MemoryStream`` object but always under this entry's stable
        # ``stream_id``, and ``agent._get_stream_turn_lock`` keys on the id. So
        # releasing this lock does not make a conversation concurrent, and a
        # caller that releases it while a run is still inside ``ask`` (the
        # modern-era pause) must keep the next call away by other means.
        self.turn_lock = asyncio.Lock()


class SessionStore:
    """Bounded LRU registry mapping a conversation's key to a persistent stream.

    The key is a handle this store minted or, on the handshake era, the caller's
    MCP session id. It never adopts a key from a caller — :meth:`by_handle`
    resolves only its own — so nobody can name a conversation of their choosing
    and evict other callers' out of the bound.

    Each conversation has a stable stream id over a shared :class:`Storage`, and
    every serving method hands out a *fresh* :class:`MemoryStream` object reading
    that history back, so per-call progress subscribers never accumulate.
    """

    __slots__ = ("_storage", "_max", "_ttl", "_entries", "_by_handle", "_lock", "_clock", "on_evict")

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
        self._by_handle: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._clock = clock
        # Called with a handle as it is dropped, so anything else keyed by it
        # goes too. Assigned by the owner rather than taken in the constructor,
        # which would make the store know what a paused run is.
        self.on_evict: Callable[[str], None] | None = None

    @property
    def bounds(self) -> ConversationBounds:
        """The configured bound and idle expiry, for a client-facing description."""
        return ConversationBounds(max_conversations=self._max, ttl=self._ttl)

    @asynccontextmanager
    async def session(self, session_id: str, *, principal: str | None = None) -> AsyncGenerator[Conversation]:
        """Yield the conversation named by ``session_id``, holding its turn lock."""
        entry = await self._entry(session_id, principal=principal)
        async with self._held(entry) as conversation:
            yield conversation

    @asynccontextmanager
    async def fresh(self, *, principal: str | None = None) -> AsyncGenerator[Conversation]:
        """Mint a conversation under a new handle and yield it, holding its turn lock.

        The handle is a version-4 UUID: opaque and unguessable, as the protocol
        requires of a stateful handle.
        """
        handle = str(uuid4())
        entry = await self._entry(handle, principal=principal, handle=handle)
        async with self._held(entry) as conversation:
            yield conversation

    @asynccontextmanager
    async def by_handle(self, handle: str, *, principal: str | None = None) -> AsyncGenerator[Conversation]:
        """Yield the conversation ``handle`` names, holding its turn lock.

        Raises:
            UnknownConversationError: when no live conversation carries that
                handle, or when it was created by a different principal. Both
                read the same from outside, so the error does not disclose that
                an unreachable handle exists.
        """
        entry = await self._handle_entry(handle, principal)
        async with self._held(entry) as conversation:
            yield conversation

    async def touch(self, handle: str) -> None:
        """Mark ``handle``'s conversation as used just now, without holding it.

        For work that keeps a conversation alive without going through the
        serving methods: resuming a paused run continues a turn already inside
        one, and without this a long pause is evicted mid-question. Silent for an
        unknown handle — this refreshes an idle clock, and the callers that must
        refuse one raise where the handle is *resolved*.
        """
        async with self._lock:
            key = self._by_handle.get(handle)
            entry = self._entries.get(key) if key is not None else None
            if key is None or entry is None:
                return
            entry.last = self._clock()
            self._entries.move_to_end(key)

    async def acquire(self, session_id: str, *, principal: str | None = None) -> MemoryStream:
        """Return a stream carrying ``session_id``'s accumulated conversation.

        Does not hold the turn lock — prefer :meth:`session` on the serving path.
        ``principal`` is recorded when this call is what creates the
        conversation; ignoring it would mint one no authenticated caller could
        ever name.
        """
        entry = await self._entry(session_id, principal=principal)
        return MemoryStream(storage=self._storage, id=entry.stream_id)

    @asynccontextmanager
    async def _held(self, entry: _Entry) -> AsyncGenerator[Conversation]:
        """Yield ``entry``'s conversation while holding its turn lock."""
        async with entry.turn_lock:
            yield Conversation(stream=MemoryStream(storage=self._storage, id=entry.stream_id), handle=entry.handle)

    async def _entry(self, key: str, *, principal: str | None, handle: str | None = None) -> _Entry:
        async with self._lock:
            now = self._clock()
            await self._evict_expired(now)
            entry = self._entries.get(key)
            if entry is None:
                entry = _Entry(stream_id=uuid4(), handle=handle or str(uuid4()), principal=principal, last=now)
                self._entries[key] = entry
                self._by_handle[entry.handle] = key
            else:
                entry.last = now
                self._entries.move_to_end(key)
            await self._evict_overflow()
            return entry

    async def _handle_entry(self, handle: str, principal: str | None) -> _Entry:
        async with self._lock:
            now = self._clock()
            await self._evict_expired(now)
            key = self._by_handle.get(handle)
            entry = self._entries.get(key) if key is not None else None
            # Revalidated on every call, not at creation: a handle travels
            # through model context and logs, and a credential can be swapped or
            # revoked between two calls.
            if key is None or entry is None or entry.principal != principal:
                raise UnknownConversationError()
            entry.last = now
            self._entries.move_to_end(key)
            return entry

    async def _evict_expired(self, now: float) -> None:
        if self._ttl is None:
            return
        expired = [sid for sid, e in self._entries.items() if now - e.last > self._ttl]
        for sid in expired:
            await self._drop(sid)

    async def _evict_overflow(self) -> None:
        while len(self._entries) > self._max:
            await self._drop(next(iter(self._entries)))

    async def _drop(self, key: str) -> None:
        entry = self._entries.pop(key)
        self._by_handle.pop(entry.handle, None)
        if self.on_evict is not None:
            self.on_evict(entry.handle)
        await self._storage.drop_history(entry.stream_id)


__all__ = (
    "CONVERSATION_META_KEY",
    "STDIO_SESSION",
    "Conversation",
    "ConversationBounds",
    "SessionConfig",
    "SessionStore",
)
