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

# Where a conversation handle travels back in a result's ``_meta``, for clients
# threading it programmatically. Reverse-DNS from the project's domain, as the
# ``_meta`` key rules require; prefixes whose second label is ``mcp`` or
# ``modelcontextprotocol`` are reserved and are not used here. It lives beside
# the conversation it names rather than in the executor so that reading it needs
# no ``ag2[mcp]`` install.
CONVERSATION_META_KEY = "ai.ag2/conversation"


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Tunables for multi-turn conversation history on :class:`MCPServer`.

    Each conversation gets its own history that accumulates across ``tools/call``
    invocations, whichever name it goes by — a *conversation handle* the caller
    threads through, or the MCP session a handshake-era caller already has. The
    registry is bounded so a long-lived server cannot leak memory:

    * ``max_sessions`` — LRU cap; the least-recently-used conversation's history
      is dropped once the cap is exceeded. Any call that names no conversation
      and has no MCP session to fall back on gets a fresh one — every modern-era
      call, and every handshake-era call on a ``stateless=True`` transport — so
      one-shot traffic occupies slots too; size the cap for the call rate and set
      a ``ttl`` alongside it.
    * ``ttl`` — optional idle expiry in seconds; a conversation untouched for
      longer than this has its history dropped on the next access (``None`` = no
      expiry).
    * ``storage`` — pluggable history backend shared across conversations (each
      keyed by its own stream id). Defaults to an in-memory :class:`MemoryStorage`;
      pass e.g. a Redis-backed :class:`Storage` to keep histories out of process
      memory. Note that the registry mapping a conversation's *name* to its
      history is per-process either way, so a shared backend does not on its own
      make a handle usable against another replica.
    """

    max_sessions: int = 1024
    ttl: float | None = None
    storage: Storage | None = None


@dataclass(frozen=True, slots=True)
class ConversationBounds:
    """How long a conversation lives in a :class:`SessionStore`.

    The store reports its configured bound and idle expiry as data; the tool
    descriptor words them for a client. The protocol requires a stateful handle's
    lifetime to be stated in the tool description, and a store that returned the
    sentence itself would put client-facing prose behind the registry.
    """

    max_conversations: int
    ttl: float | None = None


@dataclass(frozen=True, slots=True)
class Conversation:
    """One conversation as the serving path sees it: its stream and its handle.

    ``handle`` is the opaque, server-minted name a caller presents to continue
    this conversation. It is ``None`` only for a stateless call, which has no
    conversation to continue.
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
        # Serializes turns of one conversation: a fresh MemoryStream is handed
        # out per call, so the agent's per-stream turn lock can't serialize
        # same-conversation concurrency — this entry-scoped lock does.
        self.turn_lock = asyncio.Lock()


class SessionStore:
    """Bounded LRU registry mapping a conversation's key to a persistent stream.

    The key is an opaque string: a *conversation handle* this store minted, or —
    in the handshake era, where the protocol has one — the caller's MCP session
    id. The store never derives a key itself, and never adopts one from a caller:
    :meth:`by_handle` resolves only handles it minted, so a caller cannot name a
    conversation of their choosing and evict other callers' out of the bound.

    Each conversation is bound to a stable :class:`~uuid.UUID` stream id over a
    shared :class:`Storage`; the serving methods return a *fresh*
    :class:`MemoryStream` object on every call (so per-call progress subscribers
    never accumulate) that reads prior turns back from storage — mirroring the
    subagents' ``persistent_stream`` pattern. Eviction (LRU overflow + idle TTL)
    drops the evicted conversation's stored history so memory stays bounded.
    """

    __slots__ = ("_storage", "_max", "_ttl", "_entries", "_by_handle", "_lock", "_clock")

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

    @property
    def bounds(self) -> ConversationBounds:
        """The configured bound and idle expiry, for a client-facing description.

        Reported as data rather than prose so the sentence the tool advertises
        cannot drift from the configuration behind it.
        """
        return ConversationBounds(max_conversations=self._max, ttl=self._ttl)

    @asynccontextmanager
    async def session(self, session_id: str, *, principal: str | None = None) -> AsyncGenerator[Conversation]:
        """Yield the conversation named by ``session_id``, holding its turn lock.

        Holding the lock for the duration of the turn serializes concurrent calls
        on the same conversation, so their accumulated history can't interleave.
        """
        entry = await self._entry(session_id, principal=principal)
        async with self._held(entry) as conversation:
            yield conversation

    @asynccontextmanager
    async def fresh(self, *, principal: str | None = None) -> AsyncGenerator[Conversation]:
        """Mint a conversation under a new handle and yield it, holding its turn lock.

        The handle is a version-4 UUID: opaque and unguessable, as the protocol
        requires of a stateful handle, and incidentally ASCII- and header-safe.
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

    async def acquire(self, session_id: str, *, principal: str | None = None) -> MemoryStream:
        """Return a stream carrying ``session_id``'s accumulated conversation.

        Does not hold the turn lock — prefer :meth:`session` on the serving path.

        ``principal`` is recorded when this call is what creates the conversation,
        exactly as :meth:`session` records it, so the handle minted alongside it
        stays reachable through :meth:`by_handle` by the same caller. Defaulting
        it to ``None`` and ignoring the argument would mint a conversation no
        authenticated caller could ever name.
        """
        entry = await self._entry(session_id, principal=principal)
        return MemoryStream(storage=self._storage, id=entry.stream_id)

    @asynccontextmanager
    async def _held(self, entry: _Entry) -> AsyncGenerator[Conversation]:
        """Yield ``entry``'s conversation while holding its turn lock.

        Every serving method ends here: a fresh :class:`MemoryStream` object per
        call (so per-call progress subscribers never accumulate) reading prior
        turns back from storage, handed out under the entry's turn lock.
        """
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
            # Authorization is revalidated here, on every call rather than at
            # creation, because a handle travels through model context and logs
            # and a credential can be swapped or revoked between two calls.
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
        await self._storage.drop_history(entry.stream_id)


__all__ = (
    "CONVERSATION_META_KEY",
    "STDIO_SESSION",
    "Conversation",
    "ConversationBounds",
    "SessionConfig",
    "SessionStore",
)
