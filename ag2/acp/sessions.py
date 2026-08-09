# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Agent-side ACP session state.

An ACP session is one independent conversation. Unlike MCP — where the session id
is supplied by the transport — ACP mints the id explicitly at ``session/new``, so
:class:`SessionStore` generates it and hands it back to the Client.

Each session owns a slice of history over a shared :class:`~ag2.history.Storage`,
keyed by its own stream id, plus the run-scoped state a turn needs: the task
currently executing (what ``session/cancel`` kills) and the queue of prompts
waiting behind it.

Not to be confused with the *client*-side :class:`~ag2.acp.session.ACPSession`,
which owns a subprocess connection to an external CLI agent.
"""

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from ag2.history import MemoryStorage, Storage
from ag2.stream import MemoryStream

from .types import McpServer

logger = logging.getLogger(__name__)

__all__ = (
    "AgentSession",
    "SessionConfig",
    "SessionLimitError",
    "SessionStore",
    "UnknownSessionError",
)


class UnknownSessionError(KeyError):
    """Raised when a Client names a session id the store has never issued.

    The ACP agent turns this into a protocol error rather than letting it
    surface as an unhandled exception.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self.session_id = session_id

    def __str__(self) -> str:
        return f"Unknown ACP session id: {self.session_id!r}"


class SessionLimitError(RuntimeError):
    """Raised when a new session cannot be admitted without breaking the cap.

    Only fires when the registry is full *and* every session in it is busy, so
    there is nothing safe to evict. Refusing admission is what keeps
    ``max_sessions`` a real bound: the alternative — letting the registry grow
    while turns are in flight — is no bound at all, because a Client can hold
    every session busy indefinitely.
    """

    def __init__(self, max_sessions: int) -> None:
        super().__init__(max_sessions)
        self.max_sessions = max_sessions

    def __str__(self) -> str:
        return (
            f"At the {self.max_sessions}-session limit and every session is busy. "
            "Retry once a turn finishes, or close a session."
        )


class ConnectionOverloadedError(RuntimeError):
    """Raised when a connection already has ``max_active_prompts`` in flight.

    The per-session bounds cannot see each other, so this is what stops one
    Client spreading its load across many sessions and leaving the connection
    itself unbounded.
    """

    def __init__(self, max_active_prompts: int) -> None:
        super().__init__(max_active_prompts)
        self.max_active_prompts = max_active_prompts

    def __str__(self) -> str:
        return f"This connection already has {self.max_active_prompts} prompts in flight. Retry once one finishes."


class SessionBusyError(RuntimeError):
    """Raised when a session's prompt queue is already at ``max_queued``.

    Prompts on a busy session normally *wait* (see :meth:`AgentSession.turn`);
    this fires only on overflow, so a chatty Client cannot grow memory without
    bound.
    """

    def __init__(self, session_id: str, max_queued: int) -> None:
        super().__init__(session_id, max_queued)
        self.session_id = session_id
        self.max_queued = max_queued

    def __str__(self) -> str:
        return f"ACP session {self.session_id!r} already has {self.max_queued} prompts queued."


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Tunables for :class:`SessionStore`.

    Mirrors :class:`ag2.mcp.sessions.SessionConfig` — the two AG2 protocol
    adapters should bound their session registries the same way — plus
    ``max_queued``, which ACP needs because it lets a Client fire a second
    ``session/prompt`` while the first is still running.

    * ``max_sessions`` — LRU cap; the least-recently-used session is dropped
      (and its history deleted) once the cap is exceeded.
    * ``ttl`` — optional idle expiry in seconds; a session untouched for longer
      is dropped on the next access (``None`` = no expiry).
    * ``storage`` — pluggable history backend shared across sessions, each keyed
      by its own stream id. Defaults to an in-memory :class:`MemoryStorage`.
    * ``max_queued`` — how many prompts may wait behind the running turn on one
      session before further prompts are rejected.

    The two bounds above are *per session*, which on their own leave the whole
    connection unbounded: a Client may hold ``max_sessions`` conversations and
    start a paid turn in every one of them at once. These cap the connection as a
    whole:

    * ``max_concurrent_turns`` — how many turns may run at the same time across
      all sessions. Prompts past this wait for a slot rather than being refused,
      since a burst across separate conversations is normal traffic, not abuse.
      This is the knob that bounds spend and provider rate-limit pressure.
    * ``max_active_prompts`` — how many prompts may be admitted at once across
      all sessions, running and waiting together. Past this, prompts are refused,
      so an unbounded queue of parked request handlers cannot build up.
    """

    max_sessions: int = 1024
    ttl: float | None = None
    storage: Storage | None = None
    max_queued: int = 8
    max_concurrent_turns: int = 8
    max_active_prompts: int = 64


@dataclass(slots=True)
class AgentSession:
    """One ACP session: its identity, its history slice, and its live turn state.

    ``stream_id`` names this session's slice of the shared
    :class:`~ag2.history.Storage`; :attr:`stream` is the one stream object built
    against it and kept for the session's lifetime, so an inbox message or a
    background task outliving its turn is still reachable from the next one (see
    :meth:`SessionStore.stream`).

    ``cwd``, ``additional_directories``, ``mcp_servers`` and ``meta`` are
    Client-provided *context*, captured verbatim and acted on by nothing here.
    Per the ACP integration design, a path named by a Client is not authority to
    reach it; an embedding application decides what, if anything, to honour.
    """

    session_id: str
    stream_id: UUID
    cwd: str = "."
    additional_directories: list[str] = field(default_factory=list)
    mcp_servers: list[McpServer] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    max_queued: int = 8

    # This conversation's context variables. Seeded from the agent's defaults at
    # ``session/new`` and then owned by the session: the same dict is handed to
    # every turn, so a tool's writes survive into the next prompt the way they do
    # across ``AgentReply.ask``. Seeded by *value*, so one conversation's writes
    # are invisible to another — see :func:`ag2.acp.executor.isolate_variables`.
    variables: dict[Any, Any] = field(default_factory=dict)

    # Clock the session stamps ``last_active`` from; injected so tests can drive
    # expiry deterministically.
    clock: Callable[[], float] = field(default=time.monotonic, compare=False, repr=False)
    # When this session was last *used* — set on every access and, crucially,
    # again when a turn finishes. Idle expiry measures from here, so a turn that
    # runs longer than the TTL cannot expire the session it is running in.
    last_active: float = field(default=0.0, compare=False, repr=False)

    # This session's stream, created on first use and kept for the session's
    # lifetime so its inbox and background tasks survive between turns. Run
    # state, not identity.
    stream: "MemoryStream | None" = field(default=None, compare=False, repr=False)

    # Run-scoped, not identity. The task currently driving a turn — the handle
    # ``session/cancel`` acts on. AG2 has no turn-level cancel API, so cancelling
    # the task is the mechanism (the same one ``ag2/a2a/executor.py`` relies on).
    turn_task: "asyncio.Task[Any] | None" = field(default=None, compare=False, repr=False)

    # Serializes turns on this session so their histories cannot interleave.
    _turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock, compare=False, repr=False)
    # Prompts admitted but not yet running — the bound ``max_queued`` applies to.
    # Counted, not stored: the waiters are coroutines parked on ``_turn_lock``.
    _queued: int = field(default=0, compare=False, repr=False)
    # Prompts admitted and not yet finished (waiting *plus* the one running).
    # Drives the cancel latch: it clears only once the whole batch has drained.
    _inflight: int = field(default=0, compare=False, repr=False)
    # Set while a cancel is in force, so prompts already queued behind the
    # cancelled turn abort instead of running. Cleared once the queue drains.
    _cancelled: bool = field(default=False, compare=False, repr=False)

    @property
    def queued(self) -> int:
        """Prompts currently waiting behind the running turn."""
        return self._queued

    @property
    def inflight(self) -> int:
        """Prompts admitted and not yet finished, including the running one."""
        return self._inflight

    @property
    def is_idle(self) -> bool:
        """Whether this session has no work running or waiting.

        Only an idle session may be evicted; see
        :meth:`SessionStore._evict_overflow`.
        """
        task = self.turn_task
        return self._inflight == 0 and (task is None or task.done())

    @property
    def cancelled(self) -> bool:
        """Whether a ``session/cancel`` is still draining this session's queue."""
        return self._cancelled

    async def cancel(self) -> None:
        """Cancel the running turn and every prompt queued behind it.

        ``session/cancel`` names a *session*, so its blast radius is this whole
        session and nothing else — other sessions are untouched. Cancelling the
        running turn only, and letting the queue roll on, would make a stop
        button start the next thing the user typed.

        Events already emitted are left alone: cancelling the task does not
        touch the stream, so completed work stays in history.
        """
        self._cancelled = self._inflight > 0 or self.turn_task is not None
        task = self.turn_task
        if task is not None and not task.done():
            task.cancel()

    @asynccontextmanager
    async def recovery(self) -> AsyncIterator[None]:
        """Hold the turn lock for post-cancellation repair.

        Repairing a cancelled turn is a read-modify-write over the session's
        whole history, so it has to exclude the next prompt: otherwise a new turn
        can start on a transcript that still has an unanswered tool call, or
        append events that the repair then overwrites with its stale snapshot.

        Unlike :meth:`turn` this ignores the queue bound and the cancel latch —
        recovery is not a prompt, and it is precisely what has to happen *while*
        the latch is still set.
        """
        async with self._turn_lock:
            yield

    @asynccontextmanager
    async def turn(self) -> AsyncIterator[None]:
        """Hold this session's turn lock for the duration of one prompt.

        Concurrent prompts on one session queue rather than interleave —
        matching :meth:`ag2.mcp.sessions.SessionStore.session`. ``max_queued``
        bounds the prompts *waiting*, not counting the one running, so a session
        admits ``max_queued + 1`` at a time; past that,
        :class:`SessionBusyError`. A prompt that was still waiting when
        :meth:`cancel` ran raises :class:`asyncio.CancelledError` without ever
        reaching the agent.
        """
        if self._queued >= self.max_queued:
            raise SessionBusyError(self.session_id, self.max_queued)
        # Only count as queued while actually waiting: once this prompt holds the
        # lock it is the running turn, and the queue behind it is free again.
        waiting = self._turn_lock.locked()
        self._queued += int(waiting)
        self._inflight += 1
        try:
            async with self._turn_lock:
                if waiting:
                    self._queued -= 1
                    waiting = False
                if self._cancelled:
                    raise asyncio.CancelledError
                yield
        finally:
            self._queued -= int(waiting)
            self._inflight -= 1
            # Restamp on the way out. Idle expiry is measured from the last time
            # the session was *used*, and a turn only just finished — without
            # this, a turn that outlives the TTL would leave behind a session
            # already stale enough for the next sweep to delete.
            self.last_active = self.clock()
            # Last one out clears the cancel latch, so the session is usable
            # again for prompts that arrive after the cancelled batch drained.
            if self._inflight == 0:
                self._cancelled = False
                self.turn_task = None


async def _stop_background(session: AgentSession) -> None:
    """Cancel the background work a session's stream is still carrying.

    ``spawn_background`` hands a tool or subagent a task that deliberately
    outlives the turn that started it. That is fine while the session is alive;
    once it is being torn down, the Client that asked for the work is gone and
    anything the task still does reaches the outside world unattributed.
    """
    stream = session.stream
    if stream is None:
        return
    tasks = [task for task in getattr(stream, "_background_tasks", ()) if not task.done()]
    for task in tasks:
        task.cancel()
    for task in tasks:
        with suppress(asyncio.CancelledError, Exception):
            await task


class SessionStore:
    """Bounded LRU registry of :class:`AgentSession`, keyed by ACP session id.

    Adapted from :class:`ag2.mcp.sessions.SessionStore`: same LRU-plus-TTL
    bounding and the same "fresh :class:`MemoryStream` per turn over a stable
    stream id" arrangement, with the id minted here (ACP's ``session/new``
    returns it) instead of arriving from the transport.
    """

    __slots__ = (
        "_storage",
        "_max",
        "_ttl",
        "_max_queued",
        "_sessions",
        "_lock",
        "_clock",
        "_turn_slots",
        "_max_active_prompts",
        "_active_prompts",
    )

    def __init__(
        self,
        *,
        max_sessions: int = 1024,
        ttl: float | None = None,
        storage: Storage | None = None,
        max_queued: int = 8,
        max_concurrent_turns: int = 8,
        max_active_prompts: int = 64,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_sessions < 1:
            raise ValueError(f"max_sessions must be >= 1, got {max_sessions}.")
        if ttl is not None and ttl <= 0:
            raise ValueError(f"ttl must be > 0 when set, got {ttl}.")
        if max_queued < 1:
            raise ValueError(f"max_queued must be >= 1, got {max_queued}.")
        if max_concurrent_turns < 1:
            raise ValueError(f"max_concurrent_turns must be >= 1, got {max_concurrent_turns}.")
        if max_active_prompts < max_concurrent_turns:
            raise ValueError(
                f"max_active_prompts ({max_active_prompts}) must be >= max_concurrent_turns "
                f"({max_concurrent_turns}); otherwise no turn could ever reach a slot."
            )
        self._storage = storage or MemoryStorage()
        self._max = max_sessions
        self._ttl = ttl
        self._max_queued = max_queued
        # OrderedDict order is the LRU order; each session carries its own
        # ``last_active`` stamp, which is what idle expiry measures.
        self._sessions: OrderedDict[str, AgentSession] = OrderedDict()
        self._lock = asyncio.Lock()
        self._clock = clock
        self._turn_slots = asyncio.Semaphore(max_concurrent_turns)
        self._max_active_prompts = max_active_prompts
        self._active_prompts = 0

    @classmethod
    def from_config(cls, config: SessionConfig) -> "SessionStore":
        return cls(
            max_sessions=config.max_sessions,
            ttl=config.ttl,
            storage=config.storage,
            max_queued=config.max_queued,
            max_concurrent_turns=config.max_concurrent_turns,
            max_active_prompts=config.max_active_prompts,
        )

    @property
    def max_queued(self) -> int:
        return self._max_queued

    @property
    def active_prompts(self) -> int:
        """Prompts admitted across all this connection's sessions, running or waiting."""
        return self._active_prompts

    @asynccontextmanager
    async def admit(self) -> AsyncIterator[None]:
        """Admit one prompt against the connection-wide bounds.

        Wraps the whole prompt, *outside* the session's own turn lock, so a
        prompt queued behind its session still counts against
        ``max_active_prompts``. That is the point: the cost being bounded here is
        a parked request handler, which exists whether or not its turn has
        started.
        """
        if self._active_prompts >= self._max_active_prompts:
            raise ConnectionOverloadedError(self._max_active_prompts)
        self._active_prompts += 1
        try:
            yield
        finally:
            self._active_prompts -= 1

    @asynccontextmanager
    async def running_turn(self) -> AsyncIterator[None]:
        """Hold one of the connection's concurrent-turn slots.

        Taken *inside* the session's turn lock, so a prompt waiting its turn on a
        busy session does not sit on a slot another session could be using.
        """
        async with self._turn_slots:
            yield

    @property
    def storage(self) -> Storage:
        return self._storage

    def __len__(self) -> int:
        return len(self._sessions)

    async def create(self, **context: Any) -> AgentSession:
        """Mint a new session id and register it. ``context`` seeds ``AgentSession``.

        Raises :class:`SessionLimitError` when the registry is full and every
        session in it is busy — see that class for why admission is refused
        rather than the cap being stretched.
        """
        async with self._lock:
            now = self._clock()
            await self._evict_expired(now)
            await self._make_room()
            session = AgentSession(
                session_id=uuid4().hex,
                stream_id=uuid4(),
                max_queued=self._max_queued,
                clock=self._clock,
                last_active=now,
                **context,
            )
            self._sessions[session.session_id] = session
            return session

    async def get(self, session_id: str) -> AgentSession:
        """Return a live session, refreshing its LRU position.

        Raises :class:`UnknownSessionError` for an id this store never issued or
        has since evicted.
        """
        async with self._lock:
            now = self._clock()
            await self._evict_expired(now)
            session = self._sessions.get(session_id)
            if session is None:
                raise UnknownSessionError(session_id)
            session.last_active = now
            self._sessions.move_to_end(session_id)
            return session

    def stream(self, session: AgentSession) -> MemoryStream:
        """``session``'s stream, created once and kept for its lifetime.

        A fresh object per turn — the shape :class:`ag2.mcp.sessions.SessionStore`
        uses — keeps per-turn subscribers from piling up, but a stream carries
        more than history. Its inbox (``pending_messages``) and its background
        tasks live on the object, so a background task finishing after the turn
        that spawned it would deliver onto a stream nobody reads again, and
        nothing on the session could see that work was still running at teardown.

        Keeping one stream fixes both. Per-turn subscribers are dealt with where
        they are added, by unsubscribing when the turn ends.
        """
        if session.stream is None:
            session.stream = MemoryStream(storage=self._storage, id=session.stream_id)
        return session.stream

    async def close(self, session_id: str) -> None:
        """Cancel a session's live work and drop it, deleting its history.

        Explicit teardown, so unlike eviction it *does* stop work in progress.
        Durable state belongs to an injected :class:`Storage`; the default
        in-memory one has none to keep.
        """
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise UnknownSessionError(session_id)
        await self._discard(session)

    async def aclose(self) -> None:
        """Cancel and drop every session — the shutdown path."""
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            await self._discard(session)

    async def _discard(self, session: AgentSession) -> None:
        """Stop a session's work, wait for it to unwind, then delete its history.

        Awaiting the cancelled task matters: dropping history while the turn is
        still winding down races the agent's own last writes, which would leave
        the store holding events for a session that no longer exists.

        Background work spawned by a tool or a subagent outlives the turn that
        started it, so it is stopped here too. Otherwise a closed session could
        keep calling out to the world long after the Client that asked for it
        disconnected.
        """
        await session.cancel()
        task = session.turn_task
        if task is not None and not task.done():
            with suppress(asyncio.CancelledError, Exception):
                await task
        await _stop_background(session)
        await self._storage.drop_history(session.stream_id)

    async def _evict_expired(self, now: float) -> None:
        """Drop idle sessions past the TTL.

        A session running or queueing a turn is never expired out from under
        itself — an eviction policy is a memory bound, not a reason to kill work
        somebody is waiting on. Its clock restarts when the turn finishes.
        """
        if self._ttl is None:
            return
        expired = [
            sid for sid, session in self._sessions.items() if session.is_idle and now - session.last_active > self._ttl
        ]
        for sid in expired:
            session = self._sessions.pop(sid)
            await self._storage.drop_history(session.stream_id)

    async def _make_room(self) -> None:
        """Free a slot for one new session, or refuse to admit it.

        Called *before* the new session exists, so there is no risk of choosing
        it as the victim and handing back an id that is already dead.

        Eviction never takes a session with work running or queued — an eviction
        policy is a memory bound, not a reason to kill something a Client is
        waiting on. When nothing idle is left to take, admission fails instead.
        Stretching the cap "just while turns are in flight" sounds bounded but is
        not: a Client can keep every session busy indefinitely and grow the
        registry without limit.
        """
        while len(self._sessions) >= self._max:
            victim = next((sid for sid, session in self._sessions.items() if session.is_idle), None)
            if victim is None:
                raise SessionLimitError(self._max)
            session = self._sessions.pop(victim)
            await self._storage.drop_history(session.stream_id)
