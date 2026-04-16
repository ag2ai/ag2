# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Hub — FS-backed registry, session state machine, and dispatch.

Phase 1 surface:

* ``register`` / ``unregister`` / ``find`` / ``describe`` — registry CRUD
* ``create_session`` — handshake that allocates a session id, writes
  metadata, delivers an invite to the recipient, waits for the ack, and
  returns an ACTIVE :class:`SessionMetadata`.
* ``post_envelope`` — validates the envelope against the session adapter,
  appends it to the WAL, and delivers it to the recipient's endpoint.
* ``close_session`` — explicit close transition.
* ``connection_handler`` — Link callback that drives the ``hello`` /
  ``send`` / ``receipt`` / ``subscribe`` frame loop for a connected actor.

The hub never calls ``Actor.ask`` — it only delivers envelopes. The actor's
:class:`~autogen.beta.network.client.ActorClient` is responsible for running
the notify handler and posting the reply back.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from ..adapters import (
    AdapterResult,
    AuctionAdapter,
    BroadcastAdapter,
    ConsultingAdapter,
    ConversationAdapter,
    DiscussionAdapter,
    NotificationAdapter,
    SessionAdapter,
)
from ..auth import AuthRegistry, default_registry
from ..envelope import (
    EV_SESSION_CLOSED,
    EV_SESSION_INVITE,
    EV_SESSION_INVITE_ACK,
    EV_SESSION_INVITE_REJECT,
    EV_SESSION_OPENED,
    EV_TASK_ASSIGNED,
    EV_TASK_CANCELLED,
    EV_TASK_ERROR,
    EV_TASK_EXPIRED,
    EV_TASK_PHASE_COMPLETED,
    EV_TASK_PHASE_ENTERED,
    EV_TASK_PROGRESS,
    EV_TASK_RESULT,
    TASK_EVENT_TYPES,
    Envelope,
)
from ..errors import (
    AccessDeniedError,
    DuplicateRegistrationError,
    InboxFullError,
    InviteRejectedError,
    LimitExceededError,
    RuleViolationError,
    SessionClosedError,
    SessionError,
    SessionTypeError,
    TaskStateError,
    UnknownActorError,
    UnknownSessionError,
    UnknownTaskError,
)
from ..events import EventRegistry, EventTypeSpec, UnknownEventTypeError
from ..identity import ActorIdentity
from ..ids import new_id
from ..rule import Rule
from ..session_types import (
    Participant,
    ParticipantRole,
    SessionMetadata,
    SessionState,
    SessionType,
)
from ..task import (
    TaskMetadata,
    TaskSpec,
    TaskState,
)
from ..transport.frames import (
    AcceptFrame,
    ChunkFrame,
    ErrorFrame,
    EventFrame,
    Frame,
    HelloFrame,
    NotifyFrame,
    ReceiptFrame,
    RuleChangedFrame,
    SendFrame,
    SubscribeFrame,
    UnsubscribeFrame,
    WelcomeFrame,
)
from . import layout
from ._limits import RateLimiter

if TYPE_CHECKING:
    from autogen.beta.knowledge import KnowledgeStore

    from ..transport.local import _EndpointSide

log = logging.getLogger("autogen.beta.network.hub")


# ---------------------------------------------------------------------------
# Small dataclasses used internally
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HubConfig:
    """Hub-level configuration.

    ``invite_ack_timeout_s`` is how long :meth:`Hub.create_session` waits
    for invited participants to ack before rolling the session back. It
    used to be a per-call keyword argument; moving it to config lets
    operators tune handshake latency once per deployment.

    ``idempotency_ttl_s`` is the window the hub keeps
    ``(session_id, idempotency_key)`` → ``(envelope_id, wal_offset)``
    mappings in memory for Phase 2 ``send`` dedup. Default is 600
    seconds (10 minutes), matching the design doc (§13.4). Repeat
    sends within this window get the cached result without
    re-running adapters or rate limits.
    """

    hub_id: str = ""
    default_rule: Rule = field(default_factory=Rule)
    invite_ack_timeout_s: float = 5.0
    idempotency_ttl_s: float = 600.0


@dataclass(slots=True)
class _Subscription:
    subscription_id: str
    actor_id: str
    session_id: str | None
    causation_id: str | None
    endpoint: _EndpointSide
    since: int = 0


@dataclass(slots=True)
class _IdempotencyEntry:
    """Cached ``(envelope_id, wal_offset)`` result for a dedup window."""

    envelope_id: str
    wal_offset: int
    expires_at: float


@dataclass(slots=True)
class _PendingInvite:
    """Tracks outstanding invite acks for a multi-participant handshake.

    ``pending_ids`` is the set of actor ids we are still waiting on.
    ``required`` is the quorum threshold — once that many participants
    have acked (and the outstanding rejects leave enough headroom to
    reach it), the future resolves with the updated metadata.

    Single-participant sessions (consulting / conversation /
    notification) collapse to ``required=1, pending_ids={recipient}`` —
    the one-ack fast path is a special case of the quorum rule.
    """

    session_id: str
    future: asyncio.Future[SessionMetadata]
    required: int
    pending_ids: set[str] = field(default_factory=set)
    acked_ids: set[str] = field(default_factory=set)
    rejected_ids: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hub
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _add_seconds(iso_ts: str, seconds: int) -> str:
    """Return ``iso_ts`` advanced by ``seconds``, in the same ISO-Z format."""

    base = datetime.strptime(iso_ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (base + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_seconds_since(start: str, now: str) -> float:
    """Return the number of seconds between two ISO-Z timestamps.

    Used by :meth:`Hub.metrics` to compute ``uptime_s``. Clock-monotonic
    guarantees come from the hub's injected ``clock`` — for real clocks
    this is wall time; for deterministic tests it's whatever the fake
    clock returns. Returns ``0.0`` if either timestamp fails to parse
    so the metrics endpoint never crashes the hub.
    """

    try:
        a = datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        b = datetime.strptime(now, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:  # pragma: no cover
        return 0.0
    return (b - a).total_seconds()


def _type_name(session_type: SessionType | str) -> str:
    """Normalize a session-type argument to its canonical string value.

    ``SessionType`` members are str subclasses and compare equal to their
    values, but ``str(SessionType.CONSULTING)`` returns the qualified
    name ``"SessionType.CONSULTING"`` in Python 3.11+ instead of
    ``"consulting"``. Use this helper anywhere the name is logged, used
    as a dict key, or compared to a rule pattern.
    """

    if isinstance(session_type, SessionType):
        return session_type.value
    return str(session_type)


class Hub:
    """FS-backed hub. See module docstring for the public surface."""

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        auth: AuthRegistry | None = None,
        adapters: list[SessionAdapter] | None = None,
        default_rule: Rule | None = None,
        hub_id: str | None = None,
        invite_ack_timeout_s: float | None = None,
        idempotency_ttl_s: float | None = None,
        event_registry: EventRegistry | None = None,
        clock: Callable[[], str] = _now_iso,
    ) -> None:
        self._store = store
        self._auth = auth or default_registry()
        self._clock = clock
        self.config = HubConfig(
            hub_id=hub_id or new_id(),
            default_rule=default_rule or Rule(),
            invite_ack_timeout_s=(
                invite_ack_timeout_s if invite_ack_timeout_s is not None else 5.0
            ),
            idempotency_ttl_s=(
                idempotency_ttl_s if idempotency_ttl_s is not None else 600.0
            ),
        )
        self._event_registry = event_registry or EventRegistry()

        # Adapters are keyed by string session-type name so operators can
        # register custom types that are not members of the built-in enum.
        # Built-in adapters use ``SessionType`` enum members whose values
        # subclass ``str``, so ``a.session_type`` already coerces to the
        # right key.
        self._adapters: dict[str, SessionAdapter] = {}
        if adapters is None:
            adapters = [
                ConsultingAdapter(),
                ConversationAdapter(),
                NotificationAdapter(),
                BroadcastAdapter(),
                DiscussionAdapter(),
                AuctionAdapter(),
            ]
        for adapter in adapters:
            self.register_adapter(adapter)

        # In-memory indexes rebuilt from the store on demand.
        self._name_to_id: dict[str, str] = {}
        self._identities: dict[str, ActorIdentity] = {}
        self._rules: dict[str, Rule] = {}
        # Dedupe RuleChangedFrame emissions. ``set_rule`` writes the
        # rule to the store, which synchronously fires the on_change
        # callback on stores that support it (MemoryKnowledgeStore
        # today, more in Phase 3b). Both ``set_rule`` and the
        # ``_reload_actor_rule`` listener end up calling
        # ``_emit_rule_changed``; we compare the serialized rule
        # against the last emission per actor and skip the duplicate.
        self._last_emitted_rule_json: dict[str, str] = {}
        self._sessions: dict[str, SessionMetadata] = {}
        self._active_sessions: dict[str, set[str]] = {}  # actor_id → session_ids
        self._endpoints: dict[str, _EndpointSide] = {}
        self._subscriptions: dict[str, _Subscription] = {}
        self._pending_invites: dict[str, _PendingInvite] = {}

        self._lock = asyncio.Lock()
        # Serializes WAL appends with subscription registration so that
        # every posted envelope is either delivered via replay (the sub
        # was registered after this append) or via fan-out (the sub was
        # registered before this append) — never both, never neither.
        # See the note in :meth:`_handle_subscribe` for the invariant.
        self._wal_lock = asyncio.Lock()
        self._rate_limiter = RateLimiter()
        # Idempotency dedup — (session_id, idempotency_key) → cached
        # ``(envelope_id, wal_offset, expires_at)``. Lazily GC'd on
        # lookup so Phase 2 doesn't need a background sweeper.
        self._idempotency: dict[tuple[str, str], _IdempotencyEntry] = {}
        # Phase 3a structured inbox — per-actor count of envelopes
        # currently sitting in ``hub/actors/{id}/inbox/pending/``. Bumped
        # on _deliver_to, decremented on ReceiptFrame(ack)/(nack) via
        # _handle_receipt, rebuilt from the store on hydrate. Used for
        # fast ``max_pending`` checks in _preflight_inbox_capacity so
        # we don't stat the pending dir on every post_envelope call.
        self._inbox_pending: dict[str, int] = {}
        # Phase 4 — network task state. ``_tasks`` is the authoritative
        # in-memory cache of every task the hub knows about; cold
        # restart :meth:`hydrate` rebuilds it from
        # ``hub/tasks/*/metadata.json``. Task transitions apply both
        # to the cache entry AND to the on-disk metadata file so
        # recovery stays consistent.
        self._tasks: dict[str, TaskMetadata] = {}
        # Per-session set of non-terminal task ids, used for the
        # session-close cascade (every active task is cancelled when
        # its session closes) and for fast ``session.track_tasks()``
        # lookups without rescanning the whole cache.
        self._session_tasks: dict[str, set[str]] = {}
        # Phase 3b — in-memory metrics counters. Updated incrementally
        # on every mutation so ``Hub.metrics()`` and ``/v1/admin/metrics``
        # do not scan the audit log or the store. Rebuilt from in-memory
        # state on demand (see :meth:`metrics`).
        self._started_at = self._clock()
        self._sessions_closed_total: int = 0
        self._tasks_completed_total: int = 0
        self._tasks_failed_total: int = 0
        # Phase 3b — async audit log writer task queue. Populated by
        # :meth:`_audit`; drained by a background task started on
        # :meth:`open` / :meth:`hydrate` so auditing never blocks the
        # hot path. See :meth:`_start_audit_writer`.
        self._audit_queue: asyncio.Queue[dict[str, Any]] | None = None
        self._audit_task: asyncio.Task[None] | None = None
        # Phase 3b — on_change cache invalidation. When the store
        # fires a change on ``hub/actors/{id}/identity.json`` or
        # ``rule.json`` (e.g. an operator edits the file out of band
        # or ``PUT /v1/actors/{id}/rule`` is called by a sidecar
        # process sharing the same store), the callback drops the
        # corresponding in-memory cache entry. Subscription handle
        # is closed on :meth:`close`.
        self._cache_sub = None  # type: Any
        # Set of paths we've seen recently in a cache-invalidation
        # callback — used to dedupe bursts of writes from the same
        # mutation path so we do not spend the lock twice per edit.
        self._cache_invalidation_lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def open(
        cls,
        store: KnowledgeStore,
        *,
        auth: AuthRegistry | None = None,
        adapters: list[SessionAdapter] | None = None,
        default_rule: Rule | None = None,
        hub_id: str | None = None,
        invite_ack_timeout_s: float | None = None,
        idempotency_ttl_s: float | None = None,
        event_registry: EventRegistry | None = None,
        clock: Callable[[], str] = _now_iso,
    ) -> Hub:
        """Construct a :class:`Hub` and hydrate its in-memory indexes.

        Prefer this over ``Hub(store)`` whenever the store may already
        contain sessions from a previous process. ``__init__`` stays
        synchronous (so tests and throwaway fixtures don't need to be
        async) but cold-restart callers want the async hydrate path.
        """

        hub = cls(
            store,
            auth=auth,
            adapters=adapters,
            default_rule=default_rule,
            hub_id=hub_id,
            invite_ack_timeout_s=invite_ack_timeout_s,
            idempotency_ttl_s=idempotency_ttl_s,
            event_registry=event_registry,
            clock=clock,
        )
        await hub.hydrate()
        await hub._start_audit_writer()
        await hub._start_cache_invalidation()
        return hub

    # ------------------------------------------------------------------
    # Cache invalidation via store.on_change (Phase 3b)
    # ------------------------------------------------------------------

    async def _start_cache_invalidation(self) -> None:
        """Subscribe to ``hub/actors/*`` changes and invalidate caches.

        When an operator (or a sidecar process) rewrites
        ``hub/actors/{id}/identity.json`` or ``hub/actors/{id}/rule.json``
        via the shared store, the watcher's callback drops the stale
        in-memory entry and re-reads the file. A long-running hub
        therefore picks up out-of-band edits without a full restart.

        Stores that cannot observe changes (``MemoryKnowledgeStore``
        without active subscribers, or backends returning
        ``NoopChangeSubscription``) silently no-op — the invalidation
        layer is opportunistic, not required for correctness.

        Scoped intentionally small: only identity + rule invalidation.
        Cross-process subscription fan-out for session envelopes
        stays out of Phase 3b (see §14 "Deferred"), because that
        would conflate single-machine convenience with true
        multi-writer coordination.
        """

        try:
            self._cache_sub = await self._store.on_change(
                layout.ACTORS_ROOT, self._on_actor_path_change
            )
        except Exception:  # pragma: no cover
            log.warning(
                "on_change subscription failed; cache invalidation disabled",
                exc_info=True,
            )

    async def _stop_cache_invalidation(self) -> None:
        """Close the on_change subscription. Idempotent."""

        sub = self._cache_sub
        self._cache_sub = None
        if sub is None:
            return
        try:
            await sub.close()
        except Exception:  # pragma: no cover
            log.warning("on_change close failed", exc_info=True)

    async def _on_actor_path_change(self, changed_path: str) -> None:
        """Handle a change notification for ``hub/actors/{id}/(identity|rule).json``.

        Dispatches to :meth:`_reload_actor_identity` or
        :meth:`_reload_actor_rule` depending on which file was
        rewritten. Paths that do not match either pattern (e.g.
        ``runtime.json``, ``inbox/pending/foo.json``, ``SKILL.md``)
        are ignored — the caches for those either do not exist or are
        intentionally re-read on every use.
        """

        if not changed_path.startswith(layout.ACTORS_ROOT + "/"):
            return
        parts = changed_path[len(layout.ACTORS_ROOT) + 1 :].split("/")
        if len(parts) < 2:
            return
        actor_id = parts[0]
        filename = parts[1]
        async with self._cache_invalidation_lock:
            if filename == "identity.json":
                await self._reload_actor_identity(actor_id)
            elif filename == "rule.json":
                await self._reload_actor_rule(actor_id)

    async def _reload_actor_identity(self, actor_id: str) -> None:
        """Re-read the identity from the store and refresh caches.

        Handles three cases:
        * File present, same actor_id → update :attr:`_identities`
          and the name-index (dropping any stale name pointer).
        * File absent → drop the actor from every cache
          (treat as unregister).
        * Read error → log and leave the stale entry in place
          (better to serve old data than crash).
        """

        try:
            raw = await self._store.read(layout.actor_identity(actor_id))
        except Exception:  # pragma: no cover
            log.warning(
                "cache invalidation: identity read failed for %s",
                actor_id,
                exc_info=True,
            )
            return
        if raw is None:
            old = self._identities.pop(actor_id, None)
            self._rules.pop(actor_id, None)
            self._last_emitted_rule_json.pop(actor_id, None)
            self._active_sessions.pop(actor_id, None)
            self._inbox_pending.pop(actor_id, None)
            if old is not None:
                self._name_to_id.pop(old.name, None)
            return
        try:
            identity = ActorIdentity.from_json(raw)
        except Exception:  # pragma: no cover
            log.warning(
                "cache invalidation: identity parse failed for %s",
                actor_id,
                exc_info=True,
            )
            return
        skill = await self._store.read(layout.actor_skill(actor_id))
        if skill is not None:
            identity.skill_md = skill
        old = self._identities.get(actor_id)
        if old is not None and old.name != identity.name:
            self._name_to_id.pop(old.name, None)
        self._identities[actor_id] = identity
        self._name_to_id[identity.name] = actor_id

    async def _reload_actor_rule(self, actor_id: str) -> None:
        """Re-read the rule from the store and refresh the cache.

        Also emits :class:`RuleChangedFrame` to the actor's live
        endpoint so out-of-band rule edits (operator tooling, another
        process sharing the store) propagate to the client's transforms
        pipeline. Without this the §4.3 "any write to rule.json emits
        a RuleChanged event" contract would only hold for
        :meth:`set_rule`.
        """

        try:
            raw = await self._store.read(layout.actor_rule(actor_id))
        except Exception:  # pragma: no cover
            log.warning(
                "cache invalidation: rule read failed for %s",
                actor_id,
                exc_info=True,
            )
            return
        if raw is None:
            self._rules.pop(actor_id, None)
            return
        try:
            rule = Rule.from_json(raw)
        except Exception:  # pragma: no cover
            log.warning(
                "cache invalidation: rule parse failed for %s",
                actor_id,
                exc_info=True,
            )
            return
        self._rules[actor_id] = rule
        await self._emit_rule_changed(actor_id, rule)

    # ------------------------------------------------------------------
    # Audit log writer (Phase 3b)
    # ------------------------------------------------------------------

    async def _start_audit_writer(self) -> None:
        """Start the background audit log writer task.

        Idempotent: calling twice is a no-op. The writer drains
        :attr:`_audit_queue` and appends each entry as a single JSON
        line to ``hub/admin/audit.jsonl`` via the store's ``append``
        method. Failures are logged and swallowed — audit is always
        best-effort so a store outage never refuses a mutation.
        """

        if self._audit_task is not None and not self._audit_task.done():
            return
        self._audit_queue = asyncio.Queue(maxsize=10000)
        self._audit_task = asyncio.create_task(
            self._audit_writer_loop(), name="hub-audit-writer"
        )

    async def _stop_audit_writer(self) -> None:
        """Drain pending entries and stop the writer task.

        Called by :meth:`close`. The writer wakes on a sentinel ``None``
        entry in the queue and exits cleanly. If the task is still
        running after a short grace period, it is cancelled.
        """

        if self._audit_task is None:
            return
        queue = self._audit_queue
        if queue is not None:
            try:
                queue.put_nowait(None)  # type: ignore[arg-type]
            except asyncio.QueueFull:  # pragma: no cover
                self._audit_task.cancel()
        try:
            await asyncio.wait_for(self._audit_task, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):  # pragma: no cover
            self._audit_task.cancel()
        self._audit_task = None
        self._audit_queue = None

    async def _audit_writer_loop(self) -> None:
        """Consume the audit queue and append entries to ``audit.jsonl``."""

        audit_path = layout.admin_audit_log()
        queue = self._audit_queue
        assert queue is not None
        while True:
            entry = await queue.get()
            if entry is None:
                return
            try:
                line = json.dumps(entry, sort_keys=True) + "\n"
                await self._store.append(audit_path, line)
            except Exception:  # pragma: no cover
                log.warning("audit write failed for %s", entry.get("action"))

    async def close(self) -> None:
        """Release hub-owned background resources.

        Phase 3b starts the audit log writer and the cache
        invalidation subscription as background resources when
        :meth:`open` is used. ``close`` drains the audit queue,
        stops the writer task, and closes the on_change
        subscription for a clean asyncio teardown. Tests that
        construct ``Hub(store)`` directly (no ``open``) never
        start either resource and do not need to call ``close``
        — it's a safe no-op.
        """

        await self._stop_audit_writer()
        await self._stop_cache_invalidation()

    # ------------------------------------------------------------------
    # Adapter registry
    # ------------------------------------------------------------------

    def register_adapter(self, adapter: SessionAdapter) -> None:
        """Register a session adapter under its declared ``session_type``.

        Replaces any existing adapter with the same name, logging a
        warning — operators deliberately ship stricter replacements for
        built-in types this way.
        """

        key = _type_name(adapter.session_type)
        if key in self._adapters and self._adapters[key] is not adapter:
            log.warning(
                "hub: replacing session adapter for %r with %s",
                key,
                type(adapter).__name__,
            )
        self._adapters[key] = adapter

    def _adapter_for(self, session_type: SessionType | str) -> SessionAdapter:
        key = _type_name(session_type)
        try:
            return self._adapters[key]
        except KeyError as exc:
            raise SessionTypeError(f"no adapter registered for {key!r}") from exc

    def adapter_names(self) -> list[str]:
        """Return the sorted list of registered session-type names."""

        return sorted(self._adapters)

    # ------------------------------------------------------------------
    # Event type registry
    # ------------------------------------------------------------------

    @property
    def event_registry(self) -> EventRegistry:
        """The :class:`EventRegistry` consulted at post-envelope time."""

        return self._event_registry

    def register_event_type(self, spec: EventTypeSpec | str) -> EventTypeSpec:
        """Register an event type name on this hub's registry."""

        return self._event_registry.register(spec)

    # ------------------------------------------------------------------
    # Cold-restart hydrate
    # ------------------------------------------------------------------

    async def hydrate(self) -> None:
        """Rebuild in-memory indexes from the backing store.

        Walks ``hub/actors/*/identity.json``, ``hub/actors/*/rule.json``,
        and ``hub/sessions/*/metadata.json`` from the store, rebuilds
        :attr:`_identities`, :attr:`_rules`, :attr:`_name_to_id`,
        :attr:`_sessions`, and :attr:`_active_sessions`, and reconciles
        half-written PENDING sessions by transitioning them to
        :attr:`SessionState.EXPIRED` (nobody can finish a handshake
        whose future is gone). Closed / expired sessions are still
        loaded so WAL reads and session queries keep working, but they
        do not consume participant slots in ``_active_sessions``.

        Hydrate is required before Phase 3 multi-process deployments —
        without it the first cold restart silently loses every live
        session's membership state. Phase 2 ships it as an opt-in
        behavior triggered via ``Hub(store, hydrate=True)``; Phase 3
        will make it the default once the WsLink transport is in place.
        """

        # Actors first — identities + rules + name index.
        actor_ids = await self._store.list(layout.ACTORS_ROOT)
        for entry in actor_ids:
            if not entry.endswith("/"):
                continue
            actor_id = entry.rstrip("/")
            identity_raw = await self._store.read(layout.actor_identity(actor_id))
            rule_raw = await self._store.read(layout.actor_rule(actor_id))
            if identity_raw is None or rule_raw is None:
                # Partial write from an older crash — skip; the
                # operator can clean up or re-register.
                log.warning(
                    "hub.hydrate: actor %s is missing identity/rule, skipping",
                    actor_id,
                )
                continue
            identity = ActorIdentity.from_json(identity_raw)
            # SKILL.md is a sidecar; pull it back into the in-memory
            # identity so ``describe_actor`` returns the full object.
            skill = await self._store.read(layout.actor_skill(actor_id))
            if skill is not None:
                identity.skill_md = skill
            rule = Rule.from_json(rule_raw)
            self._identities[actor_id] = identity
            self._rules[actor_id] = rule
            self._name_to_id[identity.name] = actor_id
            self._active_sessions.setdefault(actor_id, set())
            # Rebuild the pending inbox counter by listing pending/ —
            # authoritative after restart since the in-memory count was
            # wiped with the process.
            pending = await self._store.list(
                layout.actor_inbox_pending_dir(actor_id) + "/"
            )
            self._inbox_pending[actor_id] = sum(
                1 for entry in pending if entry.endswith(".json")
            )

        # Sessions next — metadata only (WAL stays on disk).
        session_ids = await self._store.list(layout.SESSIONS_ROOT)
        for entry in session_ids:
            if not entry.endswith("/"):
                continue
            session_id = entry.rstrip("/")
            meta_raw = await self._store.read(layout.session_metadata(session_id))
            if meta_raw is None:
                log.warning(
                    "hub.hydrate: session %s has no metadata, skipping", session_id
                )
                continue
            metadata = SessionMetadata.from_json(meta_raw)

            # Half-written PENDING sessions cannot be resumed: the
            # in-memory ``_PendingInvite`` future is gone, so any ack
            # arriving now has no one to notify. Mark them EXPIRED so
            # read-only inspection still works but new envelopes are
            # rejected. Participant slots are NOT reserved for these.
            if metadata.state is SessionState.PENDING:
                metadata.state = SessionState.EXPIRED
                metadata.closed_at = self._clock()
                metadata.close_reason = "hydrate_orphaned_pending"
                await self._write_session_metadata(metadata)

            self._sessions[session_id] = metadata
            if metadata.state is SessionState.ACTIVE:
                for p in metadata.participants:
                    self._active_sessions.setdefault(p.actor_id, set()).add(session_id)
            # Phase 3b — sessions whose WAL has already been archived
            # remain in the metadata cache (so /v1/actors/{id}/activity
            # and /v1/sessions still surface them via
            # ``metadata.archived_at``) but their live WAL at
            # ``hub/sessions/{id}/wal.jsonl`` has been moved to
            # ``hub/archive/sessions/{id}/wal.jsonl`` plus a
            # ``summary.json`` sibling. ``read_wal`` intentionally
            # returns an empty list for archived sessions in 3b —
            # callers that need the archived bytes read them from
            # the archive path directly via the store.

        # Phase 4 — rebuild the task cache from ``hub/tasks/*/metadata.json``.
        # Tasks whose session has been archived are loaded but not added to
        # the session→task index; the TTL sweeper will still expire them
        # if the in-memory record is non-terminal. Terminal tasks stay
        # loaded so read-only inspection works.
        task_entries = await self._store.list(layout.TASKS_ROOT)
        for entry in task_entries:
            if not entry.endswith("/"):
                continue
            task_id = entry.rstrip("/")
            task_raw = await self._store.read(layout.task_metadata(task_id))
            if task_raw is None:
                log.warning("hub.hydrate: task %s has no metadata, skipping", task_id)
                continue
            try:
                task = TaskMetadata.from_json(task_raw)
            except Exception:  # pragma: no cover — corrupt file
                log.warning(
                    "hub.hydrate: task %s metadata failed to parse, skipping",
                    task_id,
                )
                continue
            self._tasks[task.task_id] = task
            if not task.is_terminal():
                self._session_tasks.setdefault(task.session_id, set()).add(task.task_id)

    # ------------------------------------------------------------------
    # Registry CRUD
    # ------------------------------------------------------------------

    async def register(
        self,
        identity: ActorIdentity,
        rule: Rule | None = None,
        *,
        auth_claim: dict[str, Any] | None = None,
    ) -> ActorIdentity:
        """Register an identity and return a stamped copy."""

        await self._auth.validate(identity, auth_claim or {})

        async with self._lock:
            if identity.name in self._name_to_id:
                raise DuplicateRegistrationError(
                    f"name already registered: {identity.name}"
                )

            actor_id = new_id()
            stamped = identity.with_actor_id(actor_id)
            applied_rule = rule if rule is not None else self.config.default_rule

            await self._store.write(
                layout.actor_identity(actor_id),
                stamped.to_json(),
            )
            await self._store.write(
                layout.actor_rule(actor_id),
                applied_rule.to_json(),
            )
            await self._store.write(
                layout.actor_runtime(actor_id),
                json.dumps(
                    {
                        "actor_id": actor_id,
                        "binding": "local",
                        "reachable": False,
                        "last_heartbeat": self._clock(),
                    },
                    sort_keys=True,
                ),
            )
            if stamped.skill_md:
                await self._store.write(layout.actor_skill(actor_id), stamped.skill_md)
            await self._store.write(layout.name_pointer(stamped.name), actor_id)

            self._name_to_id[stamped.name] = actor_id
            self._identities[actor_id] = stamped
            self._rules[actor_id] = applied_rule
            self._active_sessions[actor_id] = set()
            self._inbox_pending[actor_id] = 0

        self._audit(
            actor_id=actor_id,
            action="register_actor",
            resource_type="actor",
            resource_id=actor_id,
            decision="allow",
            reason=stamped.name,
        )
        return stamped

    async def unregister(self, actor_id: str) -> None:
        async with self._lock:
            identity = self._identities.pop(actor_id, None)
            if identity is None:
                raise UnknownActorError(actor_id)
            self._rules.pop(actor_id, None)
            self._last_emitted_rule_json.pop(actor_id, None)
            self._active_sessions.pop(actor_id, None)
            self._name_to_id.pop(identity.name, None)
            self._inbox_pending.pop(actor_id, None)

        await self._store.delete(layout.actor_dir(actor_id))
        await self._store.delete(layout.name_pointer(identity.name))
        endpoint = self._endpoints.pop(actor_id, None)
        if endpoint is not None and not endpoint.closed:
            await endpoint.close()

        self._audit(
            actor_id=actor_id,
            action="unregister_actor",
            resource_type="actor",
            resource_id=actor_id,
            decision="allow",
            reason=identity.name,
        )

    async def find(
        self,
        capability: str | None = None,
        *,
        query: str | None = None,
    ) -> list[ActorIdentity]:
        """Discover registered identities by capability and/or free-text query.

        ``capability`` matches an exact entry in ``identity.capabilities``.

        ``query`` is a case-insensitive substring match against the
        identity's ``name``, ``display``, ``summary``, ``domains`` (any
        entry), and ``strengths``. Useful for the Phase 6 ``find_actors``
        LLM verb where the model has a free-text intent but no idea
        what capability strings the operator declared.

        When both filters are supplied the result is the AND.
        """

        async with self._lock:
            identities = list(self._identities.values())

        if capability is not None:
            identities = [i for i in identities if capability in i.capabilities]

        if query:
            needle = query.casefold()

            def _matches(identity: ActorIdentity) -> bool:
                haystacks: list[str] = [
                    identity.name or "",
                    identity.display or "",
                    identity.summary or "",
                    identity.strengths or "",
                ]
                haystacks.extend(identity.domains or ())
                return any(needle in h.casefold() for h in haystacks if h)

            identities = [i for i in identities if _matches(i)]

        return identities

    async def describe(self, name_or_id: str) -> ActorIdentity:
        actor_id = self._resolve_actor(name_or_id)
        return self._identities[actor_id]

    async def get_rule(self, actor_id: str) -> Rule:
        if actor_id not in self._rules:
            raise UnknownActorError(actor_id)
        return self._rules[actor_id]

    async def set_rule(self, actor_id: str, rule: Rule) -> None:
        """Replace an actor's rule and persist the new version.

        Used by the Phase 3b ``PUT /v1/actors/{id}/rule`` endpoint and
        by operator tooling that edits rules out of band. The in-memory
        cache is updated atomically with the store write so a
        concurrent post_envelope never sees a half-updated rule.
        Emits an audit entry and a :class:`RuleChangedFrame` to the
        actor's live endpoint so the client's transforms pipeline
        reloads. Phase 5a wires the frame to actually drive
        transform execution.
        """

        async with self._lock:
            if actor_id not in self._identities:
                raise UnknownActorError(actor_id)
            await self._store.write(layout.actor_rule(actor_id), rule.to_json())
            self._rules[actor_id] = rule

        self._audit(
            actor_id=actor_id,
            action="update_rule",
            resource_type="rule",
            resource_id=actor_id,
            decision="allow",
            reason="",
        )

        await self._emit_rule_changed(actor_id, rule)

    async def _emit_rule_changed(self, actor_id: str, rule: Rule) -> None:
        """Push a :class:`RuleChangedFrame` to the actor's live endpoint.

        Called from both :meth:`set_rule` (explicit rule replacement)
        and :meth:`_reload_actor_rule` (FS-watcher pickup of out-of-band
        edits) so §4.3's "any write to rule.json emits a RuleChanged
        event" contract holds for both paths. Silently no-ops when the
        actor has no live endpoint — offline actors see the new rule on
        their next hello.

        Deduplication: ``set_rule`` writes to the store before calling
        this helper, and stores whose ``on_change`` fires synchronously
        (e.g. :class:`MemoryKnowledgeStore`) will re-invoke the helper
        via :meth:`_reload_actor_rule` inside the same call stack. We
        track the last-emitted serialized rule per actor and skip the
        emission when the content is byte-identical to the previous
        one — so a single logical rule change produces exactly one
        frame regardless of how many hub entry points observed it.
        """

        endpoint = self._endpoints.get(actor_id)
        if endpoint is None or endpoint.closed:
            return
        serialized = rule.to_json()
        if self._last_emitted_rule_json.get(actor_id) == serialized:
            return
        try:
            await endpoint.send_frame(
                RuleChangedFrame(
                    actor_id=actor_id,
                    transforms=[t.to_dict() for t in rule.transforms],
                    version=rule.version,
                )
            )
        except Exception:  # pragma: no cover
            log.warning(
                "rule_changed push failed for %s", actor_id, exc_info=True
            )
            return
        self._last_emitted_rule_json[actor_id] = serialized

    def _resolve_actor(self, name_or_id: str) -> str:
        if name_or_id in self._identities:
            return name_or_id
        actor_id = self._name_to_id.get(name_or_id)
        if actor_id is None:
            raise UnknownActorError(name_or_id)
        return actor_id

    # ------------------------------------------------------------------
    # Session creation and handshake
    # ------------------------------------------------------------------

    async def create_session(
        self,
        *,
        creator_id: str,
        session_type: SessionType | str,
        participant_names: list[str],
        labels: dict[str, Any] | None = None,
        ordering: str | None = None,
        on_failure: str | None = None,
        invite_ack_timeout_s: float | None = None,
        required_acks: int | None = None,
        participant_role: ParticipantRole | None = None,
    ) -> SessionMetadata:
        """Allocate a session id, write metadata, and handshake invites.

        The creator is assigned :data:`ParticipantRole.INITIATOR`; every
        other participant is assigned ``participant_role`` (defaults to
        :data:`ParticipantRole.RESPONDENT` for two-party types and
        :data:`ParticipantRole.PARTICIPANT` for multi-party types). Order
        in ``participant_names`` is preserved and stamped onto each
        :class:`Participant` as its ``order`` index — discussion adapters
        use this for static turn-taking.

        ``required_acks`` controls the handshake quorum: ``None`` means
        every invited participant must ack; an integer N means the hub
        returns as soon as N have acked. If the remaining possible acks
        drop below the quorum target the future resolves with an
        :class:`InviteRejectedError`.
        """

        session_type_name = _type_name(session_type)
        if creator_id not in self._identities:
            raise UnknownActorError(creator_id)
        adapter = self._adapter_for(session_type_name)

        creator = self._identities[creator_id]
        creator_rule = self._rules[creator_id]
        if not creator_rule.access.session_types.may_initiate(session_type_name):
            raise AccessDeniedError(
                f"{creator.name} may not initiate {session_type_name} sessions"
            )

        if (
            len(self._active_sessions[creator_id])
            >= creator_rule.limits.max_concurrent_sessions
        ):
            raise LimitExceededError(
                f"{creator.name} is at max_concurrent_sessions={creator_rule.limits.max_concurrent_sessions}"
            )

        # Resolve participant ids. Preserve order so static / round_robin
        # discussions and auction bid order stay deterministic.
        other_ids: list[str] = []
        for name in participant_names:
            if name == creator.name:
                continue
            resolved = self._resolve_actor(name)
            if resolved not in other_ids:
                other_ids.append(resolved)

        if not other_ids:
            raise SessionTypeError(
                "session requires at least one non-creator participant"
            )

        # Cross-check rules on every direction.
        for other_id in other_ids:
            other_identity = self._identities[other_id]
            other_rule = self._rules[other_id]
            if not creator_rule.access.allows_outbound(other_identity.name):
                raise AccessDeniedError(
                    f"{creator.name} not allowed to reach {other_identity.name} (outbound rule)"
                )
            if not other_rule.access.allows_inbound(creator.name):
                raise AccessDeniedError(
                    f"{other_identity.name} not allowed to accept from {creator.name} (inbound rule)"
                )
            if not other_rule.access.session_types.may_accept(session_type_name):
                raise AccessDeniedError(
                    f"{other_identity.name} not allowed to accept {session_type_name}"
                )
            if (
                len(self._active_sessions[other_id])
                >= other_rule.limits.max_concurrent_sessions
            ):
                raise LimitExceededError(
                    f"{other_identity.name} is at max_concurrent_sessions"
                )

        role_for_others = participant_role or (
            ParticipantRole.RESPONDENT
            if len(other_ids) == 1
            else ParticipantRole.PARTICIPANT
        )
        participants = [
            Participant(
                actor_id=creator_id,
                role=ParticipantRole.INITIATOR,
                joined_at=self._clock(),
                order=0,
            ),
            *[
                Participant(
                    actor_id=oid,
                    role=role_for_others,
                    joined_at=self._clock(),
                    order=i + 1,
                )
                for i, oid in enumerate(other_ids)
            ],
        ]

        created_at = self._clock()
        expires_at = _add_seconds(created_at, creator_rule.limits.session_ttl_seconds())
        metadata = SessionMetadata(
            session_id=new_id(),
            type=session_type_name,
            creator_id=creator_id,
            participants=participants,
            state=SessionState.PENDING,
            created_at=created_at,
            expires_at=expires_at,
            labels=dict(labels or {}),
            ordering=ordering,
            on_failure=on_failure,
        )
        adapter.validate_create(metadata)

        await self._write_session_metadata(metadata)
        self._sessions[metadata.session_id] = metadata

        # Reserve the slot on every participant.
        for p in metadata.participants:
            self._active_sessions[p.actor_id].add(metadata.session_id)

        # Resolve the quorum target. ``None`` means all non-creator
        # participants must ack; otherwise clamp to ``[1, len(others)]``.
        quorum_target = (
            len(other_ids)
            if required_acks is None
            else max(1, min(required_acks, len(other_ids)))
        )
        pending = _PendingInvite(
            session_id=metadata.session_id,
            future=asyncio.get_event_loop().create_future(),
            required=quorum_target,
            pending_ids=set(other_ids),
        )
        self._pending_invites[metadata.session_id] = pending

        # Deliver the invite envelope(s) and wait for ack(s).
        for other_id in other_ids:
            invite = Envelope(
                session_id=metadata.session_id,
                sender_id=creator_id,
                recipient_id=other_id,
                event_type=EV_SESSION_INVITE,
                event_data={
                    "session_type": session_type_name,
                    "initiator_id": creator_id,
                    "initiator_name": creator.name,
                },
            )
            invite.envelope_id = new_id()
            invite.created_at = self._clock()
            async with self._wal_lock:
                await self._append_to_wal(metadata, invite)
            await self._deliver(invite)

        timeout_s = (
            invite_ack_timeout_s
            if invite_ack_timeout_s is not None
            else self.config.invite_ack_timeout_s
        )
        try:
            updated = await asyncio.wait_for(pending.future, timeout=timeout_s)
        except asyncio.TimeoutError:
            await self._rollback_session(metadata, reason="invite_timeout")
            self._audit(
                actor_id=creator_id,
                action="create_session",
                resource_type="session",
                resource_id=metadata.session_id,
                decision="timeout",
                reason="invite_timeout",
            )
            raise InviteRejectedError(
                f"only {len(pending.acked_ids)}/{quorum_target} invites acked "
                "before timeout"
            )
        except InviteRejectedError:
            await self._rollback_session(metadata, reason="invite_rejected")
            self._audit(
                actor_id=creator_id,
                action="create_session",
                resource_type="session",
                resource_id=metadata.session_id,
                decision="rejected",
                reason="invite_rejected",
            )
            raise
        finally:
            self._pending_invites.pop(metadata.session_id, None)

        self._audit(
            actor_id=creator_id,
            action="create_session",
            resource_type="session",
            resource_id=metadata.session_id,
            decision="allow",
            reason=session_type_name,
        )
        return updated

    async def _rollback_session(
        self, metadata: SessionMetadata, *, reason: str
    ) -> None:
        metadata = self._sessions.get(metadata.session_id, metadata)
        metadata.state = SessionState.CLOSED
        metadata.closed_at = self._clock()
        metadata.close_reason = reason
        self._sessions[metadata.session_id] = metadata
        await self._write_session_metadata(metadata)
        for p in metadata.participants:
            self._active_sessions[p.actor_id].discard(metadata.session_id)

    # ------------------------------------------------------------------
    # Posting envelopes
    # ------------------------------------------------------------------

    async def post_envelope(self, envelope: Envelope) -> tuple[str, int]:
        """Validate, persist, and deliver an envelope posted by an actor.

        Returns ``(envelope_id, wal_offset)`` where ``wal_offset`` is the
        session WAL byte offset immediately *after* this envelope's append —
        i.e. the position at which the next envelope will start. Callers
        that want to wait for a correlated reply open a subscription with
        ``since=wal_offset`` so they do not re-replay the full WAL on every
        ``Session.ask``.

        For system envelopes (invite ack/reject) the returned offset is 0;
        nothing subscribes with a cursor immediately after these.
        """

        metadata = self._sessions.get(envelope.session_id)
        if metadata is None:
            raise UnknownSessionError(envelope.session_id)
        if metadata.state not in (SessionState.ACTIVE, SessionState.PENDING):
            raise SessionClosedError(
                f"session {metadata.session_id} is in state {metadata.state.value}"
            )
        if envelope.sender_id not in self._identities:
            raise UnknownActorError(envelope.sender_id)
        if not metadata.has_participant(envelope.sender_id):
            raise AccessDeniedError("sender is not a participant in this session")

        # System envelopes (invite acks/rejects) have their own routing.
        if envelope.event_type in (
            EV_SESSION_INVITE_ACK,
            EV_SESSION_INVITE_REJECT,
        ):
            envelope_id = await self._handle_system_envelope(metadata, envelope)
            return envelope_id, 0

        # Event-type gate: a strict registry refuses unregistered
        # event types at post time so operators can enforce a closed
        # wire-format set. Permissive (default) registries accept
        # anything for forward compatibility.
        try:
            self._event_registry.check(envelope.event_type)
        except UnknownEventTypeError as exc:
            raise SessionTypeError(str(exc)) from exc

        sender_identity = self._identities[envelope.sender_id]
        sender_rule = self._rules[envelope.sender_id]

        # Rate limit (token bucket) — enforced per sender.
        if not self._rate_limiter.check_and_consume(
            envelope.sender_id, sender_rule.limits.rate
        ):
            raise LimitExceededError(
                f"{sender_identity.name} exceeded rate limit "
                f"({sender_rule.limits.rate.per_minute}/min)"
            )

        # Delegation depth — the envelope carries its own hop count so
        # a chain of A→B→C→… eventually stops. Callers are responsible
        # for incrementing ``depth`` on follow-up envelopes; the hub
        # only enforces the ceiling at post time.
        if (
            sender_rule.limits.delegation_depth > 0
            and envelope.depth > sender_rule.limits.delegation_depth
        ):
            raise LimitExceededError(
                f"{sender_identity.name} exceeded delegation_depth "
                f"({envelope.depth} > {sender_rule.limits.delegation_depth})"
            )

        if envelope.recipient_id is not None:
            recipient_identity = self._identities.get(envelope.recipient_id)
            if recipient_identity is None:
                raise UnknownActorError(envelope.recipient_id)
            if not sender_rule.access.allows_outbound(recipient_identity.name):
                raise AccessDeniedError(
                    f"{sender_identity.name} may not reach {recipient_identity.name}"
                )
            recipient_rule = self._rules[envelope.recipient_id]
            if not recipient_rule.access.allows_inbound(sender_identity.name):
                raise AccessDeniedError(
                    f"{recipient_identity.name} may not accept from {sender_identity.name}"
                )

        # Phase 4 — task-event branch. Task envelopes carry their own
        # hub-owned state machine; they bypass ``adapter.validate_send``
        # and ``adapter.on_accepted`` so the session-type lifecycle
        # (consulting's 1Q1R auto-close, discussion's speaker-order
        # rotation, etc.) stays orthogonal to task lifecycle. Access /
        # rate / depth / inbox checks above still apply — the bypass is
        # strictly about session-adapter delivery rules, not the
        # network rule surface.
        if envelope.event_type in TASK_EVENT_TYPES:
            return await self._process_actor_task_event(metadata, envelope)

        # User envelopes go through the adapter.
        prior = await self._read_user_envelopes(metadata.session_id)
        adapter = self._adapter_for(metadata.type)
        adapter.validate_send(metadata, envelope, prior)

        # Preflight the structured inbox: any recipient whose rule has
        # ``inbox.max_pending`` hit with ``overflow="reject"`` fails the
        # whole post before we touch the WAL. Spool-mode overflow falls
        # through to _deliver_to and is handled lazily there. This is
        # what makes the post_envelope atomicity hold — if delivery
        # fails for one recipient, the envelope is not persisted.
        self._preflight_inbox_capacity(metadata, envelope)

        envelope.envelope_id = new_id()
        envelope.created_at = self._clock()
        envelope.trace_id = envelope.trace_id or envelope.envelope_id

        # Appending to the WAL and snapshotting the live subscription set
        # happens atomically under _wal_lock. See _handle_subscribe for
        # the dual side of this invariant — why it prevents the
        # "delivered twice" and "delivered neither" races.
        async with self._wal_lock:
            next_offset = await self._append_to_wal(metadata, envelope)
            subs_snapshot = list(self._subscriptions.values())

        # Deliver the user envelope BEFORE applying any adapter state
        # transition — otherwise an adapter that closes the session on
        # this envelope will fan out a SessionClosed broadcast ahead of
        # the real content, surprising subscribers. When
        # ``recipient_id`` is None, the envelope is a session-wide
        # broadcast: fan-out to every non-sender participant.
        if envelope.recipient_id is None:
            await self._fanout_to_participants(metadata, envelope)
        else:
            await self._deliver(envelope)
        await self._fanout_to_subs(
            subs_snapshot, metadata, envelope, wal_offset=next_offset
        )

        result = adapter.on_accepted(metadata, envelope, prior)
        await self._apply_adapter_result(metadata, result)
        if result.follow_ups:
            for follow_up in result.follow_ups:
                await self._post_adapter_follow_up(metadata, follow_up)

        return envelope.envelope_id, next_offset

    async def _post_adapter_follow_up(
        self, metadata: SessionMetadata, envelope: Envelope
    ) -> None:
        """Append + deliver a hub-generated envelope produced by an adapter.

        Used by multi-participant adapters (discussion, auction) to emit
        turn-signal envelopes the hub has to persist and route even
        though no actor posted them.
        """

        if envelope.envelope_id is None:
            envelope.envelope_id = new_id()
        if envelope.created_at is None:
            envelope.created_at = self._clock()
        async with self._wal_lock:
            next_offset = await self._append_to_wal(metadata, envelope)
            subs_snapshot = list(self._subscriptions.values())
        await self._deliver(envelope)
        await self._fanout_to_subs(
            subs_snapshot, metadata, envelope, wal_offset=next_offset
        )

    async def _apply_adapter_result(
        self, metadata: SessionMetadata, result: AdapterResult
    ) -> None:
        changed = False
        closing = False
        if result.next_state is not None and metadata.state is not result.next_state:
            metadata.state = result.next_state
            if result.next_state is SessionState.CLOSED:
                metadata.closed_at = self._clock()
                metadata.close_reason = result.close_reason
                for p in metadata.participants:
                    self._active_sessions[p.actor_id].discard(metadata.session_id)
                closing = True
            changed = True
        if changed:
            await self._write_session_metadata(metadata)
            if closing:
                # Phase 4 — any tasks attached to this session transition
                # to ``cancelled`` before the SessionClosed broadcast so
                # subscribers see a clean task-terminal → session-terminal
                # ordering in the WAL. The cancel envelope emission
                # happens inside the same connection loop, so the
                # subsequent ``_broadcast_session_closed`` still runs
                # after the task-cancel envelopes have landed.
                await self._cancel_tasks_for_session(
                    metadata.session_id,
                    reason=result.close_reason or "session_closed",
                )
                await self._broadcast_session_closed(metadata)

    async def _handle_system_envelope(
        self, metadata: SessionMetadata, envelope: Envelope
    ) -> str:
        envelope.envelope_id = new_id()
        envelope.created_at = self._clock()
        async with self._wal_lock:
            await self._append_to_wal(metadata, envelope)

        if envelope.event_type == EV_SESSION_INVITE_ACK:
            await self._record_invite_ack(metadata, envelope)
        elif envelope.event_type == EV_SESSION_INVITE_REJECT:
            await self._record_invite_reject(metadata, envelope)

        return envelope.envelope_id

    async def _record_invite_ack(
        self, metadata: SessionMetadata, envelope: Envelope
    ) -> None:
        pending = self._pending_invites.get(metadata.session_id)
        if pending is None or pending.future.done():
            return

        acker = envelope.sender_id
        if acker not in pending.pending_ids:
            return
        pending.pending_ids.discard(acker)
        pending.acked_ids.add(acker)

        if len(pending.acked_ids) >= pending.required:
            metadata.state = SessionState.ACTIVE
            await self._write_session_metadata(metadata)
            opened = Envelope(
                session_id=metadata.session_id,
                sender_id="hub",
                event_type=EV_SESSION_OPENED,
                event_data={
                    "session_id": metadata.session_id,
                    "acked_ids": sorted(pending.acked_ids),
                    "rejected_ids": sorted(pending.rejected_ids),
                },
            )
            opened.envelope_id = new_id()
            opened.created_at = self._clock()
            async with self._wal_lock:
                opened_offset = await self._append_to_wal(metadata, opened)
                subs_snapshot = list(self._subscriptions.values())
            await self._fanout_to_subs(
                subs_snapshot, metadata, opened, wal_offset=opened_offset
            )
            pending.future.set_result(metadata)

    async def _record_invite_reject(
        self, metadata: SessionMetadata, envelope: Envelope
    ) -> None:
        pending = self._pending_invites.get(metadata.session_id)
        if pending is None or pending.future.done():
            return

        rejecter = envelope.sender_id
        reason = envelope.event_data.get("reason", "rejected")
        if rejecter not in pending.pending_ids:
            return
        pending.pending_ids.discard(rejecter)
        pending.rejected_ids[rejecter] = reason

        # If the remaining pending + already-acked cannot reach the
        # quorum, the handshake fails. Otherwise keep waiting.
        max_possible = len(pending.acked_ids) + len(pending.pending_ids)
        if max_possible < pending.required:
            pending.future.set_exception(
                InviteRejectedError(
                    f"invite rejected by {len(pending.rejected_ids)} participant(s): "
                    f"quorum {pending.required} unreachable — last reason: {reason}"
                )
            )

    async def _broadcast_session_closed(self, metadata: SessionMetadata) -> None:
        closed = Envelope(
            session_id=metadata.session_id,
            sender_id="hub",
            event_type=EV_SESSION_CLOSED,
            event_data={
                "session_id": metadata.session_id,
                "reason": metadata.close_reason,
            },
        )
        closed.envelope_id = new_id()
        closed.created_at = self._clock()
        async with self._wal_lock:
            closed_offset = await self._append_to_wal(metadata, closed)
            subs_snapshot = list(self._subscriptions.values())
        for participant in metadata.participants:
            closed_for = Envelope.from_dict(closed.to_dict())
            closed_for.recipient_id = participant.actor_id
            await self._deliver(closed_for)
        await self._fanout_to_subs(
            subs_snapshot, metadata, closed, wal_offset=closed_offset
        )

    # ------------------------------------------------------------------
    # Phase 4 — Network tasks
    # ------------------------------------------------------------------

    async def create_task(
        self,
        *,
        session_id: str,
        requester_id: str,
        owner_id: str,
        spec: TaskSpec,
        ttl_seconds: int | None = None,
    ) -> TaskMetadata:
        """Allocate a task inside ``session_id`` and emit the assignment.

        Direct hub method — symmetric with :meth:`create_session`. Allocates
        a UUID7 ``task_id``, writes ``hub/tasks/{task_id}/metadata.json``,
        writes a session-side back-reference, adds the task to the in-memory
        cache, and posts an ``ag2.task.assigned`` envelope addressed to the
        owner. The owner's :class:`ActorClient` sees the envelope through
        the normal notify flow and dispatches to a task handler.

        Raises:
            :class:`UnknownSessionError` if the session does not exist.
            :class:`SessionClosedError` if the session is not ACTIVE.
            :class:`AccessDeniedError` if ``requester_id`` or ``owner_id``
                is not a session participant.
            :class:`LimitExceededError` if the owner is at
                ``rule.limits.max_concurrent_tasks``.
        """

        session_metadata = self._sessions.get(session_id)
        if session_metadata is None:
            raise UnknownSessionError(session_id)
        if session_metadata.state is not SessionState.ACTIVE:
            raise SessionClosedError(
                f"session {session_id} is in state {session_metadata.state.value}"
            )
        if requester_id not in self._identities:
            raise UnknownActorError(requester_id)
        if owner_id not in self._identities:
            raise UnknownActorError(owner_id)
        if not session_metadata.has_participant(requester_id):
            raise AccessDeniedError(
                "task requester must be a participant in the session"
            )
        if not session_metadata.has_participant(owner_id):
            raise AccessDeniedError("task owner must be a participant in the session")

        owner_rule = self._rules[owner_id]
        # max_concurrent_tasks gates the owner's non-terminal task count.
        active_owner_tasks = sum(
            1
            for t in self._tasks.values()
            if t.owner_id == owner_id and not t.is_terminal()
        )
        if active_owner_tasks >= owner_rule.limits.max_concurrent_tasks:
            owner_identity = self._identities[owner_id]
            raise LimitExceededError(
                f"{owner_identity.name} is at "
                f"max_concurrent_tasks={owner_rule.limits.max_concurrent_tasks}"
            )

        task_id = new_id()
        created_at = self._clock()
        ttl = (
            ttl_seconds
            if ttl_seconds is not None
            else owner_rule.limits.task_ttl_seconds()
        )
        expires_at = _add_seconds(created_at, ttl)

        task = TaskMetadata(
            task_id=task_id,
            session_id=session_id,
            owner_id=owner_id,
            requester_id=requester_id,
            spec=spec,
            state=TaskState.CREATED,
            created_at=created_at,
            expires_at=expires_at,
            current_phase=None,
        )

        self._tasks[task_id] = task
        self._session_tasks.setdefault(session_id, set()).add(task_id)
        await self._write_task_metadata(task)
        await self._store.write(layout.session_task_ref(session_id, task_id), task_id)

        # Emit the ``ag2.task.assigned`` envelope addressed to the owner.
        # Fan-out to session subscribers happens through the same WAL
        # append / subs-snapshot invariant used by every other envelope.
        assigned = Envelope(
            session_id=session_id,
            sender_id="hub",
            recipient_id=owner_id,
            event_type=EV_TASK_ASSIGNED,
            event_data={
                "task_id": task_id,
                "spec": spec.to_dict(),
                "owner_id": owner_id,
                "requester_id": requester_id,
                "expires_at": expires_at,
            },
            task_id=task_id,
        )
        await self._post_hub_task_envelope(session_metadata, assigned)
        return task.copy()

    def peek_task(self, task_id: str) -> TaskMetadata | None:
        """Return the task metadata from the in-memory index or ``None``.

        Non-raising lookup for callers that need a best-effort read.
        Use :meth:`get_task` for an authoritative raising variant.
        """

        task = self._tasks.get(task_id)
        return task.copy() if task is not None else None

    async def get_task(self, task_id: str) -> TaskMetadata:
        task = self._tasks.get(task_id)
        if task is None:
            raise UnknownTaskError(task_id)
        return task.copy()

    def tasks_for_session(self, session_id: str) -> list[TaskMetadata]:
        """Return every task the hub knows about for ``session_id``.

        Includes terminal tasks (for post-mortem inspection); callers that
        only care about active tasks should filter on ``is_terminal``.
        """

        return [
            self._tasks[t].copy()
            for t in sorted(self._tasks)
            if self._tasks[t].session_id == session_id
        ]

    async def cancel_task(
        self,
        task_id: str,
        *,
        requested_by: str,
        reason: str = "",
    ) -> TaskMetadata:
        """Cancel ``task_id`` on behalf of ``requested_by``.

        Only the task's requester (the actor that called :meth:`create_task`)
        or the owner may cancel. Cancelling a task that is already in a
        terminal state is a no-op and returns the current metadata.
        """

        task = self._tasks.get(task_id)
        if task is None:
            raise UnknownTaskError(task_id)
        if requested_by not in (task.requester_id, task.owner_id):
            raise AccessDeniedError("only the task's requester or owner may cancel it")
        if task.is_terminal():
            return task.copy()

        session_metadata = self._sessions.get(task.session_id)
        if session_metadata is None:
            # The session was archived out from under an active task —
            # treat as cancelled locally without emitting an envelope
            # so the task record still reaches a terminal state.
            task.state = TaskState.CANCELLED
            task.completed_at = self._clock()
            task.error = reason or "session missing"
            await self._write_task_metadata(task)
            self._session_tasks.get(task.session_id, set()).discard(task_id)
            return task.copy()

        cancelled = Envelope(
            session_id=task.session_id,
            sender_id="hub",
            recipient_id=None,  # broadcast to every participant
            event_type=EV_TASK_CANCELLED,
            event_data={
                "task_id": task_id,
                "requested_by": requested_by,
                "reason": reason,
            },
            task_id=task_id,
        )
        await self._post_hub_task_envelope(session_metadata, cancelled)
        return self._tasks[task_id].copy()

    async def expire_due_tasks(self, *, now: str | None = None) -> list[str]:
        """TTL sweeper entry point — expire tasks past their deadline.

        Walks the in-memory task cache and transitions every non-terminal
        task whose ``expires_at`` is at or before ``now`` to
        :attr:`TaskState.EXPIRED`, emitting a broadcast ``ag2.task.expired``
        envelope so subscribers and session participants see the transition.
        Tasks already in a terminal state are skipped.

        ``now`` defaults to ``self._clock()`` but tests may pass an explicit
        ISO-Z string in the future to exercise the sweeper deterministically
        without driving real time.
        """

        threshold = now or self._clock()
        expired_ids: list[str] = []
        # Snapshot the task ids so a concurrent cancel/result during the
        # sweep does not mutate the dict under us.
        for task_id in list(self._tasks):
            task = self._tasks.get(task_id)
            if task is None or task.is_terminal():
                continue
            if task.expires_at > threshold:
                continue
            session_metadata = self._sessions.get(task.session_id)
            if session_metadata is None:
                # Orphaned task — transition locally without emitting.
                task.state = TaskState.EXPIRED
                task.completed_at = threshold
                await self._write_task_metadata(task)
                self._session_tasks.get(task.session_id, set()).discard(task_id)
                expired_ids.append(task_id)
                continue
            expired = Envelope(
                session_id=task.session_id,
                sender_id="hub",
                recipient_id=None,  # broadcast
                event_type=EV_TASK_EXPIRED,
                event_data={
                    "task_id": task_id,
                    "expires_at": task.expires_at,
                    "expired_at": threshold,
                },
                task_id=task_id,
            )
            await self._post_hub_task_envelope(session_metadata, expired)
            expired_ids.append(task_id)
        return expired_ids

    async def _process_actor_task_event(
        self, session_metadata: SessionMetadata, envelope: Envelope
    ) -> tuple[str, int]:
        """Process a task envelope an actor posted via ``SendFrame``.

        Called from the task-event branch in :meth:`post_envelope`. Applies
        the hub's task state machine, appends to the session WAL, delivers
        to recipients (unicast or broadcast), fans out to subscribers, and
        rewrites ``hub/tasks/{task_id}/metadata.json``. Returns the stamped
        ``(envelope_id, wal_offset)`` pair like the user envelope path.
        """

        task_id = envelope.task_id
        if task_id is None:
            raise TaskStateError("task envelope must carry a ``task_id`` field")
        task = self._tasks.get(task_id)
        if task is None:
            raise UnknownTaskError(task_id)
        if task.session_id != session_metadata.session_id:
            raise TaskStateError(
                f"task {task_id} belongs to session {task.session_id}, "
                f"not {session_metadata.session_id}"
            )

        self._validate_actor_task_event(task, envelope)
        self._preflight_inbox_capacity(session_metadata, envelope)

        envelope.envelope_id = new_id()
        envelope.created_at = self._clock()
        envelope.trace_id = envelope.trace_id or envelope.envelope_id

        async with self._wal_lock:
            next_offset = await self._append_to_wal(session_metadata, envelope)
            subs_snapshot = list(self._subscriptions.values())

        if envelope.recipient_id is None:
            await self._fanout_to_participants(session_metadata, envelope)
        else:
            await self._deliver(envelope)
        await self._fanout_to_subs(
            subs_snapshot, session_metadata, envelope, wal_offset=next_offset
        )

        await self._apply_task_event(task, envelope)
        return envelope.envelope_id, next_offset

    async def _post_hub_task_envelope(
        self, session_metadata: SessionMetadata, envelope: Envelope
    ) -> None:
        """Persist and deliver a hub-emitted task envelope.

        Hub-emitted task events (``ag2.task.assigned`` / ``cancelled`` /
        ``expired``) follow the same WAL + delivery + fan-out path that
        actor-posted envelopes do, but there is no sender-side rule check
        because the hub is authoritative. After delivery the task state
        transition is applied so the in-memory and on-disk state stay
        consistent with what subscribers just saw.
        """

        if envelope.envelope_id is None:
            envelope.envelope_id = new_id()
        if envelope.created_at is None:
            envelope.created_at = self._clock()
        envelope.trace_id = envelope.trace_id or envelope.envelope_id

        async with self._wal_lock:
            next_offset = await self._append_to_wal(session_metadata, envelope)
            subs_snapshot = list(self._subscriptions.values())

        if envelope.recipient_id is None:
            await self._fanout_to_participants(session_metadata, envelope)
        else:
            await self._deliver(envelope)
        await self._fanout_to_subs(
            subs_snapshot, session_metadata, envelope, wal_offset=next_offset
        )

        task_id = envelope.task_id
        if task_id is not None:
            task = self._tasks.get(task_id)
            if task is not None:
                await self._apply_task_event(task, envelope)

    def _validate_actor_task_event(
        self, task: TaskMetadata, envelope: Envelope
    ) -> None:
        """Check sender authority and state-machine legality.

        Actor-posted task events are:

        * ``phase_entered`` / ``phase_completed`` / ``progress`` —
          owner-only while the task is ``created`` or ``running``.
        * ``result`` / ``error`` — owner-only while the task is
          ``created`` or ``running``. Transitions to a terminal state.
        * ``assigned`` / ``cancelled`` / ``expired`` — hub-only. An
          actor cannot post these; the hub rejects them here.
        """

        event_type = envelope.event_type
        sender_id = envelope.sender_id

        if event_type in (EV_TASK_ASSIGNED, EV_TASK_CANCELLED, EV_TASK_EXPIRED):
            raise TaskStateError(
                f"{event_type!r} is hub-emitted only; actors cannot post it"
            )

        if sender_id != task.owner_id:
            raise AccessDeniedError(f"only the task owner may post {event_type!r}")

        if task.is_terminal():
            raise TaskStateError(
                f"task {task.task_id} is already terminal "
                f"(state={task.state.value}); "
                f"cannot post {event_type!r}"
            )

        if event_type == EV_TASK_PHASE_ENTERED:
            phase_id = envelope.event_data.get("phase_id")
            if not isinstance(phase_id, str) or not phase_id:
                raise TaskStateError(
                    "ag2.task.phase_entered requires event_data.phase_id (non-empty string)"
                )
            # If the task has a declared phase plan, the phase id must
            # match one of the declared phases. Ad-hoc phases are
            # allowed when the task was created without a phase plan.
            declared = task.spec.phase_ids()
            if declared and phase_id not in declared:
                raise TaskStateError(
                    f"phase {phase_id!r} is not in the task's declared phase plan "
                    f"({declared})"
                )
        elif event_type == EV_TASK_PHASE_COMPLETED:
            phase_id = envelope.event_data.get("phase_id")
            if not isinstance(phase_id, str) or not phase_id:
                raise TaskStateError(
                    "ag2.task.phase_completed requires event_data.phase_id (non-empty string)"
                )
            declared = task.spec.phase_ids()
            if declared and phase_id not in declared:
                raise TaskStateError(
                    f"phase {phase_id!r} is not in the task's declared phase plan "
                    f"({declared})"
                )

    async def _apply_task_event(self, task: TaskMetadata, envelope: Envelope) -> None:
        """Mutate ``task`` in place based on the event type, then persist."""

        event_type = envelope.event_type
        data = envelope.event_data
        now = self._clock()

        if event_type == EV_TASK_ASSIGNED:
            # Assignment itself does not advance state — the task stays
            # ``created`` until the owner emits its first progress /
            # phase / result event.
            pass
        elif event_type == EV_TASK_PHASE_ENTERED:
            if task.state is TaskState.CREATED:
                task.state = TaskState.RUNNING
                task.started_at = now
            phase_id = str(data.get("phase_id", ""))
            task.current_phase = phase_id or task.current_phase
            self._stamp_phase(task, phase_id, started_at=now)
        elif event_type == EV_TASK_PHASE_COMPLETED:
            phase_id = str(data.get("phase_id", ""))
            self._stamp_phase(task, phase_id, completed_at=now)
        elif event_type == EV_TASK_PROGRESS:
            if task.state is TaskState.CREATED:
                task.state = TaskState.RUNNING
                task.started_at = now
            task.last_progress_at = now
            update = data.get("update")
            if isinstance(update, dict):
                task.progress.update(update)
        elif event_type == EV_TASK_RESULT:
            task.state = TaskState.COMPLETED
            task.completed_at = now
            task.result = data.get("value")
            self._session_tasks.get(task.session_id, set()).discard(task.task_id)
            self._tasks_completed_total += 1
        elif event_type == EV_TASK_ERROR:
            task.state = TaskState.FAILED
            task.completed_at = now
            task.error = str(data.get("error", ""))
            self._session_tasks.get(task.session_id, set()).discard(task.task_id)
            self._tasks_failed_total += 1
        elif event_type == EV_TASK_CANCELLED:
            task.state = TaskState.CANCELLED
            task.completed_at = now
            reason = data.get("reason")
            if isinstance(reason, str) and reason:
                task.error = reason
            self._session_tasks.get(task.session_id, set()).discard(task.task_id)
        elif event_type == EV_TASK_EXPIRED:
            task.state = TaskState.EXPIRED
            task.completed_at = now
            self._session_tasks.get(task.session_id, set()).discard(task.task_id)

        await self._write_task_metadata(task)

    def _stamp_phase(
        self,
        task: TaskMetadata,
        phase_id: str,
        *,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> None:
        """Update the matching :class:`TaskPhase` on ``task.spec.phases``.

        Ad-hoc phases (phases not declared in the original spec) are
        silently ignored — the hub only tracks timestamps on declared
        phases. This keeps the spec dataclass immutable from the actor's
        perspective except for the two timestamp fields.
        """

        if not phase_id:
            return
        for phase in task.spec.phases:
            if phase.id == phase_id:
                if started_at is not None:
                    phase.started_at = started_at
                if completed_at is not None:
                    phase.completed_at = completed_at
                return

    async def _write_task_metadata(self, task: TaskMetadata) -> None:
        await self._store.write(layout.task_metadata(task.task_id), task.to_json())

    async def _cancel_tasks_for_session(self, session_id: str, *, reason: str) -> None:
        """Cancel every non-terminal task in a closing session.

        Invoked from :meth:`close_session` and the adapter-driven
        close path so task cleanup cannot leak past session lifetime.
        """

        active = list(self._session_tasks.get(session_id, set()))
        for task_id in active:
            task = self._tasks.get(task_id)
            if task is None or task.is_terminal():
                continue
            session_metadata = self._sessions.get(session_id)
            if session_metadata is None:
                task.state = TaskState.CANCELLED
                task.completed_at = self._clock()
                task.error = reason
                await self._write_task_metadata(task)
                continue
            cancelled = Envelope(
                session_id=session_id,
                sender_id="hub",
                recipient_id=None,
                event_type=EV_TASK_CANCELLED,
                event_data={
                    "task_id": task_id,
                    "requested_by": "hub",
                    "reason": reason,
                },
                task_id=task_id,
            )
            await self._post_hub_task_envelope(session_metadata, cancelled)

    # ------------------------------------------------------------------
    # Session queries + close
    # ------------------------------------------------------------------

    async def get_session(self, session_id: str) -> SessionMetadata:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise UnknownSessionError(session_id) from exc

    def peek_session(self, session_id: str) -> SessionMetadata | None:
        """Return the session metadata from the in-memory index or ``None``.

        Non-raising lookup for callers that need a best-effort read
        (e.g. client-side dispatch deciding which handler to run). For
        authoritative reads use :meth:`get_session`.
        """

        return self._sessions.get(session_id)

    async def peek_inbox(
        self, actor_id: str, *, limit: int = 50
    ) -> list[Envelope]:
        """Return the actor's pending inbox envelopes.

        Reads ``hub/actors/{actor_id}/inbox/pending/`` and decodes each
        ``{envelope_id}.json`` payload into an :class:`Envelope`. The
        list is sorted by envelope id (UUID7 lexicographic = arrival
        order) and capped at ``limit``. Used by the Phase 6 ``listen``
        verb in ``scope="inbox"`` mode and by operator tooling that
        wants to introspect what a slow recipient still has to chew
        through.

        Returns an empty list if the actor has no pending entries (or
        was never registered) — non-raising on purpose.
        """

        if actor_id not in self._identities:
            return []
        pending_dir = layout.actor_inbox_pending_dir(actor_id) + "/"
        try:
            entries = await self._store.list(pending_dir)
        except Exception:  # pragma: no cover — store backends differ
            log.warning("peek_inbox list failed for %s", actor_id, exc_info=True)
            return []

        json_entries = sorted(e for e in entries if e.endswith(".json"))
        envelopes: list[Envelope] = []
        for name in json_entries[:limit]:
            payload = await self._store.read(f"{pending_dir}{name}")
            if payload is None:
                continue
            try:
                envelopes.append(Envelope.from_json(payload))
            except Exception:  # pragma: no cover — corrupt entry
                log.warning("peek_inbox decode failed for %s/%s", actor_id, name)
                continue
        return envelopes

    def metrics(self) -> dict[str, Any]:
        """Return a nested counter dict for ``GET /v1/admin/metrics``.

        Computed purely from in-memory state — zero audit-log scans,
        zero store reads. Safe to call at arbitrary Prometheus scrape
        frequency. Phase 3b §14 says metrics **must not** derive from
        ``audit.jsonl``; the two surfaces are intentionally
        independent so the audit file can be rotated externally
        without affecting the liveness view.

        Shape::

            {
              "actors":   {"registered": int, "connected": int},
              "sessions": {
                  "active":       int,  # active + pending slots held
                  "pending":      int,  # in-handshake
                  "closed_total": int,  # monotonic since hub open
              },
              "tasks": {
                  "running":         int,
                  "completed_total": int,
                  "failed_total":    int,
              },
              "inbox": {"pending_total": int},
              "uptime_s": float,
            }
        """

        active = 0
        pending = 0
        for meta in self._sessions.values():
            if meta.state is SessionState.ACTIVE:
                active += 1
            elif meta.state is SessionState.PENDING:
                pending += 1

        running_tasks = sum(
            1 for t in self._tasks.values() if not t.is_terminal()
        )

        pending_total = sum(self._inbox_pending.values())

        connected = sum(
            1 for ep in self._endpoints.values() if not ep.closed
        )

        return {
            "actors": {
                "registered": len(self._identities),
                "connected": connected,
            },
            "sessions": {
                "active": active,
                "pending": pending,
                "closed_total": self._sessions_closed_total,
            },
            "tasks": {
                "running": running_tasks,
                "completed_total": self._tasks_completed_total,
                "failed_total": self._tasks_failed_total,
            },
            "inbox": {"pending_total": pending_total},
            "uptime_s": _iso_seconds_since(self._started_at, self._clock()),
        }

    async def close_session(
        self,
        session_id: str,
        *,
        reason: str = "explicit",
        requested_by: str | None = None,
    ) -> None:
        metadata = await self.get_session(session_id)
        if metadata.state in (SessionState.CLOSED, SessionState.EXPIRED):
            return
        if requested_by is not None and not metadata.has_participant(requested_by):
            raise AccessDeniedError("only session participants may close a session")

        metadata.state = SessionState.CLOSED
        metadata.closed_at = self._clock()
        metadata.close_reason = reason
        await self._write_session_metadata(metadata)
        for p in metadata.participants:
            self._active_sessions[p.actor_id].discard(session_id)
        # Phase 4 — task cancel cascade before the SessionClosed broadcast.
        # Subscribers see TaskCancelled envelopes in the WAL ahead of the
        # terminal SessionClosed so post-mortem replay stays ordered.
        await self._cancel_tasks_for_session(session_id, reason=reason)
        await self._broadcast_session_closed(metadata)
        self._sessions_closed_total += 1
        self._audit(
            actor_id=requested_by,
            action="close_session",
            resource_type="session",
            resource_id=session_id,
            decision="allow",
            reason=reason,
        )

    async def sweep_expired_sessions(self) -> list[str]:
        """Transition every ACTIVE session whose ``expires_at`` has passed to EXPIRED.

        Phase 3a Step 8: this is the worker the TTL sweeper runs on a
        framework-core ``IntervalWatch`` callback. A dedicated
        :class:`Scheduler` is not baked into ``Hub`` — operators own
        the scheduler and register the callback themselves — so that
        ``Hub`` stays network-layer only and the scheduler is a pure
        lifecycle manager (see :mod:`autogen.beta.scheduler` and
        Phase 3a Step 0).

        Returns the list of session ids that were expired on this
        pass so callers can log / emit metrics. Safe to call
        concurrently with other hub operations — each expiring
        session goes through the normal close path, which holds the
        WAL lock while emitting the terminal ``SessionClosed``
        broadcast.

        An expired session behaves almost identically to a normally
        closed one: its participants lose the active-slot reservation,
        tasks are cancelled cascade, and a close broadcast fans out.
        The only distinction is the terminal state (``EXPIRED``
        instead of ``CLOSED``) and the ``close_reason="ttl_expired"``
        marker so post-mortem tooling can tell a TTL death from an
        explicit close.
        """

        now = self._clock()
        to_expire: list[SessionMetadata] = []
        for metadata in list(self._sessions.values()):
            if metadata.state is not SessionState.ACTIVE:
                continue
            if metadata.expires_at is None:
                continue
            if metadata.expires_at > now:
                continue
            to_expire.append(metadata)

        expired_ids: list[str] = []
        for metadata in to_expire:
            try:
                metadata.state = SessionState.EXPIRED
                metadata.closed_at = now
                metadata.close_reason = "ttl_expired"
                await self._write_session_metadata(metadata)
                for p in metadata.participants:
                    self._active_sessions[p.actor_id].discard(metadata.session_id)
                await self._cancel_tasks_for_session(
                    metadata.session_id, reason="ttl_expired"
                )
                await self._broadcast_session_closed(metadata)
                expired_ids.append(metadata.session_id)
                self._sessions_closed_total += 1
                self._audit(
                    actor_id=None,
                    action="expire_session",
                    resource_type="session",
                    resource_id=metadata.session_id,
                    decision="allow",
                    reason="ttl_expired",
                )
            except Exception:  # pragma: no cover
                log.warning(
                    "hub: ttl sweep failed for session %s",
                    metadata.session_id,
                    exc_info=True,
                )

        return expired_ids

    async def archive_closed_sessions(
        self,
        *,
        age_threshold_s: float = 0.0,
        now: str | None = None,
    ) -> list[str]:
        """Move closed/expired sessions older than ``age_threshold_s`` to archive.

        Walks every session whose state is :attr:`SessionState.CLOSED`
        or :attr:`SessionState.EXPIRED`, whose ``archived_at`` is still
        ``None`` (so we never re-archive an already-archived session),
        and whose ``closed_at`` is at least ``age_threshold_s`` older
        than ``now``. For each match:

        1. Read the session WAL via the store.
        2. Compute a compact summary dict: envelope count, first and
           last ``created_at``, distinct senders, close reason.
        3. Write the summary to
           ``hub/archive/sessions/{id}/summary.json``.
        4. Copy the WAL bytes to
           ``hub/archive/sessions/{id}/wal.jsonl``.
        5. Delete the live WAL at ``hub/sessions/{id}/wal.jsonl``.
        6. Rewrite ``hub/sessions/{id}/metadata.json`` with
           ``archived_at = now`` so hydrate skips it on restart.

        Framework-core ``CompactStrategy`` is *not* reused here
        because its surface (``BaseEvent`` + ``Context`` + per-turn
        token budget) is oriented at mid-stream LLM compaction, not
        cold archival of past sessions. A deliberate simplification
        for the OSS framework: the sweeper computes a plain-text
        summary that any backend can serve via
        ``GET /v1/actors/{id}/activity`` without requiring an LLM
        call. Operators who want richer summaries subclass the hub
        and override this method.

        The sweeper is idempotent: an already-archived session is
        skipped because ``archived_at is not None``. Active sessions
        are never touched.

        Returns the list of session ids archived on this pass.

        Args:
            age_threshold_s: Minimum seconds elapsed between
                ``closed_at`` and ``now``. ``0.0`` archives every
                closed session immediately — useful for tests.
            now: Optional ISO-Z timestamp override. Defaults to
                :attr:`self._clock()`. Tests use this to drive the
                sweeper deterministically without sleeping.
        """

        current = now if now is not None else self._clock()
        archived: list[str] = []
        for metadata in list(self._sessions.values()):
            if metadata.state not in (SessionState.CLOSED, SessionState.EXPIRED):
                continue
            if metadata.archived_at is not None:
                continue
            if metadata.closed_at is None:
                # Should not happen — close_session always stamps
                # closed_at — but guard against legacy data.
                continue
            age = _iso_seconds_since(metadata.closed_at, current)
            if age < age_threshold_s:
                continue

            try:
                await self._archive_single_session(metadata, now=current)
                archived.append(metadata.session_id)
                self._audit(
                    actor_id=None,
                    action="archive_session",
                    resource_type="session",
                    resource_id=metadata.session_id,
                    decision="allow",
                    reason=f"age_s={age:.1f}",
                )
            except Exception:  # pragma: no cover
                log.warning(
                    "archive sweep failed for session %s",
                    metadata.session_id,
                    exc_info=True,
                )
        return archived

    async def _archive_single_session(
        self, metadata: SessionMetadata, *, now: str
    ) -> None:
        """Archive one closed session. Called by :meth:`archive_closed_sessions`."""

        session_id = metadata.session_id
        wal_path = layout.session_wal(session_id)
        archive_dir = layout.archive_session_dir(session_id)
        archive_wal = f"{archive_dir}/wal.jsonl"
        archive_summary = f"{archive_dir}/summary.json"

        # Read the live WAL.
        raw_wal = await self._store.read(wal_path)
        envelope_count = 0
        first_ts: str | None = None
        last_ts: str | None = None
        senders: set[str] = set()
        if raw_wal:
            lines = [line for line in raw_wal.split("\n") if line]
            envelope_count = len(lines)
            if lines:
                try:
                    first = Envelope.from_json(lines[0])
                    first_ts = first.created_at
                    senders.add(first.sender_id)
                except Exception:  # pragma: no cover
                    pass
                try:
                    last = Envelope.from_json(lines[-1])
                    last_ts = last.created_at
                    senders.add(last.sender_id)
                except Exception:  # pragma: no cover
                    pass
                for line in lines[1:-1]:
                    try:
                        env = Envelope.from_json(line)
                        senders.add(env.sender_id)
                    except Exception:  # pragma: no cover
                        continue

        summary = {
            "session_id": session_id,
            "type": metadata.type,
            "state": metadata.state.value,
            "close_reason": metadata.close_reason,
            "closed_at": metadata.closed_at,
            "envelope_count": envelope_count,
            "first_envelope_at": first_ts,
            "last_envelope_at": last_ts,
            "participants": [p.actor_id for p in metadata.participants],
            "senders": sorted(senders),
            "archived_at": now,
        }
        await self._store.write(archive_summary, json.dumps(summary, sort_keys=True))
        if raw_wal:
            await self._store.write(archive_wal, raw_wal)
            await self._store.delete(wal_path)

        metadata.archived_at = now
        await self._write_session_metadata(metadata)

    def archive_sweep_callback(
        self, *, age_threshold_s: float = 3600.0
    ) -> Callable[[Any, Any], Any]:
        """Return an async Scheduler callback for periodic archival.

        Usage::

            scheduler = Scheduler()
            scheduler.add(
                IntervalWatch(60),
                callback=hub.archive_sweep_callback(age_threshold_s=3600),
            )

        Mirrors :meth:`ttl_sweep_callback` — the operator owns the
        scheduler and the hub contributes a plain async callable.
        """

        async def _callback(_events: Any, _ctx: Any) -> None:
            await self.archive_closed_sessions(age_threshold_s=age_threshold_s)

        return _callback

    def ttl_sweep_callback(self) -> Callable[[Any, Any], Any]:
        """Return an async callback compatible with ``Scheduler.add``.

        Usage::

            from autogen.beta.scheduler import Scheduler
            from autogen.beta.watch import IntervalWatch

            scheduler = Scheduler()
            scheduler.add(IntervalWatch(30), callback=hub.ttl_sweep_callback())
            await scheduler.start()

        The callback signature matches the standalone ``Scheduler``
        contract ``(events, ctx) -> Awaitable[None]`` — it ignores
        its arguments and simply invokes :meth:`sweep_expired_sessions`.
        This keeps ``Hub`` decoupled from ``Scheduler`` (the operator
        owns both instances) while giving a one-line wiring path for
        the common case.
        """

        async def _callback(events: Any, ctx: Any) -> None:
            try:
                await self.sweep_expired_sessions()
            except Exception:  # pragma: no cover
                log.warning("hub.ttl_sweep_callback failed", exc_info=True)

        return _callback

    async def read_wal(self, session_id: str, *, since: int = 0) -> list[Envelope]:
        wal = await self._store.read_range(layout.session_wal(session_id), since)
        envelopes: list[Envelope] = []
        for line in wal.split("\n"):
            if not line:
                continue
            envelopes.append(Envelope.from_json(line))
        return envelopes

    async def _read_wal_with_offsets(
        self, session_id: str, *, since: int = 0
    ) -> list[tuple[Envelope, int]]:
        """Read the WAL and return ``(envelope, end_offset)`` tuples.

        Used by :meth:`_handle_subscribe`'s initial replay — the client
        needs each envelope's end-of-record byte offset to pin its
        cursor correctly, so a subsequent reconnect with ``since=<end>``
        resumes exactly after the envelope just delivered. The WAL
        format is one ``Envelope.to_json() + '\\n'`` per line, so the
        end offset is the running sum of line byte lengths (+1 per
        newline separator).
        """

        wal = await self._store.read_range(layout.session_wal(session_id), since)
        entries: list[tuple[Envelope, int]] = []
        if not wal:
            return entries
        offset = since
        for line in wal.split("\n"):
            if not line:
                continue
            envelope = Envelope.from_json(line)
            offset += len(line.encode("utf-8")) + 1  # +1 for '\n'
            entries.append((envelope, offset))
        return entries

    # ------------------------------------------------------------------
    # WAL + inbox helpers
    # ------------------------------------------------------------------

    async def _append_to_wal(
        self, metadata: SessionMetadata, envelope: Envelope
    ) -> int:
        """Append ``envelope`` to the session WAL. Returns the next-write offset.

        Must be called while holding ``_wal_lock`` so the append and any
        companion subscription snapshot stay consistent.
        """

        payload = envelope.to_json() + "\n"
        offset = await self._store.append(
            layout.session_wal(metadata.session_id), payload
        )
        return offset + len(payload.encode("utf-8"))

    async def _read_user_envelopes(self, session_id: str) -> list[Envelope]:
        """Return the WAL slice the session adapter sees as "prior".

        Excludes:

        * ``ag2.session.*`` system envelopes (handshake, opened, closed)
        * ``ag2.error`` envelopes
        * ``ag2.task.*`` envelopes — task lifecycle is hub-owned and
          must stay orthogonal to session-adapter delivery rules
          (§6.2). Without this filter a consulting session that hosts
          a single ``run_task`` would count the assigned + result
          envelopes against the 1Q1R rule and refuse the actor's
          subsequent text reply, breaking the design's "task envelopes
          do not exhaust the session" invariant.
        """

        raw = await self._store.read(layout.session_wal(session_id))
        if not raw:
            return []
        envelopes: list[Envelope] = []
        for line in raw.split("\n"):
            if not line:
                continue
            env = Envelope.from_json(line)
            if (
                env.event_type.startswith("ag2.session.")
                or env.event_type.startswith("ag2.error")
                or env.event_type.startswith("ag2.task.")
            ):
                continue
            envelopes.append(env)
        return envelopes

    async def _write_session_metadata(self, metadata: SessionMetadata) -> None:
        await self._store.write(
            layout.session_metadata(metadata.session_id), metadata.to_json()
        )

    async def _deliver(self, envelope: Envelope) -> None:
        """Deliver an envelope to one recipient via inbox + notify frame.

        For 1:1 envelopes the recipient is named explicitly. Broadcast
        envelopes call :meth:`_fanout_to_participants` instead — this
        method only handles the directed case.
        """

        if envelope.recipient_id is None:
            return
        await self._deliver_to(envelope.recipient_id, envelope)

    async def _deliver_to(self, recipient_id: str, envelope: Envelope) -> None:
        """Persist ``envelope`` in the recipient's structured inbox and notify.

        Writes the envelope to ``hub/actors/{id}/inbox/pending/{envelope_id}.json``
        under the normal path, or ``inbox/overflow/{envelope_id}.json`` when
        the recipient's ``rule.limits.inbox`` is at ``max_pending`` with an
        ``overflow="spool"`` policy. The notify frame is only pushed for
        pending-path deliveries — spooled envelopes wait for the recipient
        to drain them explicitly.

        **System envelopes bypass the structured inbox.** ``ag2.session.*``
        handshake signals (invites, acks, opens, closes) are ephemeral:
        a disconnected actor that comes back later does not want a
        stale invite replayed from disk — the session's own state
        machine authoritatively tracks handshake progress. They are
        delivered via ``notify`` only, never written to ``pending/``,
        and therefore never count against ``max_pending``. This is
        what keeps user workload budget (the 1000-pending-envelope
        limit) separate from handshake chatter.

        Assumes :meth:`_preflight_inbox_capacity` has already rejected
        the envelope if any recipient's rule is ``reject``-mode full, so
        ``_deliver_to`` never raises :class:`InboxFullError` itself.
        """

        # System envelopes: notify-only, no structured-inbox slot.
        if envelope.event_type.startswith("ag2.session."):
            endpoint = self._endpoints.get(recipient_id)
            if endpoint is not None and not endpoint.closed:
                try:
                    await endpoint.send_frame(NotifyFrame(envelope=envelope))
                except Exception:  # pragma: no cover
                    log.warning("failed to push notify frame", exc_info=True)
            return

        rule = self._rules.get(recipient_id)
        inbox = rule.limits.inbox if rule is not None else None

        # Determine delivery disposition from overflow policy + capacity.
        # "normal" writes to pending/ and bumps the counter;
        # "spool" writes to overflow/ without bumping; "drop" writes nothing.
        disposition = "normal"
        if inbox is not None and inbox.max_pending > 0:
            current = self._inbox_pending.get(recipient_id, 0)
            if current >= inbox.max_pending:
                if inbox.overflow == "spool":
                    disposition = "spool"
                elif inbox.overflow == "drop_oldest":
                    await self._evict_oldest_pending(recipient_id)
                    # disposition stays "normal" — the eviction freed a slot
                elif inbox.overflow == "drop_newest":
                    disposition = "drop"
                # "reject" was already raised in _preflight_inbox_capacity

        if disposition == "drop":
            recipient_name = (
                self._identities[recipient_id].name
                if recipient_id in self._identities
                else recipient_id
            )
            log.info(
                "inbox drop_newest: dropped envelope %s for %s (%d/%d)",
                envelope.envelope_id,
                recipient_name,
                self._inbox_pending.get(recipient_id, 0),
                inbox.max_pending if inbox is not None else 0,
            )
            self._audit(
                actor_id=recipient_id,
                action="inbox_drop_newest",
                resource_type="envelope",
                resource_id=envelope.envelope_id or "",
                decision="drop",
                reason="inbox_full",
                trace_id=envelope.trace_id,
            )
            return

        if disposition == "spool":
            path = layout.actor_inbox_overflow(
                recipient_id, envelope.envelope_id or ""
            )
        else:
            path = layout.actor_inbox_pending(
                recipient_id, envelope.envelope_id or ""
            )

        await self._store.write(path, envelope.to_json())

        if disposition == "normal":
            self._inbox_pending[recipient_id] = (
                self._inbox_pending.get(recipient_id, 0) + 1
            )
            endpoint = self._endpoints.get(recipient_id)
            if endpoint is not None and not endpoint.closed:
                try:
                    await endpoint.send_frame(NotifyFrame(envelope=envelope))
                except Exception:  # pragma: no cover
                    log.warning("failed to push notify frame", exc_info=True)

    async def _evict_oldest_pending(self, recipient_id: str) -> None:
        """Evict the oldest pending envelope from a recipient's inbox.

        Used by the ``drop_oldest`` overflow policy. Because envelope ids
        are UUID7s (lexicographically time-sorted), ``sorted(...)[0]``
        is the oldest pending envelope. Decrements the in-memory counter
        and writes an audit entry. Safe to call when the inbox is empty —
        in that case it is a no-op.
        """

        pending_dir = layout.actor_inbox_pending_dir(recipient_id)
        try:
            entries = await self._store.list(pending_dir)
        except Exception:  # pragma: no cover
            log.warning(
                "drop_oldest: could not list pending/ for %s",
                recipient_id,
                exc_info=True,
            )
            return
        file_entries = sorted(e for e in entries if not e.endswith("/"))
        if not file_entries:
            return
        oldest_name = file_entries[0]
        full_path = f"{pending_dir}/{oldest_name}"
        try:
            await self._store.delete(full_path)
        except Exception:  # pragma: no cover
            log.warning(
                "drop_oldest: could not delete %s", full_path, exc_info=True
            )
            return
        current = self._inbox_pending.get(recipient_id, 0)
        if current > 0:
            self._inbox_pending[recipient_id] = current - 1
        evicted_envelope_id = oldest_name.removesuffix(".json")
        recipient_name = (
            self._identities[recipient_id].name
            if recipient_id in self._identities
            else recipient_id
        )
        log.info(
            "inbox drop_oldest: evicted %s from %s",
            evicted_envelope_id,
            recipient_name,
        )
        self._audit(
            actor_id=recipient_id,
            action="inbox_drop_oldest",
            resource_type="envelope",
            resource_id=evicted_envelope_id,
            decision="evict",
            reason="inbox_full",
            trace_id=None,
        )

    def _recipient_ids_for(
        self, metadata: SessionMetadata, envelope: Envelope
    ) -> list[str]:
        """Return the list of actor ids the envelope will be delivered to.

        Unicast envelopes ship to their ``recipient_id``; broadcast
        envelopes fan out to every non-sender participant. Used both by
        :meth:`_preflight_inbox_capacity` and the delivery code.
        """

        if envelope.recipient_id is not None:
            return [envelope.recipient_id]
        return [
            p.actor_id
            for p in metadata.participants
            if p.actor_id != envelope.sender_id
        ]

    def _preflight_inbox_capacity(
        self, metadata: SessionMetadata, envelope: Envelope
    ) -> None:
        """Raise :class:`InboxFullError` if any recipient is ``reject``-full.

        Runs before the WAL append in :meth:`post_envelope` so a
        rejected envelope leaves no trace on disk. Only the ``reject``
        mode raises synchronously; the three non-raising modes are
        handled in :meth:`_deliver_to`:

        * ``spool`` — write to ``overflow/`` without bumping the counter.
        * ``drop_oldest`` — evict the oldest pending file, then deliver.
        * ``drop_newest`` — silently drop the incoming envelope.

        System envelopes (``ag2.session.*``) bypass the check: they
        are ephemeral handshake signals that never consume an inbox
        slot. See :meth:`_deliver_to` for the symmetric write-path
        skip.
        """

        if envelope.event_type.startswith("ag2.session."):
            return

        for recipient_id in self._recipient_ids_for(metadata, envelope):
            rule = self._rules.get(recipient_id)
            if rule is None:
                continue
            inbox = rule.limits.inbox
            if inbox.max_pending == 0:
                continue
            current = self._inbox_pending.get(recipient_id, 0)
            if current < inbox.max_pending:
                continue
            # Non-raising overflow modes fall through to _deliver_to.
            if inbox.overflow in ("spool", "drop_oldest", "drop_newest"):
                continue
            recipient_name = (
                self._identities[recipient_id].name
                if recipient_id in self._identities
                else recipient_id
            )
            raise InboxFullError(
                f"{recipient_name} inbox is full "
                f"({current}/{inbox.max_pending}, overflow={inbox.overflow})"
            )

    async def _fanout_to_participants(
        self, metadata: SessionMetadata, envelope: Envelope
    ) -> None:
        """Deliver ``envelope`` to every session participant except the sender.

        Used for broadcast / discussion envelopes where the hub has to
        clone the envelope to N recipients. Each clone carries its own
        ``recipient_id`` so the receiver's inbox dispatcher can route
        it normally. The WAL copy stays single (no ``recipient_id`` so
        it is addressed to the session as a whole).
        """

        for p in metadata.participants:
            if p.actor_id == envelope.sender_id:
                continue
            clone = Envelope.from_dict(envelope.to_dict())
            clone.recipient_id = p.actor_id
            await self._deliver_to(p.actor_id, clone)

    def _can_observe_session(self, metadata: SessionMetadata, observer_id: str) -> bool:
        """Decide whether ``observer_id`` may open a subscription.

        Participants always qualify. Non-participants need two things:

        1. Their own ``rule.access.subscribe.sessions`` policy must not
           restrict them to member-only observation.
        2. **Every** participant's
           ``rule.access.subscribe.sessions`` must allow hub-public
           observation (most-restrictive wins — one member-only vote
           vetoes the subscription).

        This mirrors the §13.3 rule-conflict-resolution stance:
        explicit denials and tighter defaults win over looser ones.
        """

        if metadata.has_participant(observer_id):
            return True
        observer_rule = self._rules.get(observer_id)
        if observer_rule is None:
            return False
        if not observer_rule.access.subscribe.allows_session_observer(
            is_participant=False, is_hub_member=True
        ):
            return False
        for p in metadata.participants:
            p_rule = self._rules.get(p.actor_id)
            if p_rule is None:
                return False
            if not p_rule.access.subscribe.allows_session_observer(
                is_participant=False, is_hub_member=True
            ):
                return False
        return True

    async def _fanout_to_subs(
        self,
        subs: list[_Subscription],
        metadata: SessionMetadata,
        envelope: Envelope,
        *,
        wal_offset: int = 0,
    ) -> None:
        """Fan-out an envelope to a pre-captured subscription snapshot.

        The snapshot must come from a ``_wal_lock``-protected section in
        :meth:`post_envelope` so subscriptions registered *after* this
        envelope was appended are not double-delivered — they will see
        this envelope through :meth:`_handle_subscribe`'s replay path.

        ``wal_offset`` is the byte position immediately after this
        envelope's append, stamped on every outgoing ``EventFrame``
        so clients can checkpoint their cursor and resume from it on
        reconnect (see :meth:`ActorClient.reconnect`). ``0`` is a
        safe default for subscriptions that don't pin a session.
        """

        for sub in subs:
            if sub.session_id is not None and sub.session_id != metadata.session_id:
                continue
            if (
                sub.causation_id is not None
                and envelope.causation_id != sub.causation_id
            ):
                continue
            try:
                await sub.endpoint.send_frame(
                    EventFrame(
                        subscription_id=sub.subscription_id,
                        envelope=envelope,
                        wal_offset=wal_offset,
                    )
                )
            except Exception:  # pragma: no cover
                log.warning("failed to push event frame", exc_info=True)

    # ------------------------------------------------------------------
    # Link connection handler
    # ------------------------------------------------------------------

    async def connection_handler(self, endpoint: _EndpointSide) -> None:
        """Drive a single Link client through its frame loop."""

        actor_id: str | None = None
        try:
            async for frame in endpoint.frames():
                actor_id = await self._dispatch_frame(endpoint, frame, actor_id)
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception as exc:  # pragma: no cover
            log.warning("connection handler error: %r", exc, exc_info=True)
        finally:
            # Only clear the endpoint mapping if THIS connection is the
            # current one — a reconnect that landed on a new endpoint
            # with the same actor_id already repointed ``_endpoints``
            # to the new endpoint and that new entry must not be wiped.
            was_current = False
            if actor_id is not None:
                registered = self._endpoints.get(actor_id)
                if registered is endpoint:
                    self._endpoints.pop(actor_id, None)
                    was_current = True
            for sid, sub in list(self._subscriptions.items()):
                if sub.endpoint is endpoint:
                    self._subscriptions.pop(sid, None)
            # Phase 3a §3.4: mark the runtime unreachable when the
            # current endpoint goes away. If a reconnect already
            # replaced the endpoint, the hello for the new one has
            # already stamped runtime with ``reachable=true``, so
            # we must not overwrite it here.
            if actor_id is not None and was_current:
                with contextlib.suppress(Exception):  # pragma: no cover
                    await self._write_runtime(actor_id, None, reachable=False)

    async def _dispatch_frame(
        self,
        endpoint: _EndpointSide,
        frame: Frame,
        actor_id: str | None,
    ) -> str | None:
        if isinstance(frame, HelloFrame):
            return await self._handle_hello(endpoint, frame)
        if actor_id is None:
            await endpoint.send_frame(
                ErrorFrame(code="not_authenticated", message="send hello first")
            )
            return None
        if isinstance(frame, SendFrame):
            await self._handle_send(endpoint, frame, actor_id)
            return actor_id
        if isinstance(frame, ReceiptFrame):
            await self._handle_receipt(endpoint, frame, actor_id)
            return actor_id
        if isinstance(frame, ChunkFrame):
            await self._handle_chunk(endpoint, frame, actor_id)
            return actor_id
        if isinstance(frame, SubscribeFrame):
            await self._handle_subscribe(endpoint, frame, actor_id)
            return actor_id
        if isinstance(frame, UnsubscribeFrame):
            self._subscriptions.pop(frame.subscription_id, None)
            return actor_id
        return actor_id

    async def _handle_hello(
        self, endpoint: _EndpointSide, frame: HelloFrame
    ) -> str | None:
        actor_id = frame.resume_actor_id
        if actor_id is None:
            await endpoint.send_frame(
                ErrorFrame(
                    code="missing_actor_id", message="hello must carry resume_actor_id"
                )
            )
            return None
        if actor_id not in self._identities:
            await endpoint.send_frame(
                ErrorFrame(
                    code="unknown_actor", message=f"actor {actor_id} not registered"
                )
            )
            return None
        try:
            await self._auth.validate(self._identities[actor_id], frame.auth_claim)
        except Exception as exc:
            await endpoint.send_frame(ErrorFrame(code="auth_failed", message=str(exc)))
            return None

        # Only one live endpoint per actor in Phase 1.
        previous = self._endpoints.get(actor_id)
        if previous is not None and previous is not endpoint and not previous.closed:
            await previous.close()
        self._endpoints[actor_id] = endpoint
        endpoint.actor_id = actor_id

        # Phase 3a §3.4: stamp the actor's runtime.json with the
        # binding shape of the live transport so ``describe`` /
        # discovery responses report the real address. Phase 1 wrote
        # a placeholder ``{"binding": "local", "reachable": false}``
        # at registration time and never updated it; the runtime
        # record was effectively stale the moment the actor connected.
        await self._write_runtime(actor_id, endpoint, reachable=True)

        await endpoint.send_frame(
            WelcomeFrame(actor_id=actor_id, hub_id=self.config.hub_id)
        )
        return actor_id

    async def _write_runtime(
        self,
        actor_id: str,
        endpoint: _EndpointSide | None,
        *,
        reachable: bool,
    ) -> None:
        """Rewrite an actor's runtime.json with the current transport shape.

        Called from :meth:`_handle_hello` (``reachable=True`` with the
        live endpoint) and from :meth:`connection_handler`'s cleanup
        path (``reachable=False`` with ``endpoint=None`` — the fields
        for the last-known binding are preserved so discovery can
        still report where the actor was).
        """

        existing_raw = await self._store.read(layout.actor_runtime(actor_id))
        existing: dict[str, Any] = {}
        if existing_raw:
            try:
                existing = json.loads(existing_raw)
            except json.JSONDecodeError:  # pragma: no cover
                existing = {}

        if endpoint is not None:
            runtime = {
                "actor_id": actor_id,
                "binding": getattr(endpoint, "binding", "local"),
                "target": endpoint.endpoint_id,
                "ws_url": getattr(endpoint, "ws_url", None),
                "http_url": getattr(endpoint, "http_url", None),
                "reachable": reachable,
                "last_heartbeat": self._clock(),
            }
        else:
            # Disconnect path: preserve the last-known address but
            # flip reachable to false and bump last_heartbeat.
            runtime = {
                "actor_id": actor_id,
                "binding": existing.get("binding", "local"),
                "target": existing.get("target"),
                "ws_url": existing.get("ws_url"),
                "http_url": existing.get("http_url"),
                "reachable": reachable,
                "last_heartbeat": self._clock(),
            }

        await self._store.write(
            layout.actor_runtime(actor_id),
            json.dumps(runtime, sort_keys=True),
        )

    def _audit(
        self,
        *,
        actor_id: str | None,
        action: str,
        resource_type: str,
        resource_id: str,
        decision: str,
        reason: str = "",
        trace_id: str | None = None,
    ) -> None:
        """Queue an audit entry for background write to ``hub/admin/audit.jsonl``.

        Implemented in Phase 3b task #4 as part of the audit log
        writer. Phase 3b task #3 (``drop_oldest`` / ``drop_newest``)
        calls into it before the writer loop is wired up — the
        ``None`` queue short-circuit keeps those call sites
        harmless until the writer is started. Once :meth:`open` /
        :meth:`hydrate` allocates the queue, every call enqueues an
        entry that the background task drains into the audit log.

        Audit writes are strictly non-fatal: if the writer is not
        started (tests that construct a :class:`Hub` directly) or
        the store is unavailable, audit entries are dropped silently
        so mutations are never blocked on durability of the audit
        trail.
        """

        if self._audit_queue is None:
            return
        entry = {
            "ts": self._clock(),
            "actor_id": actor_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "decision": decision,
            "reason": reason,
            "trace_id": trace_id,
        }
        try:
            self._audit_queue.put_nowait(entry)
        except asyncio.QueueFull:  # pragma: no cover
            log.warning("audit queue full — dropping %s entry", action)

    async def _handle_send(
        self,
        endpoint: _EndpointSide,
        frame: SendFrame,
        actor_id: str,
    ) -> None:
        envelope = frame.envelope
        envelope.sender_id = actor_id

        # Idempotency dedup. The ``idempotency_key`` can come from
        # either the frame (wire-level) or the envelope itself
        # (application-level). Repeat sends within the dedup TTL
        # return the cached accept without re-running any of
        # post_envelope's pipeline.
        idempotency_key = frame.idempotency_key or envelope.idempotency_key
        if idempotency_key is not None:
            cached = self._lookup_idempotent(envelope.session_id, idempotency_key)
            if cached is not None:
                await endpoint.send_frame(
                    AcceptFrame(
                        envelope_id=cached.envelope_id, wal_offset=cached.wal_offset
                    )
                )
                return

        try:
            envelope_id, wal_offset = await self.post_envelope(envelope)
        except AccessDeniedError as exc:
            await endpoint.send_frame(
                ErrorFrame(code="access_denied", message=str(exc))
            )
            return
        except LimitExceededError as exc:
            await endpoint.send_frame(
                ErrorFrame(code="limit_exceeded", message=str(exc))
            )
            return
        except RuleViolationError as exc:
            await endpoint.send_frame(
                ErrorFrame(code="rule_violation", message=str(exc))
            )
            return
        except SessionClosedError as exc:
            await endpoint.send_frame(
                ErrorFrame(code="session_closed", message=str(exc))
            )
            return
        except UnknownTaskError as exc:
            # Phase 4 — task envelope referenced a task_id the hub has
            # never heard of (or one that has been archived). Surface
            # it to the sender with its own error code so clients can
            # distinguish stale task_ids from generic session errors.
            await endpoint.send_frame(ErrorFrame(code="unknown_task", message=str(exc)))
            return
        except TaskStateError as exc:
            # Phase 4 — task state-machine violation (wrong sender,
            # terminal-state reuse, bad phase id, etc.). Distinct code
            # so the client can raise a precise ``TaskStateError``
            # instead of a generic ``SessionError``.
            await endpoint.send_frame(ErrorFrame(code="task_state", message=str(exc)))
            return
        except SessionError as exc:
            await endpoint.send_frame(
                ErrorFrame(code="session_error", message=str(exc))
            )
            return
        except UnknownActorError as exc:
            await endpoint.send_frame(
                ErrorFrame(code="unknown_actor", message=str(exc))
            )
            return
        except InboxFullError as exc:
            await endpoint.send_frame(ErrorFrame(code="inbox_full", message=str(exc)))
            return
        if idempotency_key is not None:
            self._record_idempotent(
                envelope.session_id, idempotency_key, envelope_id, wal_offset
            )
        await endpoint.send_frame(
            AcceptFrame(envelope_id=envelope_id, wal_offset=wal_offset)
        )

    async def _handle_receipt(
        self,
        endpoint: _EndpointSide,
        frame: ReceiptFrame,
        actor_id: str,
    ) -> None:
        """Apply a :class:`ReceiptFrame` against the actor's structured inbox.

        ``ack`` → move ``pending/{id}.json`` to ``received/{id}.json``
        and decrement the actor's pending counter. Idempotent: a
        receipt for an envelope that's already been processed (file
        already moved, or was never persisted — e.g. a spooled
        envelope) is a no-op.

        ``nack`` → append a structured entry to
        ``hub/actors/{id}/inbox/nacks.jsonl`` recording the reason, and
        remove the pending file. The sender has already gotten its
        AcceptFrame from the prior post, so nack is purely a signal to
        the hub about why delivery failed — it does not propagate back
        to the original sender in Phase 3a. (Phase 3b: nack surfaces
        via the audit log so operators can trace delivery failures.)
        """

        envelope_id = frame.envelope_id
        pending_path = layout.actor_inbox_pending(actor_id, envelope_id)
        received_path = layout.actor_inbox_received(actor_id, envelope_id)

        payload = await self._store.read(pending_path)

        if frame.status == "ack":
            if payload is None:
                # Already processed, or the envelope was spooled
                # (overflow/), or the actor is acking something it
                # invented. All three are no-ops.
                return
            await self._store.write(received_path, payload)
            await self._store.delete(pending_path)
            self._inbox_pending[actor_id] = max(
                0, self._inbox_pending.get(actor_id, 0) - 1
            )
            return

        if frame.status == "nack":
            nack_entry = json.dumps(
                {
                    "envelope_id": envelope_id,
                    "actor_id": actor_id,
                    "reason": frame.reason,
                    "at": self._clock(),
                },
                sort_keys=True,
            )
            await self._store.append(
                layout.actor_inbox_nacks(actor_id), nack_entry + "\n"
            )
            if payload is not None:
                await self._store.delete(pending_path)
                self._inbox_pending[actor_id] = max(
                    0, self._inbox_pending.get(actor_id, 0) - 1
                )
            return

        # Unknown status — log and ignore; the protocol only defines
        # ack/nack in Phase 3a.
        log.warning(
            "hub: ignoring receipt with unknown status %r for %s",
            frame.status,
            envelope_id,
        )

    def _lookup_idempotent(
        self, session_id: str, idempotency_key: str
    ) -> _IdempotencyEntry | None:
        """Look up a cached idempotent send. GC expired entries lazily."""

        key = (session_id, idempotency_key)
        entry = self._idempotency.get(key)
        if entry is None:
            return None
        now = time.monotonic()
        if entry.expires_at <= now:
            self._idempotency.pop(key, None)
            return None
        return entry

    def _record_idempotent(
        self,
        session_id: str,
        idempotency_key: str,
        envelope_id: str,
        wal_offset: int,
    ) -> None:
        expires = time.monotonic() + self.config.idempotency_ttl_s
        self._idempotency[(session_id, idempotency_key)] = _IdempotencyEntry(
            envelope_id=envelope_id, wal_offset=wal_offset, expires_at=expires
        )
        # Opportunistic GC: sweep at most 32 entries per insert so the
        # dict doesn't grow without bound under a restart-and-retry
        # storm from clients that never reuse a key.
        if len(self._idempotency) > 1024:
            now = time.monotonic()
            stale = [k for k, v in self._idempotency.items() if v.expires_at <= now]
            for k in stale[:32]:
                self._idempotency.pop(k, None)

    async def _handle_chunk(
        self,
        endpoint: _EndpointSide,
        frame: ChunkFrame,
        actor_id: str,
    ) -> None:
        """Route an inbound :class:`ChunkFrame` to its recipient.

        Chunks are transient — the hub relays them frame-to-frame
        without persisting them in the session WAL. The
        :class:`Envelope` they precede is still written via the normal
        SendFrame → post_envelope path, so the authoritative record
        stays clean. The sender's session must exist and the sender
        must be a participant; the hub stamps ``sender_id`` onto the
        outgoing clone so tenant code can't spoof other identities.
        """

        session_id = frame.session_id
        if session_id is None:
            await endpoint.send_frame(
                ErrorFrame(
                    code="chunk_missing_session",
                    message="chunk frame must set session_id",
                )
            )
            return
        metadata = self._sessions.get(session_id)
        if metadata is None:
            await endpoint.send_frame(
                ErrorFrame(code="unknown_session", message=session_id)
            )
            return
        if metadata.state not in (SessionState.ACTIVE, SessionState.PENDING):
            await endpoint.send_frame(
                ErrorFrame(
                    code="session_closed",
                    message=f"session {session_id} is {metadata.state.value}",
                )
            )
            return
        if not metadata.has_participant(actor_id):
            await endpoint.send_frame(
                ErrorFrame(
                    code="access_denied",
                    message="chunk sender is not a participant",
                )
            )
            return

        stamped = ChunkFrame(
            envelope_id=frame.envelope_id,
            chunk_index=frame.chunk_index,
            content=frame.content,
            session_id=session_id,
            sender_id=actor_id,
            recipient_id=frame.recipient_id,
            final=frame.final,
        )
        await self._deliver_chunk(metadata, stamped)

    async def _deliver_chunk(
        self, metadata: SessionMetadata, frame: ChunkFrame
    ) -> None:
        """Deliver a chunk to its recipient(s).

        When ``recipient_id`` is set the chunk is unicast. When it is
        ``None`` the chunk is a broadcast — fan out to every non-sender
        participant, mirroring :meth:`_fanout_to_participants`.
        """

        sender = frame.sender_id or ""
        if frame.recipient_id is not None:
            recipient_endpoint = self._endpoints.get(frame.recipient_id)
            if recipient_endpoint is not None and not recipient_endpoint.closed:
                try:
                    await recipient_endpoint.send_frame(frame)
                except Exception:  # pragma: no cover
                    log.warning("failed to push chunk frame", exc_info=True)
            return

        for p in metadata.participants:
            if p.actor_id == sender:
                continue
            endpoint = self._endpoints.get(p.actor_id)
            if endpoint is None or endpoint.closed:
                continue
            clone = ChunkFrame(
                envelope_id=frame.envelope_id,
                chunk_index=frame.chunk_index,
                content=frame.content,
                session_id=frame.session_id,
                sender_id=sender,
                recipient_id=p.actor_id,
                final=frame.final,
            )
            try:
                await endpoint.send_frame(clone)
            except Exception:  # pragma: no cover
                log.warning("failed to push chunk frame", exc_info=True)

    async def _handle_subscribe(
        self,
        endpoint: _EndpointSide,
        frame: SubscribeFrame,
        actor_id: str,
    ) -> None:
        if frame.session_id is not None and frame.session_id not in self._sessions:
            await endpoint.send_frame(
                ErrorFrame(
                    code="unknown_session",
                    message=frame.session_id,
                    request_id=frame.subscription_id,
                )
            )
            return
        if frame.session_id is not None:
            metadata = self._sessions[frame.session_id]
            if not self._can_observe_session(metadata, actor_id):
                self._audit(
                    actor_id=actor_id,
                    action="subscribe_session",
                    resource_type="session",
                    resource_id=frame.session_id,
                    decision="deny",
                    reason="access_denied",
                )
                await endpoint.send_frame(
                    ErrorFrame(
                        code="access_denied",
                        message="not permitted to subscribe to this session",
                        request_id=frame.subscription_id,
                    )
                )
                return
            self._audit(
                actor_id=actor_id,
                action="subscribe_session",
                resource_type="session",
                resource_id=frame.session_id,
                decision="allow",
                reason="",
            )

        sub = _Subscription(
            subscription_id=frame.subscription_id,
            actor_id=actor_id,
            session_id=frame.session_id,
            causation_id=frame.causation_id,
            endpoint=endpoint,
            since=frame.since or 0,
        )

        # Invariant: every envelope appended to the WAL is delivered to
        # every matching subscription exactly once, either via replay
        # (the sub was registered AFTER this envelope was appended) or
        # via fan-out (the sub was registered BEFORE this envelope was
        # appended). The invariant holds because:
        #
        #   - _wal_lock is held during (WAL append + subs snapshot) in
        #     post_envelope/_post_adapter_follow_up/_handle_system_envelope
        #   - _wal_lock is held during (WAL snapshot read + sub insert)
        #     here
        #
        # so the two critical sections serialize with each other. A
        # concurrent post_envelope either lands before this block (its
        # envelope will be in ``prior``) or after it (its envelope will
        # see ``sub`` in the snapshot and fan-out will deliver it).
        if frame.session_id is not None:
            async with self._wal_lock:
                prior = await self._read_wal_with_offsets(
                    frame.session_id, since=sub.since
                )
                self._subscriptions[sub.subscription_id] = sub
        else:
            async with self._wal_lock:
                self._subscriptions[sub.subscription_id] = sub
            prior = []

        replay_end_offset = sub.since
        for envelope, offset in prior:
            if (
                sub.causation_id is not None
                and envelope.causation_id != sub.causation_id
            ):
                continue
            await endpoint.send_frame(
                EventFrame(
                    subscription_id=sub.subscription_id,
                    envelope=envelope,
                    wal_offset=offset,
                )
            )
            replay_end_offset = offset

        # Phase 3a: acknowledge the subscription now that it is live on
        # the hub side and the initial replay has been pushed. The
        # client uses this to know "subscribe was applied + replay
        # drained" — critical for reconnect ordering so subsequent
        # sends reach the new subscription. We overload ``AcceptFrame``
        # with ``request_id=<subscription_id>`` and ``envelope_id=""``;
        # the client's frame dispatcher routes it to a per-sub future
        # instead of the outbound-send accept path. ``wal_offset`` is
        # the byte position after the last replayed envelope so the
        # client's cursor is up-to-date even before the first live
        # fan-out delivery.
        await endpoint.send_frame(
            AcceptFrame(
                envelope_id="",
                wal_offset=replay_end_offset,
                request_id=sub.subscription_id,
            )
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def hub_id(self) -> str:
        return self.config.hub_id
