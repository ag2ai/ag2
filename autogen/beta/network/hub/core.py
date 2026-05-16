# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``Hub`` — registry, dispatcher, persistence root.

The hub owns the registry (passports / resumes / rules / skills + name
and capability indexes), the channel and task state machines, the WAL,
the dispatch path, the adapter state cache, the audit log, and the
internal sweepers (TTL + expectations).

Channel machinery:
* Adapter registry by ``(manifest.type, manifest.version)``.
* Per-channel ``AdapterState`` cache, folded under the per-channel
  WAL lock so ``validate_send`` and ``on_accepted`` are O(1).
* Invite handshake: ``create_channel`` posts ``EV_CHANNEL_INVITE``,
  awaits ``EV_CHANNEL_INVITE_ACK`` from every invitee (timeout
  ``invite_ack_timeout``), broadcasts ``EV_CHANNEL_OPENED`` on quorum.
* TTL: parsed from ``Rule.limits.channel_ttl_default`` /
  ``task_ttl_default`` (or per-channel override). The ``_TtlSweeper``
  walks active channels and tasks every ``ttl_sweep_interval``;
  cascades non-terminal tasks under closing channels to ``EXPIRED``.

The hub never calls ``Agent.ask``, executes tenant transforms, or
imports tenant modules — the trust boundary runs through ``HubClient``
/ ``AgentClient``.
"""

import asyncio
import contextlib
import dataclasses
import fnmatch
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone

from autogen.beta.knowledge import KnowledgeStore
from autogen.beta.task import TERMINAL_TASK_STATES, TaskMetadata, TaskSpec, TaskState

from ..adapters.base import ChannelAdapter
from ..adapters.consulting import ConsultingAdapter
from ..adapters.conversation import ConversationAdapter
from ..adapters.discussion import DiscussionAdapter
from ..adapters.workflow import WorkflowAdapter
from ..auth import AuthRegistry
from ..channel import (
    ChannelManifest,
    ChannelMetadata,
    ChannelState,
    Expectation,
    Participant,
    ParticipantRole,
    is_terminal_channel_state,
)
from ..envelope import (
    EV_CHANNEL_CLOSED,
    EV_CHANNEL_EXPIRED,
    EV_CHANNEL_INVITE,
    EV_CHANNEL_INVITE_ACK,
    EV_CHANNEL_INVITE_REJECT,
    EV_CHANNEL_OPENED,
    EV_TEXT,
    Envelope,
)
from ..errors import (
    AccessDeniedError,
    NetworkError,
    NotFoundError,
    ProtocolError,
)
from ..identity import ObservedStat, Passport, Resume
from ..ids import make_id
from ..rule import Rule, parse_duration
from ..transport.frames import (
    AcceptFrame,
    ErrorFrame,
    Frame,
    HelloFrame,
    NotifyFrame,
    PingFrame,
    PongFrame,
    SendFrame,
    WelcomeFrame,
)
from ..transport.link import LinkEndpoint
from ..views.base import ViewPolicy
from .arbiter import Deny, HubArbiter, RuleBasedArbiter
from .audit import AuditLog
from .expectations import (
    ExpectationContext,
    ExpectationEvaluator,
    ViolationHandler,
    default_evaluators,
    default_handlers,
)
from .layout import (
    agents_root,
    by_capability_path,
    channel_metadata_path,
    channels_root,
    passport_path,
    resume_path,
    rule_path,
    skill_path,
    task_metadata_path,
    tasks_root,
    wal_path,
)
from .listener import HubListener
from .sweepers import _IntervalSweeper

__all__ = ("Hub",)


logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_ERROR_CODE_MAP: dict[type, str] = {
    NotFoundError: "not_found",
    AccessDeniedError: "access_denied",
    ProtocolError: "protocol_error",
}


def _error_code(exc: BaseException) -> str:
    for cls, code in _ERROR_CODE_MAP.items():
        if isinstance(exc, cls):
            return code
    return "error"


def _match_any(name: str, patterns: list[str]) -> bool:
    """True if ``name`` matches any of the glob patterns (``["*"]`` allows all)."""
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def _is_channel_protocol_event(event_type: str) -> bool:
    return event_type.startswith("ag2.channel.")


def _is_task_event(event_type: str) -> bool:
    return event_type.startswith("ag2.task.")


def _is_protocol_event(event_type: str) -> bool:
    return _is_channel_protocol_event(event_type) or _is_task_event(event_type)


def _expires_at(now_iso: str, ttl_seconds: int) -> str:
    """Compute ``expires_at`` ISO timestamp from a base + duration."""
    if ttl_seconds <= 0:
        return ""
    base = datetime.fromisoformat(now_iso)
    return (base + timedelta(seconds=ttl_seconds)).isoformat()


class Hub:
    """In-process registry, dispatcher, channel state-machine, persistence root.

    Construct with :meth:`open` for production (hydrates from disk and
    spawns sweepers); the sync ``__init__`` is for tests that need
    fine-grained control.
    """

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        auth: AuthRegistry | None = None,
        clock: Callable[[], str] | None = None,
        ttl_sweep_interval: float = 30.0,
        expectation_sweep_interval: float = 10.0,
        invite_ack_timeout: float = 30.0,
    ) -> None:
        # __init__ stores params; side effects deferred to start()/hydrate().
        self._store = store
        self._auth = auth if auth is not None else AuthRegistry.default()
        self._clock = clock if clock is not None else _utc_now_iso
        self._ttl_sweep_interval = ttl_sweep_interval
        self._expectation_sweep_interval = expectation_sweep_interval
        self._invite_ack_timeout = invite_ack_timeout

        # Audit log + expectation registries. AuditLog is a HubListener;
        # see _install_default_listeners() — it installs as the first
        # listener so audit records are written before any tenant
        # listener observes the same event.
        self._audit_log = AuditLog(store, clock=self._clock)
        self._expectation_evaluators: dict[str, ExpectationEvaluator] = {}
        self._violation_handlers: dict[str, ViolationHandler] = {}
        # channel_id → set of (expectation_index, expectation_name, violator_id) fired.
        # The position-based index disambiguates same-name expectations
        # (e.g. two ``turn_within`` entries with different ``on_violation``
        # handlers — without the index the first to fire would suppress
        # the second). Empty violator_id ("") = channel-wide violations.
        self._fired_violations: dict[str, set[tuple[int, str, str]]] = {}

        # Identity caches.
        self._passports: dict[str, Passport] = {}
        self._resumes: dict[str, Resume] = {}
        self._rules: dict[str, Rule] = {}
        self._skills: dict[str, str] = {}
        self._name_to_id: dict[str, str] = {}
        # capability name → set of agent_ids that claim or have observed it.
        # Persisted as registry/by_capability.json on every mutation
        # (rebuilt from resumes on hydrate — the file is a derived cache).
        self._capability_index: dict[str, set[str]] = {}

        # Adapter registry.
        self._adapters: dict[tuple[str, int], ChannelAdapter] = {}

        # Channel caches.
        self._channels: dict[str, ChannelMetadata] = {}
        self._active_channels: dict[str, ChannelMetadata] = {}
        self._adapter_states: dict[str, object] = {}
        self._channel_open_waiters: dict[str, asyncio.Future[ChannelMetadata]] = {}

        # Task caches (observed; not owned).
        self._tasks: dict[str, TaskMetadata] = {}
        self._channel_tasks: dict[str, set[str]] = {}
        # task_ids whose terminal observation has been recorded into
        # the owner's ``Resume.observed`` already. Prevents double-counting
        # when the same task receives multiple terminal events (e.g. a
        # channel-cascade EXPIRED followed by an owner-emitted COMPLETED).
        self._observed_task_ids: set[str] = set()

        # Per-recipient outstanding-envelope counter for ``InboxBlock.max_pending``
        # enforcement. Incremented on dispatch to that recipient,
        # decremented when the recipient posts any envelope (treating
        # any outbound activity as "I'm processing my inbox"). A
        # best-effort approximation; per-channel ack semantics require
        # a transport with ack frames.
        self._inbox_pending: dict[str, int] = {}

        # Transport-side state.
        self._endpoints_by_id: dict[str, LinkEndpoint] = {}
        self._agent_to_endpoint: dict[str, str] = {}
        self._endpoint_to_agents: dict[str, set[str]] = {}
        self._endpoint_tasks: set[asyncio.Task[None]] = set()

        # Per-channel locks for WAL append + dispatch ordering.
        self._channel_locks: dict[str, asyncio.Lock] = {}
        self._registration_lock = asyncio.Lock()

        self._ttl_sweeper: _IntervalSweeper | None = None
        self._expectation_sweeper: _IntervalSweeper | None = None
        # Subclass-registered periodic workers. Keyed by name so a
        # subclass can replace a sweeper at the same name (e.g. via a
        # config reload) by unregister + re-register.
        self._custom_sweepers: dict[str, _IntervalSweeper] = {}
        # Set True by ``start()`` so ``register_sweeper`` knows whether
        # to spawn the new sweeper immediately or queue it for ``start()``.
        self._started = False
        self._closed = False

        # Observability + decision-making seams. Audit log is registered
        # as the first listener so it sees every event a tenant
        # listener does, in the same order.
        self._listeners: list[HubListener] = [self._audit_log]
        self._arbiter: HubArbiter = RuleBasedArbiter()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @classmethod
    async def open(
        cls,
        store: KnowledgeStore,
        *,
        auth: AuthRegistry | None = None,
        clock: Callable[[], str] | None = None,
        ttl_sweep_interval: float = 30.0,
        expectation_sweep_interval: float = 10.0,
        invite_ack_timeout: float = 30.0,
        register_default_adapters: bool = True,
    ) -> "Hub":
        """Construct + hydrate from disk + start sweepers. Production entry point.

        ``register_default_adapters=True`` (default) registers the
        built-in adapters (``consulting@v1``, ``conversation@v1``,
        ``discussion@v1``) and the built-in expectation evaluators /
        violation handlers (``acks_within`` / ``reply_within`` /
        ``max_silence``, ``audit`` / ``notify_channel`` /
        ``auto_close``) so simple test setups don't need explicit
        registration calls.

        Set ``expectation_sweep_interval=0`` to disable the expectation
        sweeper entirely (tests usually do this to avoid background
        timer noise).
        """
        hub = cls(
            store,
            auth=auth,
            clock=clock,
            ttl_sweep_interval=ttl_sweep_interval,
            expectation_sweep_interval=expectation_sweep_interval,
            invite_ack_timeout=invite_ack_timeout,
        )
        if register_default_adapters:
            hub.register_adapter(ConsultingAdapter())
            hub.register_adapter(ConversationAdapter())
            hub.register_adapter(DiscussionAdapter())
            hub.register_adapter(WorkflowAdapter())
            for evaluator in default_evaluators():
                hub.register_expectation_evaluator(evaluator)
            for handler in default_handlers():
                hub.register_violation_handler(handler)
        await hub.hydrate()
        await hub.start()
        return hub

    async def hydrate(self) -> None:
        """Walk the store; rebuild caches. Idempotent.

        Loads identities, channels, and tasks from disk. Active channel
        WALs are re-folded through their adapter so the
        ``_adapter_states`` cache is rebuilt deterministically.
        """
        self._passports.clear()
        self._resumes.clear()
        self._rules.clear()
        self._skills.clear()
        self._name_to_id.clear()
        self._capability_index.clear()
        self._channels.clear()
        self._active_channels.clear()
        self._adapter_states.clear()
        self._tasks.clear()
        self._channel_tasks.clear()

        # Identities.
        agent_children = await self._store.list(agents_root())
        for child in agent_children:
            if not child.endswith("/"):
                continue
            agent_id = child.rstrip("/")
            await self._load_agent(agent_id)

        # Rebuild capability index from loaded resumes — by_capability.json
        # is a derived cache, the resumes are the authoritative source.
        for agent_id, resume in self._resumes.items():
            for cap in resume.claimed_capabilities:
                self._capability_index.setdefault(cap, set()).add(agent_id)
            for cap in resume.observed:
                self._capability_index.setdefault(cap, set()).add(agent_id)

        # Channels — load metadata first, then re-fold WALs.
        channel_children = await self._store.list(channels_root())
        for child in channel_children:
            if not child.endswith("/"):
                continue
            channel_id = child.rstrip("/")
            await self._load_channel(channel_id)

        # Tasks.
        task_children = await self._store.list(tasks_root())
        for child in task_children:
            if not child.endswith("/"):
                continue
            task_id = child.rstrip("/")
            await self._load_task(task_id)

    async def start(self) -> None:
        """Spawn the TTL + expectation + custom sweepers. Idempotent.

        ``ttl_sweep_interval=0`` disables the TTL sweeper;
        ``expectation_sweep_interval=0`` disables the expectation
        sweeper. Custom sweepers attached via
        :meth:`register_sweeper` start here too (registered before
        ``start()``) or immediately at registration (after ``start()``).
        """
        if self._ttl_sweep_interval > 0 and self._ttl_sweeper is None:
            self._ttl_sweeper = _IntervalSweeper(
                name="ttl",
                interval=self._ttl_sweep_interval,
                fn=self.expire_due,
            )
            self._ttl_sweeper.start()
        if self._expectation_sweep_interval > 0 and self._expectation_sweeper is None:
            self._expectation_sweeper = _IntervalSweeper(
                name="expectations",
                interval=self._expectation_sweep_interval,
                fn=self._expectation_tick,
            )
            self._expectation_sweeper.start()
        for sweeper in self._custom_sweepers.values():
            sweeper.start()
        self._started = True

    async def close(self) -> None:
        """Cancel sweepers + endpoint tasks; drain queues. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._ttl_sweeper is not None:
            await self._ttl_sweeper.stop()
            self._ttl_sweeper = None
        if self._expectation_sweeper is not None:
            await self._expectation_sweeper.stop()
            self._expectation_sweeper = None
        for sweeper in list(self._custom_sweepers.values()):
            await sweeper.stop()
        self._custom_sweepers.clear()
        for task in list(self._endpoint_tasks):
            task.cancel()
        if self._endpoint_tasks:
            await asyncio.gather(*self._endpoint_tasks, return_exceptions=True)
        self._endpoint_tasks.clear()

    # ── Sweeper extension ───────────────────────────────────────────────────

    def register_sweeper(
        self,
        name: str,
        interval_seconds: float,
        fn: Callable[[], "Awaitable[None]"],
    ) -> None:
        """Attach a custom periodic worker.

        ``fn`` is called every ``interval_seconds``. Subclasses use this
        for protocol-specific background work (e.g. polling a chat
        platform's presence list, refreshing an auth token).

        If ``Hub.start()`` has already run, the sweeper starts immediately.
        Otherwise it's queued and starts when ``start()`` runs.

        Re-registering at the same ``name`` raises ``ValueError`` — use
        :meth:`unregister_sweeper` first if you mean to replace.

        Sync vs. async: ``register_sweeper`` is synchronous because it
        only updates internal bookkeeping (and may call the underlying
        ``_IntervalSweeper.start`` which schedules a fire-and-forget
        task). :meth:`unregister_sweeper` is async because it awaits the
        sweeper's task cancellation to ensure clean shutdown.
        """
        if name in self._custom_sweepers:
            raise ValueError(f"sweeper already registered: {name!r}")
        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be positive: {interval_seconds}")
        sweeper = _IntervalSweeper(name=name, interval=interval_seconds, fn=fn)
        self._custom_sweepers[name] = sweeper
        if self._started:
            # ``start()`` has already run — start this sweeper immediately.
            sweeper.start()

    async def unregister_sweeper(self, name: str) -> None:
        """Stop and remove a custom sweeper. No-op if absent.

        Async to mirror :meth:`_IntervalSweeper.stop`, which awaits
        cancellation of the running task. Custom sweepers registered
        before :meth:`start` still go through this path on
        :meth:`close`, so subclasses don't need to track them
        themselves.
        """
        sweeper = self._custom_sweepers.pop(name, None)
        if sweeper is not None:
            await sweeper.stop()

    async def __aenter__(self) -> "Hub":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ── Adapter registry ────────────────────────────────────────────────────

    def register_adapter(self, adapter: ChannelAdapter) -> None:
        """Register a ``ChannelAdapter`` keyed by ``(type, version)``.

        Re-registering at the same key replaces the prior adapter; the
        old key's existing in-flight channels keep their snapshotted
        manifest for life.
        """
        key = (adapter.manifest.type, adapter.manifest.version)
        self._adapters[key] = adapter

    def _adapter_for(self, manifest_type: str, manifest_version: int) -> ChannelAdapter:
        adapter = self._adapters.get((manifest_type, manifest_version))
        if adapter is None:
            raise NotFoundError(f"no adapter registered for {manifest_type!r}@v{manifest_version}")
        return adapter

    # ── Observability + decision-making ─────────────────────────────────────

    def register_listener(self, listener: HubListener) -> None:
        """Attach a :class:`HubListener` to receive state-transition events.

        Listeners receive events in registration order. Each is wrapped
        in try/except so one buggy listener cannot stall dispatch — its
        exception is logged at ``ERROR`` and the next listener still
        runs.
        """
        self._listeners.append(listener)

    def unregister_listener(self, listener: HubListener) -> None:
        """Detach a previously-registered listener. No-op if absent."""
        with contextlib.suppress(ValueError):
            self._listeners.remove(listener)

    async def report_turn_failure(
        self,
        *,
        channel_id: str,
        agent_id: str,
        envelope_id: str,
        exc: BaseException,
    ) -> None:
        """Fan out an ``on_turn_failed`` event to every listener.

        Public entry point so client-side notify handlers can route
        substantive-turn crashes through the observability surface
        without touching hub privates. ``AuditLog`` (the built-in
        listener) records the failure; tenant listeners react however
        they choose.
        """
        await self._fan_out(
            "on_turn_failed",
            channel_id,
            agent_id,
            envelope_id,
            exc,
        )

    async def fire_task_event(
        self,
        task_id: str,
        kind: str,
        payload: dict,
    ) -> None:
        """Fan out an ``on_task_event`` to every listener.

        Public entry point so :class:`TaskMirror` (and other tenant
        observers) can route task-side notifications through the hub's
        listener surface without touching ``_fan_out``. ``kind`` is
        free-form — the built-in listener Protocol documents
        ``"started"`` / ``"progress"`` / ``"completed"`` / ``"failed"`` /
        ``"expired"`` / ``"cancelled"`` / ``"mirror_failed"`` as the
        recognised values, but tenants may emit additional kinds.
        """
        await self._fan_out("on_task_event", task_id, kind, payload)

    @property
    def audit_log(self) -> AuditLog:
        """Public access to the built-in audit log (a :class:`HubListener`).

        Use this to call :meth:`AuditLog.read_all` from tooling or to
        attach a live subscriber via :meth:`AuditLog.subscribe`. Custom
        hub subclasses that want a different audit format can replace
        the instance via :meth:`replace_audit_log`.
        """
        return self._audit_log

    def replace_audit_log(self, audit_log: AuditLog) -> None:
        """Swap the built-in :class:`AuditLog` for a tenant-provided one.

        Unregisters the prior audit log from the listener chain,
        registers the replacement as the first listener (preserving
        the convention that audit writes complete before tenant
        listeners observe the same event), and updates
        :attr:`audit_log` to point at it.
        """
        with contextlib.suppress(ValueError):
            self._listeners.remove(self._audit_log)
        self._audit_log = audit_log
        self._listeners.insert(0, audit_log)

    def register_arbiter(self, arbiter: HubArbiter) -> None:
        """Replace the active :class:`HubArbiter` instance.

        The default :class:`RuleBasedArbiter` is installed automatically
        and enforces per-agent :class:`Rule` (access + limits) — the
        same behavior the hub had inline before this seam existed.
        Tenants replace it to layer custom permission protocols
        (JWT scope, federation routing, etc.) on top of (or in place
        of) the rule-based defaults.

        Only one arbiter is active at a time; calling this with a new
        instance replaces the prior arbiter outright.
        """
        self._arbiter = arbiter

    @property
    def arbiter(self) -> HubArbiter:
        """The currently active arbiter (read-only access for testing)."""
        return self._arbiter

    def health(self) -> dict:
        """Return an operational snapshot of hub state.

        Cheap to compute (in-memory only). Wire to a ``/health``
        endpoint or operational dashboard. The shape is intentionally
        small — operators want a handful of indicative numbers, not
        the full state.

        Fields:

        * ``active_channels``: number of channels in OPENED/PENDING state.
        * ``registered_agents``: total registered identities (agents +
          humans).
        * ``pending_inbox_total``: sum of per-recipient outstanding
          envelope counters (best-effort approximation).
        * ``max_pending_inbox_depth``: maximum per-recipient queue depth,
          or ``None`` when nothing is queued. Indicative of the
          "stuck agent" case.
        * ``registered_listeners``: number of attached
          :class:`HubListener` instances (the built-in
          :class:`AuditLog` counts).
        * ``adapters_loaded``: number of registered
          :class:`ChannelAdapter` instances.
        """
        max_depth: int | None = None
        if self._inbox_pending:
            max_depth = max(self._inbox_pending.values())
        return {
            "active_channels": len(self._active_channels),
            "registered_agents": len(self._passports),
            "pending_inbox_total": sum(self._inbox_pending.values()),
            "max_pending_inbox_depth": max_depth,
            "registered_listeners": len(self._listeners),
            "adapters_loaded": len(self._adapters),
            "audit_log_bytes": self._audit_log.bytes_written,
        }

    async def _fan_out(self, method_name: str, *args: object) -> None:
        """Call ``method_name(*args)`` on every registered listener
        AND on the hub itself (so subclasses can override hooks
        directly without registering themselves).

        Per-listener try/except — a single bad listener cannot stall
        the hub. Exceptions log at ``ERROR`` with the listener's
        class name so the offending hook is identifiable.
        """
        # Subclass override path. Bound methods on this instance.
        own_method = getattr(self, method_name, None)
        # Avoid recursing — only treat it as an override if the bound
        # method is NOT the same _fan_out method itself.
        if own_method is not None and not method_name.startswith("_") and own_method != self._fan_out:
            try:
                await own_method(*args)
            except Exception:
                logger.exception(
                    "%s.%s raised",
                    type(self).__name__,
                    method_name,
                )
        for listener in self._listeners:
            method = getattr(listener, method_name, None)
            if method is None:
                continue
            try:
                await method(*args)
            except Exception:
                logger.exception(
                    "listener %s.%s raised",
                    type(listener).__name__,
                    method_name,
                )

    # ── Subclass hooks ──────────────────────────────────────────────────────
    #
    # Default no-op implementations of the listener method set so
    # subclasses can override directly without registering themselves
    # as listeners. The base ``Hub`` provides empty bodies; subclass
    # overrides run alongside any externally-registered listeners.

    async def on_envelope_posted(self, envelope: Envelope, metadata: ChannelMetadata) -> None:  # noqa: ARG002
        return None

    async def on_envelope_rejected(self, envelope: Envelope, reason: NetworkError) -> None:  # noqa: ARG002
        return None

    async def on_dispatch_failed(
        self,
        envelope: Envelope,
        recipient_id: str,
        reason: BaseException,
    ) -> None:  # noqa: ARG002
        return None

    async def on_channel_event(self, channel_id: str, kind: str, payload: dict) -> None:  # noqa: ARG002
        return None

    async def on_agent_event(self, agent_id: str, kind: str, payload: dict) -> None:  # noqa: ARG002
        return None

    async def on_expectation_fired(self, channel_id: str, expectation: object, violation: object) -> None:  # noqa: ARG002
        return None

    async def on_turn_failed(
        self,
        channel_id: str,
        agent_id: str,
        envelope_id: str,
        exc: BaseException,
    ) -> None:  # noqa: ARG002
        return None

    async def on_task_event(self, task_id: str, kind: str, payload: dict) -> None:  # noqa: ARG002
        return None

    async def on_inbox_pressure(self, agent_id: str, pending: int, cap: int) -> None:  # noqa: ARG002
        return None

    # ── Expectation registry ────────────────────────────────────────────────

    def register_expectation_evaluator(self, evaluator: ExpectationEvaluator) -> None:
        """Register an evaluator keyed by ``evaluator.name``.

        Re-registering the same name replaces the prior evaluator.
        """
        self._expectation_evaluators[evaluator.name] = evaluator

    def register_violation_handler(self, handler: ViolationHandler) -> None:
        """Register a violation handler keyed by ``handler.name``.

        Re-registering the same name replaces the prior handler.
        """
        self._violation_handlers[handler.name] = handler

    async def _expectation_tick(self) -> None:
        """One sweeper tick: evaluate every expectation on every active
        channel; fire registered handlers on new violations.

        Per-(channel, expectation, violator) dedup lives in
        ``_fired_violations`` so handlers don't re-fire on every tick.
        Cleared on terminal channel transitions.
        """
        if not self._expectation_evaluators or not self._violation_handlers:
            return
        now_iso = self._clock()
        now_seconds = datetime.fromisoformat(now_iso).timestamp()
        for channel_id, metadata in list(self._active_channels.items()):
            adapter_state = self._adapter_states.get(channel_id)
            wal = await self.read_wal(channel_id)
            context = ExpectationContext(
                metadata=metadata,
                state=adapter_state,
                wal=wal,
                now_iso=now_iso,
                now_seconds=now_seconds,
            )
            terminal = False
            for idx, expectation in enumerate(metadata.manifest.expectations):
                evaluator = self._expectation_evaluators.get(expectation.name)
                if evaluator is None:
                    continue
                violation = evaluator.evaluate(expectation, context)
                if violation is None:
                    continue
                handler = self._violation_handlers.get(expectation.on_violation)
                if handler is None:
                    continue
                fired = self._fired_violations.setdefault(channel_id, set())
                violator_keys = violation.violator_ids or [""]
                for vid in violator_keys:
                    key = (idx, expectation.name, vid)
                    if key in fired:
                        continue
                    fired.add(key)
                    # Fan out before handler so listeners (including
                    # AuditLog) see every violation regardless of which
                    # handler is configured.
                    await self._fan_out(
                        "on_expectation_fired",
                        channel_id,
                        expectation,
                        violation,
                    )
                    # Sweeper must survive handler exceptions — they
                    # leave the violation marked as fired so we don't
                    # spin on a bad handler.
                    with contextlib.suppress(Exception):
                        await handler.handle(self, channel_id, violation)
                    if expectation.on_violation == "auto_close":
                        # Channel is terminal — no further violations on
                        # this channel are meaningful this tick. Other
                        # channels still get evaluated.
                        terminal = True
                        break
                if terminal:
                    break

    # ── Registration ────────────────────────────────────────────────────────

    async def register(
        self,
        passport: Passport,
        resume: Resume,
        *,
        skill_md: str | None = None,
        rule: Rule | None = None,
    ) -> Passport:
        adapter = self._auth.get(passport.auth.scheme)
        await adapter.validate(passport, passport.auth.claim)

        async with self._registration_lock:
            # Reject a re-register that collides on ``name``: the prior
            # registration's passport / resume / rule / SKILL.md would
            # be orphaned on disk under a now-unreachable agent_id.
            # Tenants must explicitly ``unregister`` first.
            if passport.name in self._name_to_id:
                raise ProtocolError(
                    f"name {passport.name!r} already registered "
                    f"(agent_id={self._name_to_id[passport.name]}); "
                    "unregister it before re-registering."
                )
            agent_id = make_id()
            passport.agent_id = agent_id
            passport.created_at = self._clock()

            effective_rule = rule if rule is not None else Rule()

            await self._persist_passport(passport)
            await self._persist_resume(agent_id, resume)
            await self._persist_rule(agent_id, effective_rule)
            if skill_md is not None:
                await self._persist_skill(agent_id, skill_md)

            self._passports[agent_id] = passport
            self._resumes[agent_id] = resume
            self._rules[agent_id] = effective_rule
            if skill_md is not None:
                self._skills[agent_id] = skill_md
            self._name_to_id[passport.name] = agent_id

            for cap in resume.claimed_capabilities:
                self._capability_index.setdefault(cap, set()).add(agent_id)
            for cap in resume.observed:
                self._capability_index.setdefault(cap, set()).add(agent_id)

        await self._persist_capability_index()
        logger.info("agent registered: name=%s agent_id=%s", passport.name, agent_id)
        await self._fan_out(
            "on_agent_event",
            agent_id,
            "registered",
            {"passport": passport, "at": self._clock()},
        )
        return passport

    async def unregister(self, agent_id: str) -> None:
        if agent_id not in self._passports:
            raise NotFoundError(f"agent not registered: {agent_id}")

        async with self._registration_lock:
            passport = self._passports.pop(agent_id, None)
            self._resumes.pop(agent_id, None)
            self._rules.pop(agent_id, None)
            self._skills.pop(agent_id, None)
            if passport is not None and self._name_to_id.get(passport.name) == agent_id:
                self._name_to_id.pop(passport.name, None)

            endpoint_id = self._agent_to_endpoint.pop(agent_id, None)
            if endpoint_id is not None:
                bound = self._endpoint_to_agents.get(endpoint_id)
                if bound is not None:
                    bound.discard(agent_id)
                    if not bound:
                        self._endpoint_to_agents.pop(endpoint_id, None)

            # Drop the agent from every capability bucket; clean empty
            # buckets so the index stays compact.
            empty_caps: list[str] = []
            for cap, ids in self._capability_index.items():
                ids.discard(agent_id)
                if not ids:
                    empty_caps.append(cap)
            for cap in empty_caps:
                self._capability_index.pop(cap, None)

            # Delete on-disk identity files. Without this the next
            # ``hydrate()`` would re-load the unregistered agent from
            # disk. Channels and tasks the agent participated in are
            # kept for audit / read; only the per-agent identity files
            # are removed.
            await self._store.delete(passport_path(agent_id))
            await self._store.delete(resume_path(agent_id))
            await self._store.delete(rule_path(agent_id))
            await self._store.delete(skill_path(agent_id))

            # Drop inbox accounting so a future re-register with a
            # different agent_id starts from zero.
            self._inbox_pending.pop(agent_id, None)

        await self._persist_capability_index()
        logger.info("agent unregistered: agent_id=%s", agent_id)
        await self._fan_out(
            "on_agent_event",
            agent_id,
            "unregistered",
            {"name": passport.name if passport is not None else None, "at": self._clock()},
        )

    # ── Discovery (read-side) ────────────────────────────────────────────────

    async def get_agent(self, name_or_id: str) -> Passport:
        agent_id = self._name_to_id.get(name_or_id, name_or_id)
        passport = self._passports.get(agent_id)
        if passport is None:
            raise NotFoundError(f"agent not found: {name_or_id}")
        return passport

    async def get_resume(self, agent_id: str) -> Resume:
        resume = self._resumes.get(agent_id)
        if resume is None:
            raise NotFoundError(f"resume not found: {agent_id}")
        return resume

    async def get_skill(self, agent_id: str) -> str | None:
        if agent_id in self._skills:
            return self._skills[agent_id]
        if agent_id not in self._passports:
            return None
        body = await self._store.read(skill_path(agent_id))
        if body is not None:
            self._skills[agent_id] = body
        return body

    async def list_agents(
        self,
        *,
        capability: str | None = None,
        query: str | None = None,
        kind: str | None = None,
        sort_by: str | None = None,
        limit: int = 50,
    ) -> list[Passport]:
        """Enumerate registered participants, optionally filtered.

        ``kind`` filters by ``Passport.kind`` (``"agent"`` / ``"human"`` /
        ``"remote_agent"``). ``None`` returns all kinds (current default
        behavior); passing ``"agent"`` also matches passports with
        ``kind=None`` since ``None`` is the back-compat alias.
        """
        results: list[Passport] = []
        query_lower = query.lower() if query else None
        for agent_id, passport in self._passports.items():
            if kind is not None:
                resolved = passport.kind or "agent"
                if resolved != kind:
                    continue
            if capability is not None:
                resume = self._resumes.get(agent_id)
                if resume is None:
                    continue
                claimed = set(resume.claimed_capabilities)
                observed = set(resume.observed.keys())
                if capability not in claimed and capability not in observed:
                    continue
            if query_lower is not None:
                resume = self._resumes.get(agent_id)
                summary = resume.summary.lower() if resume else ""
                if query_lower not in summary:
                    continue
            results.append(passport)

        if sort_by == "name":
            results.sort(key=lambda p: p.name)

        return results[:limit]

    # ── Mutation ────────────────────────────────────────────────────────────

    async def set_resume(self, agent_id: str, resume: Resume) -> None:
        if agent_id not in self._passports:
            raise NotFoundError(f"agent not registered: {agent_id}")
        resume.last_updated = self._clock()
        resume.version = (self._resumes[agent_id].version + 1) if agent_id in self._resumes else resume.version

        # Diff capabilities so the index stays in sync. Without this,
        # a tenant adding a new claim via ``set_resume`` would not
        # surface under ``peers(action="find", capability=...)`` until
        # the agent re-registered or recorded an observation.
        old_resume = self._resumes.get(agent_id)
        old_caps: set[str] = set()
        if old_resume is not None:
            old_caps.update(old_resume.claimed_capabilities)
            old_caps.update(old_resume.observed.keys())
        new_caps: set[str] = set(resume.claimed_capabilities) | set(resume.observed.keys())

        await self._persist_resume(agent_id, resume)
        self._resumes[agent_id] = resume

        added = new_caps - old_caps
        removed = old_caps - new_caps
        for cap in added:
            self._capability_index.setdefault(cap, set()).add(agent_id)
        for cap in removed:
            bucket = self._capability_index.get(cap)
            if bucket is None:
                continue
            bucket.discard(agent_id)
            if not bucket:
                self._capability_index.pop(cap, None)
        if added or removed:
            await self._persist_capability_index()

        await self._fan_out(
            "on_agent_event",
            agent_id,
            "resume_set",
            {"resume": resume, "version": resume.version, "at": self._clock()},
        )

    async def set_skill(self, agent_id: str, skill_md: str | None) -> None:
        if agent_id not in self._passports:
            raise NotFoundError(f"agent not registered: {agent_id}")
        if skill_md is None:
            await self._store.delete(skill_path(agent_id))
            self._skills.pop(agent_id, None)
        else:
            await self._persist_skill(agent_id, skill_md)
            self._skills[agent_id] = skill_md
        await self._fan_out(
            "on_agent_event",
            agent_id,
            "skill_set",
            {"removed": skill_md is None, "at": self._clock()},
        )

    async def set_rule(self, agent_id: str, rule: Rule) -> None:
        if agent_id not in self._passports:
            raise NotFoundError(f"agent not registered: {agent_id}")
        rule.version = (self._rules[agent_id].version + 1) if agent_id in self._rules else rule.version
        await self._persist_rule(agent_id, rule)
        self._rules[agent_id] = rule
        await self._fan_out(
            "on_agent_event",
            agent_id,
            "rule_set",
            {"rule": rule, "version": rule.version, "at": self._clock()},
        )

    async def record_observation(
        self,
        *,
        owner_id: str,
        capability: str,
        outcome: TaskState,
        latency_ms: int | None = None,
        task_id: str | None = None,
    ) -> None:
        """Update ``Resume.observed[capability]`` from a terminal task event.

        Called by ``TaskMirror`` when an owner's task ends with a
        ``capability`` tag set on its ``TaskSpec``. Updates the
        capability index so the agent appears under that capability
        even if it wasn't in their original ``claimed_capabilities``.

        Outcome must be one of the terminal task states
        (``COMPLETED`` / ``FAILED`` / ``EXPIRED``); other states are
        ignored. ``latency_ms``, when provided, replaces the prior
        ``p50_latency_ms`` (single-sample stand-in for a future
        reservoir).

        ``task_id`` (when provided) is used to dedup: a single task
        contributing twice to ``Resume.observed.n`` (e.g. cascade
        EXPIRED + owner-emitted COMPLETED) is recorded only once.
        """
        if outcome not in TERMINAL_TASK_STATES:
            return
        if task_id is not None and task_id in self._observed_task_ids:
            return
        resume = self._resumes.get(owner_id)
        if resume is None:
            return
        stat = resume.observed.get(capability) or ObservedStat()
        stat.n += 1
        if outcome == TaskState.COMPLETED:
            stat.completed += 1
        elif outcome == TaskState.FAILED:
            stat.failed += 1
        elif outcome == TaskState.EXPIRED:
            stat.expired += 1
        if latency_ms is not None:
            stat.p50_latency_ms = latency_ms
        resume.observed[capability] = stat
        resume.last_updated = self._clock()
        resume.version += 1
        await self._persist_resume(owner_id, resume)

        bucket = self._capability_index.setdefault(capability, set())
        if owner_id not in bucket:
            bucket.add(owner_id)
            await self._persist_capability_index()

        if task_id is not None:
            self._observed_task_ids.add(task_id)

        await self._fan_out(
            "on_agent_event",
            owner_id,
            "observation_recorded",
            {
                "resume": resume,
                "version": resume.version,
                "capability": capability,
                "outcome": outcome.value,
                "at": self._clock(),
            },
        )

    def agents_with_capability(self, capability: str) -> list[str]:
        """Return agent_ids matching ``capability`` (claimed or observed)."""
        return sorted(self._capability_index.get(capability, set()))

    # ── Channels ────────────────────────────────────────────────────────────

    async def create_channel(
        self,
        *,
        creator_id: str,
        manifest_type: str,
        manifest_version: int = 1,
        participants: list[str],
        required_acks: int | None = None,
        ttl: str | int | None = None,
        knobs: dict[str, object] | None = None,
        intent: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> ChannelMetadata:
        """Allocate ``channel_id``, post invites, await acks, return metadata.

        Posts ``EV_CHANNEL_INVITE`` to every invitee, awaits an
        ``EV_CHANNEL_INVITE_ACK`` from each (the handshake is
        all-or-nothing — any reject fails creation), transitions to
        ``ACTIVE``, and broadcasts ``EV_CHANNEL_OPENED``. Times out
        after ``invite_ack_timeout`` if the acks do not arrive.
        """
        if creator_id not in self._passports:
            raise NotFoundError(f"creator not registered: {creator_id}")
        if not participants:
            raise ProtocolError("channel requires at least one participant")
        seen: set[str] = set()
        for p_id in participants:
            if p_id in seen:
                raise ProtocolError(f"participant listed twice: {p_id!r}")
            seen.add(p_id)
            if p_id != creator_id and p_id not in self._passports:
                raise NotFoundError(f"participant not registered: {p_id}")
        if creator_id not in participants:
            participants = [creator_id, *participants]

        adapter = self._adapter_for(manifest_type, manifest_version)

        # Apply ``expectations`` knob: replace the adapter's default manifest
        # expectations with the caller-supplied list.  Any other knob keys
        # are passed through to the adapter as-is.
        raw_expectations = knobs.get("expectations") if knobs else None
        if raw_expectations is not None:
            overridden_expectations = [Expectation(**e) if isinstance(e, dict) else e for e in raw_expectations]
            effective_manifest: ChannelManifest = dataclasses.replace(
                adapter.manifest, expectations=overridden_expectations
            )
        else:
            effective_manifest = adapter.manifest

        creator_passport = self._passports[creator_id]
        creator_rule = self._rules.get(creator_id, Rule())

        # ── Authorize channel open (arbiter) ────────────────────────────
        # Pre-flight invitee inbound-access check + creator concurrency
        # cap. The dispatch path silently filters envelopes whose
        # sender is not in the recipient's whitelist; without a
        # pre-check, an invite to a blocking recipient would be
        # dropped and the creator would hang on the ack waiter.
        invitee_passports: list[Passport] = []
        invitee_rules: list[Rule] = []
        for p_id in participants:
            if p_id == creator_id:
                continue
            invitee_passports.append(self._passports[p_id])
            invitee_rules.append(self._rules.get(p_id, Rule()))
        active_creator_channels = sum(1 for m in self._active_channels.values() if m.creator_id == creator_id)
        decision = await self._arbiter.authorize_channel_open(
            effective_manifest,
            creator_passport,
            creator_rule,
            invitee_passports,
            invitee_rules,
            active_creator_channels,
        )
        if isinstance(decision, Deny):
            raise decision.error(decision.reason)

        channel_id = make_id()
        now = self._clock()

        ttl_value: str | int = ttl if ttl is not None else creator_rule.limits.channel_ttl_default
        ttl_seconds = parse_duration(ttl_value)
        expires_at = _expires_at(now, ttl_seconds) or None

        metadata_participants: list[Participant] = []
        for index, p_id in enumerate(participants):
            if p_id == creator_id:
                role = ParticipantRole.INITIATOR
            elif len(participants) == 2:
                role = ParticipantRole.RESPONDENT
            else:
                role = ParticipantRole.PARTICIPANT
            metadata_participants.append(Participant(agent_id=p_id, role=role, order=index, joined_at=now))

        final_labels: dict[str, str] = dict(labels) if labels else {}
        if intent:
            final_labels["intent"] = intent

        invitees = [p_id for p_id in participants if p_id != creator_id]

        metadata = ChannelMetadata(
            channel_id=channel_id,
            manifest=effective_manifest,
            creator_id=creator_id,
            participants=metadata_participants,
            state=ChannelState.PENDING,
            created_at=now,
            expires_at=expires_at,
            knobs=dict(knobs) if knobs else {},
            labels=final_labels,
            required_acks=required_acks,
            pending_acks=list(invitees),
        )

        adapter.validate_create(metadata)

        # Activate caches before persistence so the post_envelope path
        # finds the metadata when the invite is dispatched.
        self._channels[channel_id] = metadata
        self._active_channels[channel_id] = metadata
        self._adapter_states[channel_id] = adapter.initial_state(metadata)

        await self._persist_channel_metadata(metadata)
        logger.info(
            "channel created: id=%s type=%s creator=%s participants=%d",
            channel_id,
            manifest_type,
            creator_id,
            len(metadata_participants),
        )
        await self._fan_out(
            "on_channel_event",
            channel_id,
            "created",
            {
                "metadata": metadata,
                "participants": [p.agent_id for p in metadata_participants],
                "at": now,
            },
        )

        if not invitees:
            # Self-only channel — already complete; transition to ACTIVE.
            await self._activate_channel(channel_id)
            return metadata

        waiter: asyncio.Future[ChannelMetadata] = asyncio.get_event_loop().create_future()
        self._channel_open_waiters[channel_id] = waiter

        # Post invites — each goes to one invitee via post_envelope.
        invite_data: dict[str, object] = {
            "channel_id": channel_id,
            "manifest_type": manifest_type,
            "manifest_version": manifest_version,
            "creator_id": creator_id,
            "knobs": metadata.knobs,
            "labels": metadata.labels,
        }
        try:
            for invitee_id in invitees:
                envelope = Envelope(
                    channel_id=channel_id,
                    sender_id=creator_id,
                    audience=[invitee_id],
                    event_type=EV_CHANNEL_INVITE,
                    event_data=invite_data,
                )
                await self.post_envelope(envelope)
        except Exception:
            self._channel_open_waiters.pop(channel_id, None)
            await self._transition_channel(channel_id, ChannelState.CLOSED, "invite_failed")
            raise

        try:
            return await asyncio.wait_for(waiter, timeout=self._invite_ack_timeout)
        except asyncio.TimeoutError as exc:
            await self._transition_channel(channel_id, ChannelState.CLOSED, "invite_timeout")
            raise ProtocolError(f"channel {channel_id!r} ack timeout") from exc
        finally:
            self._channel_open_waiters.pop(channel_id, None)

    async def close_channel(self, channel_id: str, *, reason: str = "") -> ChannelMetadata:
        await self._transition_channel(channel_id, ChannelState.CLOSED, reason or "explicit_close")
        return self._channels[channel_id]

    async def get_channel(self, channel_id: str) -> ChannelMetadata:
        metadata = self._channels.get(channel_id)
        if metadata is None:
            raise NotFoundError(f"channel not found: {channel_id}")
        return metadata

    def can_send(
        self,
        channel_id: str,
        sender_id: str,
        *,
        event_type: str | None = None,
    ) -> bool:
        """True if the adapter would accept a substantive send from
        ``sender_id`` against the current state.

        Wraps ``adapter.validate_send`` with a probe envelope so the
        default notify handler doesn't need to reach into private hub
        state to figure out whether it's the agent's turn.
        """
        metadata = self._channels.get(channel_id)
        if metadata is None or metadata.is_terminal():
            return False
        state = self._adapter_states.get(channel_id)
        if state is None:
            return False
        adapter = self._adapter_for(metadata.manifest.type, metadata.manifest.version)
        probe = Envelope(
            channel_id=channel_id,
            sender_id=sender_id,
            audience=None,
            event_type=event_type or EV_TEXT,
            event_data={"text": ""},
        )
        try:
            adapter.validate_send(metadata, probe, state)
        except Exception:
            return False
        return True

    def default_view_policy(
        self,
        channel_id: str,
        participant_id: str,
    ) -> "ViewPolicy":
        """Return the adapter-declared default view policy for this
        participant on this channel. Wraps
        ``adapter.default_view_policy`` so callers don't need adapter
        registry access."""
        metadata = self._channels.get(channel_id)
        if metadata is None:
            raise NotFoundError(f"channel not found: {channel_id}")
        adapter = self._adapter_for(metadata.manifest.type, metadata.manifest.version)
        return adapter.default_view_policy(metadata, participant_id)

    def adapter_for(self, channel_id: str) -> ChannelAdapter:
        """Return the adapter resolved from ``channel_id``'s manifest.

        Public surface so callers (notably the default notify handler)
        don't need to reach into ``_adapter_for(type, version)`` or the
        ``_channels`` map directly.
        """
        metadata = self._channels.get(channel_id)
        if metadata is None:
            raise NotFoundError(f"channel not found: {channel_id}")
        return self._adapter_for(metadata.manifest.type, metadata.manifest.version)

    def adapter_state(self, channel_id: str) -> object | None:
        """Return ``channel_id``'s current folded adapter state, or
        ``None`` if the channel has none cached.

        Public surface so callers don't need to reach into
        ``_adapter_states``.
        """
        return self._adapter_states.get(channel_id)

    async def list_channels(
        self,
        *,
        agent_id: str | None = None,
        state: ChannelState | None = None,
        limit: int = 50,
    ) -> list[ChannelMetadata]:
        results: list[ChannelMetadata] = []
        for metadata in self._channels.values():
            if state is not None and metadata.state != state:
                continue
            if agent_id is not None and agent_id not in metadata.participant_ids():
                continue
            results.append(metadata)
        return results[:limit]

    async def read_wal(
        self,
        channel_id: str,
        *,
        since: int = 0,
        until: int | None = None,
    ) -> list[Envelope]:
        body = await self._store.read(wal_path(channel_id))
        if not body:
            return []
        envelopes: list[Envelope] = []
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            envelopes.append(Envelope.from_json(line))
        end = len(envelopes) if until is None else until
        return envelopes[since:end]

    # ── Tasks (observe-only) ────────────────────────────────────────────────

    async def observe_task(self, metadata: TaskMetadata) -> None:
        """Register a task observed via the agent's stream.

        Hub does not create, assign, or cancel — it stores
        ``TaskMetadata``, persists it, and starts TTL accounting.

        On first observation, enforces the owner's
        ``Rule.limits.max_concurrent_tasks`` cap (``0`` disables).
        """
        if metadata.task_id in self._tasks:
            # Update in place — owner re-emitting TaskStarted on retry, etc.
            existing = self._tasks[metadata.task_id]
            existing.state = metadata.state
            existing.started_at = metadata.started_at or existing.started_at
            existing.expires_at = metadata.expires_at or existing.expires_at
            existing.channel_id = metadata.channel_id or existing.channel_id
            existing.progress.update(metadata.progress)
            await self._persist_task_metadata(existing)
            return

        owner_rule = self._rules.get(metadata.owner_id, Rule())
        max_tasks = owner_rule.limits.max_concurrent_tasks
        if max_tasks > 0:
            active = sum(
                1
                for t in self._tasks.values()
                if t.owner_id == metadata.owner_id and t.state not in TERMINAL_TASK_STATES
            )
            if active >= max_tasks:
                raise AccessDeniedError(
                    f"owner {metadata.owner_id!r} exceeded max_concurrent_tasks ({active} >= {max_tasks})"
                )

        self._tasks[metadata.task_id] = metadata
        if metadata.channel_id:
            self._channel_tasks.setdefault(metadata.channel_id, set()).add(metadata.task_id)
        await self._persist_task_metadata(metadata)

    async def get_task(self, task_id: str) -> TaskMetadata:
        metadata = self._tasks.get(task_id)
        if metadata is None:
            raise NotFoundError(f"task not found: {task_id}")
        return metadata

    async def update_task(
        self,
        task_id: str,
        *,
        state: TaskState | None = None,
        progress: dict[str, object] | None = None,
        result: object | None = None,
        error: str | None = None,
    ) -> None:
        """Update an observed task's lifecycle. Used by ``task_mirror``.

        Terminal-state transitions stamp ``completed_at``. Idempotent —
        terminal-on-terminal is a no-op (further events ignored).
        """
        metadata = self._tasks.get(task_id)
        if metadata is None:
            raise NotFoundError(f"task not found: {task_id}")
        if metadata.state in TERMINAL_TASK_STATES:
            return
        if progress:
            metadata.progress.update(progress)
            metadata.last_progress_at = self._clock()
        if result is not None:
            metadata.result = result
        if error:
            metadata.error = error
        if state is not None:
            metadata.state = state
            if state in TERMINAL_TASK_STATES:
                metadata.completed_at = self._clock()
        await self._persist_task_metadata(metadata)

    async def list_tasks(
        self,
        *,
        agent_id: str | None = None,
        channel_id: str | None = None,
        state: TaskState | None = None,
        limit: int = 50,
    ) -> list[TaskMetadata]:
        results: list[TaskMetadata] = []
        for metadata in self._tasks.values():
            if agent_id is not None and metadata.owner_id != agent_id:
                continue
            if channel_id is not None and metadata.channel_id != channel_id:
                continue
            if state is not None and metadata.state != state:
                continue
            results.append(metadata)
        return results[:limit]

    # ── Sweeper hook ────────────────────────────────────────────────────────

    async def expire_due(self) -> None:
        """Walk active channels and tasks; expire ones past their TTL.

        Cascades non-terminal tasks under closing channels (via
        :meth:`_transition_channel`).
        """
        now = self._clock()

        expired_channels: list[str] = []
        for channel_id, metadata in list(self._active_channels.items()):
            if metadata.expires_at and metadata.expires_at <= now:
                expired_channels.append(channel_id)
        for channel_id in expired_channels:
            await self._transition_channel(channel_id, ChannelState.EXPIRED, "ttl_expired")

        # Expire standalone tasks (those not under an expiring channel).
        expired_tasks: list[str] = []
        for task_id, metadata in list(self._tasks.items()):
            if metadata.state in TERMINAL_TASK_STATES:
                continue
            if metadata.expires_at and metadata.expires_at <= now:
                expired_tasks.append(task_id)
        for task_id in expired_tasks:
            await self._transition_task(task_id, TaskState.EXPIRED, "ttl_expired")

    # ── Envelope dispatch ───────────────────────────────────────────────────

    async def post_envelope(self, envelope: Envelope) -> str:
        """Validate sender + adapter + WAL append + dispatch.

        Per-channel lock makes ``validate_send`` / ``fold`` /
        ``on_accepted`` see a consistent state. Dispatch and post-accept
        transitions happen outside the lock so the broadcast of
        ``EV_CHANNEL_CLOSED`` does not deadlock on the same lock.

        Access / limits decisions go through :attr:`arbiter` so
        federation / custom permission protocols can replace the default
        rule-based behavior without forking the hub. Hub fires
        :meth:`HubListener.on_envelope_posted` (success) or
        :meth:`on_envelope_rejected` (any pre-WAL failure) for every
        attempt.
        """
        try:
            envelope_id = await self._post_envelope_impl(envelope)
        except NetworkError as exc:
            await self._fan_out("on_envelope_rejected", envelope, exc)
            logger.warning(
                "post_envelope rejected: channel=%s sender=%s event=%s reason=%s",
                envelope.channel_id,
                envelope.sender_id,
                envelope.event_type,
                exc,
            )
            raise
        return envelope_id

    async def _post_envelope_impl(self, envelope: Envelope) -> str:
        sender = self._passports.get(envelope.sender_id)
        if sender is None:
            raise NotFoundError(f"sender not registered: {envelope.sender_id}")

        sender_rule = self._rules.get(envelope.sender_id, Rule())

        # ── Outbound access + delegation depth (arbiter) ────────────────
        # Only explicit-audience sends go through the outbound gate.
        # Broadcasts (audience=None) skip — protocol broadcasts like
        # ``EV_CHANNEL_OPENED`` include the creator in their own audience.
        explicit_recipients: list[Passport] = []
        if envelope.audience is not None:
            for recipient_id in envelope.audience:
                if recipient_id == envelope.sender_id:
                    continue
                recipient = self._passports.get(recipient_id)
                if recipient is not None:
                    explicit_recipients.append(recipient)
        decision = await self._arbiter.authorize_send(envelope, sender, sender_rule, explicit_recipients)
        if isinstance(decision, Deny):
            raise decision.error(decision.reason)

        metadata = self._channels.get(envelope.channel_id)
        if metadata is None:
            raise NotFoundError(f"channel not found: {envelope.channel_id}")
        if metadata.is_terminal():
            raise ProtocolError(f"channel {envelope.channel_id!r} is {metadata.state.value}")
        if not _is_protocol_event(envelope.event_type) and metadata.state != ChannelState.ACTIVE:
            raise ProtocolError(f"channel {envelope.channel_id!r} not active (state={metadata.state.value})")

        # Adapter must be registered to dispatch on this channel.
        # Distinct from create_channel's NotFoundError (where the user
        # is asking for an unknown manifest at channel-creation time):
        # here the channel exists but its manifest's adapter is no
        # longer loaded — typically a hydrate where the manifest type
        # wasn't re-registered before ``hub.start()``. Surface as a
        # ProtocolError so callers can distinguish "channel is down"
        # from "channel never existed."
        if (metadata.manifest.type, metadata.manifest.version) not in self._adapters:
            raise ProtocolError(
                f"channel {envelope.channel_id!r} has no registered adapter "
                f"(manifest {metadata.manifest.type!r}@v{metadata.manifest.version})"
            )

        # ── Inbox capacity (arbiter, substantive events only) ───────────
        # Protocol invites / acks / opens / closes must always reach
        # participants for the channel machine to advance.
        if not _is_protocol_event(envelope.event_type):
            if envelope.audience is not None:
                inbox_audience: list[str] = list(envelope.audience)
            else:
                inbox_audience = [p.agent_id for p in metadata.participants if p.agent_id != envelope.sender_id]
            for recipient_id in inbox_audience:
                if recipient_id == envelope.sender_id:
                    continue
                recipient = self._passports.get(recipient_id)
                recipient_rule = self._rules.get(recipient_id)
                if recipient is None or recipient_rule is None:
                    continue
                current = self._inbox_pending.get(recipient_id, 0)
                inbox_decision = await self._arbiter.authorize_inbox(envelope, recipient, recipient_rule, current)
                if isinstance(inbox_decision, Deny):
                    raise inbox_decision.error(inbox_decision.reason)

        adapter = self._adapter_for(metadata.manifest.type, metadata.manifest.version)

        # Critical section: validate, append, fold, on_accepted under lock.
        async with self._wal_lock(envelope.channel_id):
            state = self._adapter_states.get(envelope.channel_id)
            if state is None:
                # Channel metadata exists but its adapter state was
                # never folded — typically a hydrate where the manifest's
                # adapter was not registered. Surface as a protocol
                # error rather than a bare KeyError.
                raise ProtocolError(
                    f"channel {envelope.channel_id!r} has no adapter state "
                    f"(manifest {metadata.manifest.type!r}@v{metadata.manifest.version} "
                    "may not be registered)"
                )
            adapter.validate_send(metadata, envelope, state)

            envelope.envelope_id = make_id()
            envelope.created_at = self._clock()

            await self._wal_append(envelope)
            new_state = adapter.fold(envelope, state)
            self._adapter_states[envelope.channel_id] = new_state
            result = adapter.on_accepted(metadata, envelope, new_state)

        # Sender showed they're processing their inbox — decrement
        # their outstanding count. Substantive events only; protocol
        # acks and opens shouldn't drain inbox accounting.
        if not _is_protocol_event(envelope.event_type):
            current = self._inbox_pending.get(envelope.sender_id, 0)
            if current > 0:
                self._inbox_pending[envelope.sender_id] = current - 1

        # Outside lock: dispatch + post-accept handling.
        # Acks/rejects are absorbed by the hub — they aren't dispatched.
        if envelope.event_type == EV_CHANNEL_INVITE_ACK:
            await self._handle_invite_ack(envelope, metadata)
            return envelope.envelope_id
        if envelope.event_type == EV_CHANNEL_INVITE_REJECT:
            await self._handle_invite_reject(envelope, metadata)
            return envelope.envelope_id

        await self._dispatch(envelope, metadata)

        # Listener fan-out — read-only state-transition notification.
        # Fires for every successfully posted envelope (including
        # protocol events like invites / acks / opens / closes) so
        # observers see the full event stream.
        await self._fan_out("on_envelope_posted", envelope, metadata)
        logger.debug(
            "post_envelope ok: channel=%s sender=%s event=%s",
            envelope.channel_id,
            envelope.sender_id,
            envelope.event_type,
        )

        if result.next_state is not None:
            await self._transition_channel(
                envelope.channel_id,
                result.next_state,
                result.auto_close_reason,
            )

        return envelope.envelope_id

    # ── Endpoint management ─────────────────────────────────────────────────

    def attach_endpoint(self, endpoint: LinkEndpoint) -> None:
        if self._closed:
            return
        self._endpoints_by_id[endpoint.endpoint_id] = endpoint
        task = asyncio.create_task(self._handle_endpoint(endpoint))
        self._endpoint_tasks.add(task)
        task.add_done_callback(self._endpoint_tasks.discard)

    def bind_endpoint(self, endpoint_id: str, agent_id: str) -> None:
        if endpoint_id not in self._endpoints_by_id:
            raise NotFoundError(f"endpoint not attached: {endpoint_id}")
        if agent_id not in self._passports:
            raise NotFoundError(f"agent not registered: {agent_id}")
        self._endpoints_by_id[endpoint_id].agent_id = agent_id
        self._agent_to_endpoint[agent_id] = endpoint_id
        self._endpoint_to_agents.setdefault(endpoint_id, set()).add(agent_id)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _wal_lock(self, channel_id: str) -> asyncio.Lock:
        lock = self._channel_locks.get(channel_id)
        if lock is None:
            lock = asyncio.Lock()
            self._channel_locks[channel_id] = lock
        return lock

    async def _wal_append(self, envelope: Envelope) -> None:
        await self._store.append(wal_path(envelope.channel_id), envelope.to_json() + "\n")

    async def _dispatch(self, envelope: Envelope, metadata: ChannelMetadata) -> None:
        """Send NotifyFrames to the audience (or all participants if broadcast).

        Inbound access is consulted via :meth:`HubArbiter.authorize_dispatch`
        per recipient. ``Deny`` causes the hub to silently skip that
        recipient — the rest of the audience still receives. Unknown
        audience ids are routed through
        :meth:`HubArbiter.resolve_unknown_audience` so federation hooks
        can replace them with locally-deliverable proxies.
        """
        if envelope.audience is None:
            recipients = [p.agent_id for p in metadata.participants if p.agent_id != envelope.sender_id]
        else:
            recipients = list(envelope.audience)

        sender_passport = self._passports.get(envelope.sender_id)
        substantive = not _is_protocol_event(envelope.event_type)

        # Federation hook for unknown audience ids. Default arbiter
        # returns None (drop silently). A federated arbiter can redirect
        # to a local proxy id that forwards to the remote hub.
        unknown = [rid for rid in recipients if rid not in self._passports and rid != envelope.sender_id]
        if unknown:
            replacement = await self._arbiter.resolve_unknown_audience(envelope, unknown)
            if replacement is not None:
                recipients = [rid for rid in recipients if rid not in unknown] + list(replacement)

        for recipient_id in recipients:
            if recipient_id == envelope.sender_id:
                # Self-routing always delivers (protocol broadcasts
                # include the sender so the sender's Channel handle
                # sees the lifecycle event).
                pass
            elif sender_passport is not None:
                recipient_passport = self._passports.get(recipient_id)
                recipient_rule = self._rules.get(recipient_id)
                if recipient_passport is not None and recipient_rule is not None:
                    decision = await self._arbiter.authorize_dispatch(
                        envelope, sender_passport, recipient_passport, recipient_rule
                    )
                    if isinstance(decision, Deny):
                        continue
            endpoint = self._endpoint_for(recipient_id)
            if endpoint is None:
                continue
            if substantive:
                prev_count = self._inbox_pending.get(recipient_id, 0)
                new_count = prev_count + 1
                self._inbox_pending[recipient_id] = new_count
                await self._maybe_fire_inbox_pressure(recipient_id, prev_count, new_count)
            try:
                await endpoint.send_frame(NotifyFrame(envelope=envelope, recipient_id=recipient_id))
            except Exception as exc:
                await self._fan_out("on_dispatch_failed", envelope, recipient_id, exc)
                logger.warning(
                    "dispatch failed: channel=%s recipient=%s event=%s reason=%s",
                    envelope.channel_id,
                    recipient_id,
                    envelope.event_type,
                    exc,
                )

    async def _maybe_fire_inbox_pressure(self, recipient_id: str, prev: int, new: int) -> None:
        """Fire ``on_inbox_pressure`` when crossing the high-water mark.

        Fires exactly once per crossing — does not re-fire on every
        subsequent envelope while above the mark. Resolves ``high_water``:
        explicit value → that; ``None`` → 80% of ``max_pending``;
        ``max_pending == 0`` → no signal.
        """
        rule = self._rules.get(recipient_id)
        if rule is None:
            return
        cap = rule.limits.inbox.max_pending
        if cap <= 0:
            return
        hw_config = rule.limits.inbox.high_water
        if hw_config is None:
            high_water = max(1, int(cap * 0.8))
        elif hw_config <= 0:
            return
        else:
            high_water = hw_config
        if prev < high_water <= new:
            await self._fan_out("on_inbox_pressure", recipient_id, new, cap)

    def _endpoint_for(self, agent_id: str) -> LinkEndpoint | None:
        endpoint_id = self._agent_to_endpoint.get(agent_id)
        if endpoint_id is None:
            return None
        return self._endpoints_by_id.get(endpoint_id)

    async def _handle_endpoint(self, endpoint: LinkEndpoint) -> None:
        try:
            async for frame in endpoint.frames():
                await self._dispatch_frame(endpoint, frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _dispatch_frame(self, endpoint: LinkEndpoint, frame: Frame) -> None:
        if isinstance(frame, SendFrame):
            try:
                envelope_id = await self.post_envelope(frame.envelope)
                await endpoint.send_frame(AcceptFrame(envelope_id=envelope_id))
            except NetworkError as exc:
                await endpoint.send_frame(ErrorFrame(code=_error_code(exc), message=str(exc)))
        elif isinstance(frame, HelloFrame):
            agent_id = self._name_to_id.get(frame.name)
            if agent_id is None:
                await endpoint.send_frame(ErrorFrame(code="not_found", message=f"unknown name: {frame.name}"))
                return
            try:
                self.bind_endpoint(endpoint.endpoint_id, agent_id)
            except NetworkError as exc:
                await endpoint.send_frame(ErrorFrame(code=_error_code(exc), message=str(exc)))
                return
            await endpoint.send_frame(WelcomeFrame(endpoint_id=endpoint.endpoint_id, hub_time=self._clock()))
        elif isinstance(frame, PingFrame):
            await endpoint.send_frame(PongFrame())

    # ── Channel transition helpers ──────────────────────────────────────────

    async def _handle_invite_ack(self, envelope: Envelope, metadata: ChannelMetadata) -> None:
        if metadata.state != ChannelState.PENDING:
            return
        if envelope.sender_id in metadata.pending_acks:
            metadata.pending_acks.remove(envelope.sender_id)
            await self._persist_channel_metadata(metadata)
        if not metadata.pending_acks and not metadata.rejected_by:
            await self._activate_channel(metadata.channel_id)

    async def _handle_invite_reject(self, envelope: Envelope, metadata: ChannelMetadata) -> None:
        if metadata.state != ChannelState.PENDING:
            return
        if envelope.sender_id in metadata.pending_acks:
            metadata.pending_acks.remove(envelope.sender_id)
        if envelope.sender_id not in metadata.rejected_by:
            metadata.rejected_by.append(envelope.sender_id)
        await self._persist_channel_metadata(metadata)
        # All-or-nothing handshake: any reject fails the channel.
        await self._transition_channel(metadata.channel_id, ChannelState.CLOSED, "invite_rejected")
        waiter = self._channel_open_waiters.get(metadata.channel_id)
        if waiter is not None and not waiter.done():
            waiter.set_exception(ProtocolError(f"channel rejected by {envelope.sender_id}"))

    async def _activate_channel(self, channel_id: str) -> None:
        metadata = self._channels.get(channel_id)
        if metadata is None or metadata.state != ChannelState.PENDING:
            return
        metadata.state = ChannelState.ACTIVE
        await self._persist_channel_metadata(metadata)
        logger.info("channel opened: id=%s", channel_id)
        await self._fan_out(
            "on_channel_event",
            channel_id,
            "opened",
            {"metadata": metadata},
        )
        opened_envelope = Envelope(
            channel_id=channel_id,
            sender_id=metadata.creator_id,
            audience=[p.agent_id for p in metadata.participants],
            event_type=EV_CHANNEL_OPENED,
            event_data={"channel_id": channel_id},
        )
        await self.post_envelope(opened_envelope)
        waiter = self._channel_open_waiters.get(channel_id)
        if waiter is not None and not waiter.done():
            waiter.set_result(metadata)

    async def _transition_channel(
        self,
        channel_id: str,
        new_state: ChannelState,
        reason: str,
    ) -> None:
        metadata = self._channels.get(channel_id)
        if metadata is None or metadata.is_terminal():
            return

        # Cascade non-terminal tasks before flipping channel state so
        # observers see ``ag2.task.expired`` before ``ag2.channel.closed``.
        if is_terminal_channel_state(new_state):
            for task_id in list(self._channel_tasks.get(channel_id, set())):
                task_meta = self._tasks.get(task_id)
                if task_meta is not None and task_meta.state not in TERMINAL_TASK_STATES:
                    await self._transition_task(task_id, TaskState.EXPIRED, "channel_closed")

        was_pending = metadata.state == ChannelState.PENDING
        metadata.state = new_state
        metadata.close_reason = reason
        if is_terminal_channel_state(new_state):
            metadata.closed_at = self._clock()
            self._active_channels.pop(channel_id, None)
            self._fired_violations.pop(channel_id, None)
            # Release any pending create_channel waiter so callers
            # don't hang until invite_ack_timeout when the sweeper /
            # auto_close handler closes a PENDING channel out-of-band.
            if was_pending:
                waiter = self._channel_open_waiters.pop(channel_id, None)
                if waiter is not None and not waiter.done():
                    waiter.set_exception(ProtocolError(f"channel {channel_id!r} closed during handshake: {reason}"))
            else:
                self._channel_open_waiters.pop(channel_id, None)

        await self._persist_channel_metadata(metadata)

        if is_terminal_channel_state(new_state):
            event_type = EV_CHANNEL_EXPIRED if new_state == ChannelState.EXPIRED else EV_CHANNEL_CLOSED
            close_envelope = Envelope(
                channel_id=channel_id,
                sender_id=metadata.creator_id,
                audience=[p.agent_id for p in metadata.participants],
                event_type=event_type,
                event_data={"reason": reason, "channel_id": channel_id},
            )
            close_envelope.envelope_id = make_id()
            close_envelope.created_at = self._clock()
            async with self._wal_lock(channel_id):
                await self._wal_append(close_envelope)
            await self._dispatch(close_envelope, metadata)
            kind_label = "expired" if new_state == ChannelState.EXPIRED else "closed"
            logger.info("channel %s: id=%s reason=%s", kind_label, channel_id, reason)
            await self._fan_out(
                "on_channel_event",
                channel_id,
                kind_label,
                {"reason": reason, "metadata": metadata, "at": metadata.closed_at},
            )

            # Bound long-lived hub memory: drop per-channel synchronization
            # primitives that are meaningless once the channel is terminal.
            # ``_adapter_states`` is intentionally retained — fold state has
            # analytical value (tests, debug tools, future re-fold checks)
            # and is a single dataclass per channel; only the heavier
            # ``asyncio.Lock`` is released. Done AFTER the close-envelope
            # WAL append + dispatch so we don't accidentally re-create
            # the lock we are trying to clean up.
            self._channel_locks.pop(channel_id, None)
        # ACTIVE state transitions originate from ``_activate_channel``
        # which fires the ``opened`` event directly — no need to fan
        # out again here.

    async def _transition_task(
        self,
        task_id: str,
        new_state: TaskState,
        reason: str,
    ) -> None:
        metadata = self._tasks.get(task_id)
        if metadata is None or metadata.state in TERMINAL_TASK_STATES:
            return
        metadata.state = new_state
        if new_state in TERMINAL_TASK_STATES:
            metadata.completed_at = self._clock()
            if new_state == TaskState.EXPIRED:
                metadata.error = reason or metadata.error or "expired"
        await self._persist_task_metadata(metadata)
        if new_state in TERMINAL_TASK_STATES:
            await self._fan_out(
                "on_task_event",
                task_id,
                new_state.value,
                {
                    "owner_id": metadata.owner_id,
                    "channel_id": metadata.channel_id,
                    "reason": reason,
                    "capability": metadata.spec.capability,
                    "outcome": new_state.value,
                    "at": metadata.completed_at,
                },
            )

    # ── Persistence helpers ──────────────────────────────────────────────────

    async def _persist_passport(self, passport: Passport) -> None:
        assert passport.agent_id is not None
        await self._store.write(passport_path(passport.agent_id), json.dumps(passport.to_dict()))

    async def _persist_resume(self, agent_id: str, resume: Resume) -> None:
        await self._store.write(resume_path(agent_id), json.dumps(resume.to_dict()))

    async def _persist_rule(self, agent_id: str, rule: Rule) -> None:
        await self._store.write(rule_path(agent_id), json.dumps(rule.to_dict()))

    async def _persist_skill(self, agent_id: str, skill_md: str) -> None:
        await self._store.write(skill_path(agent_id), skill_md)

    async def _persist_capability_index(self) -> None:
        # Sorted lists for deterministic JSON output.
        snapshot = {cap: sorted(ids) for cap, ids in self._capability_index.items()}
        await self._store.write(by_capability_path(), json.dumps(snapshot, sort_keys=True))

    async def _persist_channel_metadata(self, metadata: ChannelMetadata) -> None:
        await self._store.write(
            channel_metadata_path(metadata.channel_id),
            json.dumps(metadata.to_dict()),
        )

    async def _persist_task_metadata(self, metadata: TaskMetadata) -> None:
        await self._store.write(
            task_metadata_path(metadata.task_id),
            json.dumps(_task_metadata_to_dict(metadata)),
        )

    async def _load_agent(self, agent_id: str) -> None:
        passport_data = await self._store.read(passport_path(agent_id))
        if passport_data is None:
            return
        passport = Passport.from_dict(json.loads(passport_data))
        self._passports[agent_id] = passport
        self._name_to_id[passport.name] = agent_id

        resume_data = await self._store.read(resume_path(agent_id))
        if resume_data is not None:
            self._resumes[agent_id] = Resume.from_dict(json.loads(resume_data))

        rule_data = await self._store.read(rule_path(agent_id))
        if rule_data is not None:
            self._rules[agent_id] = Rule.from_dict(json.loads(rule_data))
        else:
            self._rules[agent_id] = Rule()

    async def _load_channel(self, channel_id: str) -> None:
        metadata_data = await self._store.read(channel_metadata_path(channel_id))
        if metadata_data is None:
            return
        metadata = ChannelMetadata.from_dict(json.loads(metadata_data))
        self._channels[channel_id] = metadata

        adapter = self._adapters.get((metadata.manifest.type, metadata.manifest.version))
        if adapter is None:
            # No adapter for this channel's manifest. We keep the
            # metadata in ``_channels`` (so observers can read its
            # final-or-current shape) but do **not** mark it active —
            # ``post_envelope`` would otherwise hit a missing
            # ``_adapter_states`` entry. Re-register the adapter and
            # call ``hydrate()`` again to fold its WAL.
            return

        if not metadata.is_terminal():
            self._active_channels[channel_id] = metadata

        state = adapter.initial_state(metadata)
        wal = await self.read_wal(channel_id)
        for envelope in wal:
            state = adapter.fold(envelope, state)
        self._adapter_states[channel_id] = state

    async def _load_task(self, task_id: str) -> None:
        metadata_data = await self._store.read(task_metadata_path(task_id))
        if metadata_data is None:
            return
        metadata = _task_metadata_from_dict(json.loads(metadata_data))
        self._tasks[task_id] = metadata
        if metadata.channel_id:
            self._channel_tasks.setdefault(metadata.channel_id, set()).add(task_id)


def _task_metadata_to_dict(metadata: TaskMetadata) -> dict[str, object]:
    """Serialise framework-core ``TaskMetadata`` for hub persistence.

    Mirrors the dataclass shape but coerces ``state`` (an Enum) to a
    string and ``spec`` (a ``TaskSpec`` dataclass) to a dict.
    """
    return {
        "task_id": metadata.task_id,
        "owner_id": metadata.owner_id,
        "spec": {
            "title": metadata.spec.title,
            "description": metadata.spec.description,
            "payload": dict(metadata.spec.payload),
            "capability": metadata.spec.capability,
        },
        "state": metadata.state.value,
        "created_at": metadata.created_at,
        "started_at": metadata.started_at,
        "completed_at": metadata.completed_at,
        "expires_at": metadata.expires_at,
        "last_progress_at": metadata.last_progress_at,
        "progress": dict(metadata.progress),
        "result": metadata.result,
        "error": metadata.error,
        "channel_id": metadata.channel_id,
    }


def _task_metadata_from_dict(data: dict[str, object]) -> TaskMetadata:
    spec_data = data.get("spec") or {}
    if isinstance(spec_data, dict):
        capability_raw = spec_data.get("capability")
        spec = TaskSpec(
            title=str(spec_data.get("title", "")),
            description=str(spec_data.get("description", "")),
            payload=dict(spec_data.get("payload") or {}),  # type: ignore[arg-type]
            capability=capability_raw if isinstance(capability_raw, str) else None,
        )
    else:
        spec = TaskSpec(title="")
    state_raw = data.get("state", TaskState.CREATED.value)
    state = TaskState(state_raw) if isinstance(state_raw, str) else state_raw
    return TaskMetadata(
        task_id=str(data["task_id"]),
        owner_id=str(data["owner_id"]),
        spec=spec,
        state=state,  # type: ignore[arg-type]
        created_at=str(data.get("created_at", "")),
        started_at=data.get("started_at"),  # type: ignore[arg-type]
        completed_at=data.get("completed_at"),  # type: ignore[arg-type]
        expires_at=data.get("expires_at"),  # type: ignore[arg-type]
        last_progress_at=data.get("last_progress_at"),  # type: ignore[arg-type]
        progress=dict(data.get("progress") or {}),  # type: ignore[arg-type]
        result=data.get("result"),
        error=str(data.get("error", "")),
        channel_id=data.get("channel_id"),  # type: ignore[arg-type]
    )
