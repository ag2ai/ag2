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
from typing import TYPE_CHECKING

from ag2.knowledge import KnowledgeStore
from ag2.task import TERMINAL_TASK_STATES, TaskMetadata, TaskState

from ..adapters.base import ChannelAdapter
from ..adapters.consulting import ConsultingAdapter
from ..adapters.conversation import ConversationAdapter
from ..adapters.discussion import DiscussionAdapter
from ..adapters.workflow import WorkflowAdapter
from ..auth import AuthRegistry
from ..channel import (
    ChannelMetadata,
    ChannelState,
    Participant,
    ParticipantRole,
    is_terminal_channel_state,
)
from ..client.hub_client import HubClient
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
    AuthError,
    NetworkError,
    NotFoundError,
    ProtocolError,
)
from ..identity import ObservedStat, Passport, Resume
from ..ids import _MonotonicIds, make_id
from ..remote import RemoteAgentProxy
from ..rule import Rule, parse_duration
from ..transport.frames import (
    ErrorFrame,
    Frame,
    HelloFrame,
    NotifyFrame,
    PingFrame,
    PongFrame,
    ReceiptFrame,
    RequestFrame,
    ResponseFrame,
    WelcomeFrame,
)
from ..transport.link import LinkEndpoint
from ..transport.local import LocalLink
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
    inbox_cursor_path,
    passport_path,
    resume_path,
    rule_path,
    skill_path,
    task_checkpoint_path,
    task_metadata_path,
    tasks_root,
    wal_path,
)
from .listener import HubListener
from .sweepers import _IntervalSweeper

if TYPE_CHECKING:
    # Annotation-only — the hub forwards these to ``HubClient`` and never
    # touches a tenant ``Agent`` at runtime, preserving the trust boundary.
    from ag2.agent import Agent

    from ..client.agent_client import AgentClient
    from ..transport.link import LinkFactory

__all__ = ("Hub", "PendingTurn")


logger = logging.getLogger(__name__)


@dataclasses.dataclass(slots=True, frozen=True)
class PendingTurn:
    """A turn the protocol expects this agent to take.

    ``triggering_envelope_id`` names the envelope that put the turn on
    the agent — typically the previous speaker's send. ``None`` if no
    envelope drove the expectation (e.g. fresh channel where the
    expected speaker is the creator with nothing yet posted).
    ``expected_at`` is when the turn became theirs (the triggering
    envelope's ``created_at``, or the current hub clock if no trigger
    envelope is available).
    """

    channel_id: str
    triggering_envelope_id: str | None
    expected_at: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_ERROR_CODE_MAP: dict[type, str] = {
    NotFoundError: "not_found",
    AccessDeniedError: "access_denied",
    ProtocolError: "protocol_error",
    AuthError: "auth_failed",
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

        # Per-recipient, per-channel delivery cursor:
        # ``agent_id -> {channel_id -> last-acked envelope_id}``. Each
        # channel is an independently-ordered stream (its own WAL +
        # lock), so the cursor is scoped per channel: an ack in one
        # channel must not advance the high-water mark of another, or an
        # older unacked envelope elsewhere would never replay. On
        # reconnect with ``HelloFrame.since_envelope_id`` set, the hub
        # replays, per channel, every dispatched envelope whose id sorts
        # strictly above ``max(channel_cursor, since_envelope_id)``.
        # Envelope ids come from ``self._mint_envelope_id`` (strictly
        # monotonic, time-ordered), so lexicographic compare matches
        # dispatch ordering within a channel.
        self._inbox_cursors: dict[str, dict[str, str]] = {}

        # Strictly-monotonic source for envelope ids. The cursor and
        # replay above rely on per-channel WAL order == sort order, but
        # ``time.time_ns`` can repeat within a tick on coarse-resolution
        # clocks (Windows ~15 ms), so a plain ``make_id`` would sort two
        # same-tick envelopes by their random suffix — non-deterministically.
        # This clamps each id strictly above the last so sort order always
        # tracks mint order. State is per-hub (no shared global counter).
        self._mint_envelope_id = _MonotonicIds()

        # Federation dispatch registry keyed by ``proxy.scheme``. When
        # a recipient passport has ``effective_kind == "remote_agent"``
        # the hub looks up the proxy by ``recipient.auth.scheme`` and
        # hands the envelope to ``proxy.dispatch(...)`` instead of
        # sending a ``NotifyFrame`` to a local endpoint. No proxies
        # ship in the framework; tenants register their own.
        self._remote_proxies: dict[str, RemoteAgentProxy] = {}

        # ``(channel_id, sender_id, causation_id) -> envelope_id``.
        # Lets handlers short-circuit logically-duplicate work after an
        # at-least-once redelivery: a sender that retries with the same
        # ``causation_id`` looks up the prior envelope and skips the
        # repeated side effect. Populated on every WAL append, pruned
        # when a channel transitions to a terminal state, and rebuilt
        # from active-channel WALs on ``hydrate()``.
        self._causation_index: dict[tuple[str, str, str], str] = {}

        # Transport-side state.
        self._endpoints_by_id: dict[str, LinkEndpoint] = {}
        self._agent_to_endpoint: dict[str, str] = {}
        self._endpoint_to_agents: dict[str, set[str]] = {}
        self._endpoint_tasks: set[asyncio.Task[None]] = set()

        # Clients created by the ``register(agent)`` convenience — one per
        # call, each owned by its returned ``AgentClient``. The hub tracks
        # them so ``close()`` can drain any the caller left open;
        # ``AgentClient.close`` closes (and is free to leave tracked —
        # ``HubClient.close`` is idempotent) the rest.
        self._owned_clients: list[HubClient] = []

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
        self._inbox_cursors.clear()
        self._causation_index.clear()

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
        # Close clients created by ``register(agent)`` first — closing a
        # client's link signals its hub-side endpoint loop to terminate
        # cleanly before the remaining endpoint tasks are cancelled.
        # Idempotent, so already-``AgentClient.close``d ones are no-ops.
        for client in self._owned_clients:
            await client.close()
        self._owned_clients.clear()
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

    def register_remote_proxy(self, proxy: RemoteAgentProxy) -> None:
        """Register a federation proxy keyed by ``proxy.scheme``.

        When the hub dispatches an envelope to a recipient whose
        passport has ``effective_kind == "remote_agent"``, it looks up
        the proxy by the recipient's ``auth.scheme`` and calls
        ``proxy.dispatch(envelope, recipient)`` instead of sending a
        ``NotifyFrame`` to a local endpoint. Re-registering at the
        same ``scheme`` replaces the prior proxy.
        """
        self._remote_proxies[proxy.scheme] = proxy

    def unregister_remote_proxy(self, scheme: str) -> RemoteAgentProxy | None:
        """Remove the proxy registered for ``scheme`` and return it.

        Returns ``None`` if no proxy was registered for ``scheme``.
        The caller is responsible for awaiting ``proxy.close()``
        — the hub leaves lifecycle decisions to whoever owns the
        proxy instance.
        """
        return self._remote_proxies.pop(scheme, None)

    def remote_proxy_for(self, scheme: str) -> RemoteAgentProxy | None:
        """Read-only lookup against the proxy registry."""
        return self._remote_proxies.get(scheme)

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
        agent: "Agent",
        passport: "Passport | None" = None,
        resume: "Resume | None" = None,
        *,
        skill_md: str | None = None,
        rule: Rule | None = None,
        attach_plugin: bool = True,
        link: "LinkFactory | None" = None,
    ) -> "AgentClient":
        """Register an ``Agent`` directly against this in-process hub.

        Convenience over the explicit
        ``HubClient(LocalLink(hub), hub=hub).register(agent)`` flow: each
        call creates a dedicated :class:`HubClient` (on a ``LocalLink(self)``
        by default, or the supplied ``link``) and returns the
        :class:`AgentClient` handle. The handle **owns** that connection —
        :meth:`AgentClient.close` unregisters the agent and closes it, and
        :meth:`close` closes any handles left open.

        Mirrors :meth:`HubClient.register`: ``passport`` / ``resume``
        default from ``agent.name`` / an empty :class:`Resume`, and a
        ``kind="human"`` passport is rejected with a pointer to
        ``HubClient.register_human``. The explicit ``HubClient`` flow
        remains available for custom transports and multi-agent-per-client
        topologies; to register a low-level identity without an agent, use
        :meth:`register_identity`.
        """
        hub_client = HubClient(link if link is not None else LocalLink(self), hub=self)
        self._owned_clients.append(hub_client)
        try:
            client = await hub_client.register(
                agent,
                passport,
                resume,
                skill_md=skill_md,
                rule=rule,
                attach_plugin=attach_plugin,
            )
        # BaseException (not Exception) so an ``asyncio.CancelledError``
        # raised during the ``await`` above — which is a BaseException
        # since Python 3.8 — still triggers cleanup. We re-raise it
        # unchanged below, so cancellation/shutdown is not swallowed.
        except BaseException:
            # Registration failed (e.g. a human passport) — don't leak the
            # freshly-created client; close and untrack it before re-raising.
            self._owned_clients.remove(hub_client)
            await hub_client.close()
            raise
        # The handle exclusively owns this connection, so its ``close``
        # may tear the transport down without affecting other agents.
        client._owns_client = True
        return client

    async def register_identity(
        self,
        passport: Passport,
        resume: Resume,
        *,
        skill_md: str | None = None,
        rule: Rule | None = None,
    ) -> Passport:
        """Register a low-level participant identity; return the stamped passport.

        Stamps an ``agent_id``, persists the passport / resume / rule
        (and optional SKILL.md), and indexes the name + capabilities. This
        is the identity primitive underneath :meth:`register` — most
        callers want :meth:`register`, which attaches an ``Agent`` and
        returns an ``AgentClient``.
        """
        # Remote-agent passports represent participants on another hub
        # and ``auth.scheme`` is a routing label consumed by the
        # registered ``RemoteAgentProxy`` — not a credential to be
        # validated locally. Skip the local auth check for them;
        # authentication on the wire belongs to the originating hub.
        if passport.effective_kind != "remote_agent":
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
            self._inbox_cursors.pop(agent_id, None)
            await self._store.delete(inbox_cursor_path(agent_id))

        await self._persist_capability_index()
        logger.info("agent unregistered: agent_id=%s", agent_id)
        await self._fan_out(
            "on_agent_event",
            agent_id,
            "unregistered",
            {"name": passport.name if passport is not None else None, "at": self._clock()},
        )

    # ── Discovery (read-side) ────────────────────────────────────────────────

    def name_for(self, agent_id: str, *, default: str | None = None) -> str:
        """Resolve ``agent_id`` to its registered ``Passport.name``.

        Reads the in-memory passport directory.
        Returns ``default`` when the id is unknown (or ``agent_id`` itself
        if ``default`` is ``None``), so callers can use this as a safe
        ``NameResolver`` for view projection without needing to handle
        the unregistered / unregistered-mid-turn case.
        """
        passport = self._passports.get(agent_id)
        if passport is not None:
            return passport.name
        return default if default is not None else agent_id

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

    def find_agent_id(self, name: str) -> str | None:
        """Resolve ``name`` to its registered ``agent_id``, or ``None``.

        Non-raising peer to :meth:`get_agent` — callers that need to
        branch on "is this name registered?" without catching an
        exception use this directly. Returns ``None`` when ``name``
        has no current registration.
        """
        return self._name_to_id.get(name)

    def name_to_id_map(self) -> dict[str, str]:
        """Snapshot of the ``name → agent_id`` directory.

        Public read surface so callers that need reverse name
        resolution (e.g. ``WorkflowAdapter`` resolving handoff target
        names to ids) don't reach into the private index. Returns a
        copy so mutation can't corrupt the registry.
        """
        return dict(self._name_to_id)

    async def get_rule(self, agent_id: str) -> Rule:
        """Return the rule attached to ``agent_id``.

        Raises :class:`NotFoundError` if no rule is registered — the
        registration path stamps a default :class:`Rule` for every
        agent, so a missing entry indicates the agent itself is
        unregistered.
        """
        rule = self._rules.get(agent_id)
        if rule is None:
            raise NotFoundError(f"rule not found: {agent_id}")
        return rule

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
            adapter.manifest,
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
            manifest=adapter.manifest,
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

    async def checkpoint_task(self, task_id: str, state: dict[str, object]) -> None:
        """Persist an owner-supplied resume snapshot for ``task_id``.

        Writes a single JSON blob at ``tasks/{task_id}/checkpoint.json``;
        last-write-wins, no history. The framework treats the payload as
        opaque — owners pick what to store and how to interpret it on
        resume. Pairs with :meth:`read_task_checkpoint` for the read
        side; the canonical entry point is the ``HubBackedCheckpointStore``
        on the client.
        """
        await self._store.write(task_checkpoint_path(task_id), json.dumps(state))

    async def read_task_checkpoint(self, task_id: str) -> dict[str, object] | None:
        """Read the resume snapshot for ``task_id``, or ``None`` if absent.

        Returned dict is the value the owner most recently passed to
        :meth:`checkpoint_task`. Malformed JSON on disk surfaces as an
        exception — the framework does not silently swallow corruption.
        """
        raw = await self._store.read(task_checkpoint_path(task_id))
        if raw is None:
            return None
        return json.loads(raw)

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

            envelope.envelope_id = self._mint_envelope_id()
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
        # If the agent already has an endpoint binding (reconnect from a
        # different connection), evict the prior mapping before stamping
        # the new one. The prior endpoint stays alive — other agents
        # bound to it keep working — but envelopes addressed to this
        # agent now route through the new endpoint.
        prior_endpoint_id = self._agent_to_endpoint.get(agent_id)
        if prior_endpoint_id is not None and prior_endpoint_id != endpoint_id:
            prior_bound = self._endpoint_to_agents.get(prior_endpoint_id)
            if prior_bound is not None:
                prior_bound.discard(agent_id)
                if not prior_bound:
                    self._endpoint_to_agents.pop(prior_endpoint_id, None)
            prior_endpoint = self._endpoints_by_id.get(prior_endpoint_id)
            if prior_endpoint is not None and prior_endpoint.agent_id == agent_id:
                prior_endpoint.agent_id = None
        self._endpoints_by_id[endpoint_id].agent_id = agent_id
        self._agent_to_endpoint[agent_id] = endpoint_id
        self._endpoint_to_agents.setdefault(endpoint_id, set()).add(agent_id)

    async def pending_turns_for(self, agent_id: str) -> "list[PendingTurn]":
        """Return turns in active channels where the protocol expects this agent.

        Walks every active channel the agent participates in, asks the
        registered adapter via :meth:`ChannelAdapter.expected_next`,
        and returns a :class:`PendingTurn` per channel where the agent
        is named. Channels with no specific expected speaker (free-form
        conversations) or where another participant is expected are
        skipped. The triggering envelope's ``created_at`` is read from
        the WAL; if the trigger envelope cannot be located the current
        hub clock is used as a fallback.
        """
        if agent_id not in self._passports:
            return []
        results: list[PendingTurn] = []
        for channel_id, metadata in self._active_channels.items():
            if not any(p.agent_id == agent_id for p in metadata.participants):
                continue
            adapter = self._adapters.get((metadata.manifest.type, metadata.manifest.version))
            state = self._adapter_states.get(channel_id)
            if adapter is None or state is None:
                continue
            expected = adapter.expected_next(metadata, state)
            if expected is None or expected.agent_id != agent_id:
                continue
            expected_at = ""
            if expected.triggering_envelope_id:
                wal = await self.read_wal(channel_id)
                for env in wal:
                    if env.envelope_id == expected.triggering_envelope_id:
                        expected_at = env.created_at or ""
                        break
            results.append(
                PendingTurn(
                    channel_id=channel_id,
                    triggering_envelope_id=expected.triggering_envelope_id,
                    expected_at=expected_at or self._clock(),
                )
            )
        return results

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _wal_lock(self, channel_id: str) -> asyncio.Lock:
        lock = self._channel_locks.get(channel_id)
        if lock is None:
            lock = asyncio.Lock()
            self._channel_locks[channel_id] = lock
        return lock

    async def _wal_append(self, envelope: Envelope) -> None:
        await self._store.append(wal_path(envelope.channel_id), envelope.to_json() + "\n")
        # Track the envelope id under its (channel, sender, causation)
        # key so a sender retrying with the same ``causation_id`` can
        # find the prior accepted envelope and skip the side effect.
        if envelope.causation_id and envelope.envelope_id:
            key = (envelope.channel_id, envelope.sender_id, envelope.causation_id)
            self._causation_index[key] = envelope.envelope_id

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
            recipient_passport = self._passports.get(recipient_id)
            if recipient_id == envelope.sender_id:
                # Self-routing always delivers (protocol broadcasts
                # include the sender so the sender's Channel handle
                # sees the lifecycle event).
                pass
            elif sender_passport is not None:
                recipient_rule = self._rules.get(recipient_id)
                if recipient_passport is not None and recipient_rule is not None:
                    decision = await self._arbiter.authorize_dispatch(
                        envelope, sender_passport, recipient_passport, recipient_rule
                    )
                    if isinstance(decision, Deny):
                        continue

            # Federation path: recipients living on another hub are
            # dispatched through a registered ``RemoteAgentProxy``
            # keyed by the recipient's auth scheme. Local endpoints are
            # skipped entirely for these — the remote hub is what holds
            # the binding.
            if recipient_passport is not None and recipient_passport.effective_kind == "remote_agent":
                scheme = recipient_passport.auth.scheme
                proxy = self._remote_proxies.get(scheme)
                if proxy is None:
                    reason = NotFoundError(f"no remote proxy registered for scheme {scheme!r}")
                    await self._fan_out("on_dispatch_failed", envelope, recipient_id, reason)
                    logger.warning(
                        "dispatch failed: channel=%s recipient=%s event=%s reason=%s",
                        envelope.channel_id,
                        recipient_id,
                        envelope.event_type,
                        reason,
                    )
                    continue
                try:
                    await proxy.dispatch(envelope, recipient_passport)
                except Exception as exc:
                    await self._fan_out("on_dispatch_failed", envelope, recipient_id, exc)
                    logger.warning(
                        "remote dispatch failed: channel=%s recipient=%s scheme=%s reason=%s",
                        envelope.channel_id,
                        recipient_id,
                        scheme,
                        exc,
                    )
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
        finally:
            self._cleanup_endpoint(endpoint)

    def _cleanup_endpoint(self, endpoint: LinkEndpoint) -> None:
        """Release an endpoint's bindings when its frame loop ends.

        Called from the ``finally`` of :meth:`_handle_endpoint` so a
        connection drop (client close, transport error) does not leave
        a stale endpoint entry behind. Any agent still mapped to this
        endpoint loses its binding — fresh attaches re-bind on demand.
        """
        endpoint_id = endpoint.endpoint_id
        self._endpoints_by_id.pop(endpoint_id, None)
        bound = self._endpoint_to_agents.pop(endpoint_id, None)
        if not bound:
            return
        for agent_id in bound:
            if self._agent_to_endpoint.get(agent_id) == endpoint_id:
                self._agent_to_endpoint.pop(agent_id, None)

    async def _dispatch_frame(self, endpoint: LinkEndpoint, frame: Frame) -> None:
        if isinstance(frame, RequestFrame):
            await self._handle_request(endpoint, frame)
        elif isinstance(frame, HelloFrame):
            agent_id = self._name_to_id.get(frame.name)
            if agent_id is None:
                await endpoint.send_frame(ErrorFrame(code="not_found", message=f"unknown name: {frame.name}"))
                return
            passport = self._passports.get(agent_id)
            if passport is None:
                # Name index disagrees with passport cache — defensive, shouldn't happen.
                await endpoint.send_frame(ErrorFrame(code="not_found", message=f"no passport for {frame.name}"))
                return
            try:
                adapter = self._auth.get(frame.auth_scheme)
                await adapter.validate(passport, frame.auth_claim)
            except AuthError as exc:
                await endpoint.send_frame(ErrorFrame(code="auth_failed", message=str(exc)))
                return
            try:
                self.bind_endpoint(endpoint.endpoint_id, agent_id)
            except NetworkError as exc:
                await endpoint.send_frame(ErrorFrame(code=_error_code(exc), message=str(exc)))
                return
            await endpoint.send_frame(WelcomeFrame(endpoint_id=endpoint.endpoint_id, hub_time=self._clock()))
            if frame.since_envelope_id is not None:
                await self._replay_for_recipient(agent_id, frame.since_envelope_id)
        elif isinstance(frame, ReceiptFrame):
            await self._handle_receipt(frame)
        elif isinstance(frame, PingFrame):
            await endpoint.send_frame(PongFrame())

    async def _handle_request(self, endpoint: LinkEndpoint, frame: RequestFrame) -> None:
        """Execute a control-plane RPC and reply with a ``ResponseFrame``.

        ``frame.op`` selects the operation; ``frame.params`` is the
        JSON argument dict. Results are serialised to JSON-compatible
        values (``to_dict`` for record types, lists thereof for
        collections). A :class:`NetworkError` maps to
        ``ok=False`` with the matching ``error_code``; any other
        exception surfaces as ``error_code="error"``. The same hub
        methods back both the in-process and the wire path, so
        behaviour is identical regardless of where the caller lives.
        """
        try:
            result = await self._dispatch_request_op(endpoint, frame.op, frame.params)
        except NetworkError as exc:
            await endpoint.send_frame(
                ResponseFrame(
                    request_id=frame.request_id,
                    ok=False,
                    error_code=_error_code(exc),
                    error_message=str(exc),
                )
            )
            return
        except Exception as exc:
            logger.exception("control-plane op failed: op=%s", frame.op)
            await endpoint.send_frame(
                ResponseFrame(
                    request_id=frame.request_id,
                    ok=False,
                    error_code="error",
                    error_message=str(exc),
                )
            )
            return
        await endpoint.send_frame(ResponseFrame(request_id=frame.request_id, ok=True, result=result))

    async def _dispatch_request_op(self, endpoint: LinkEndpoint, op: str, params: dict) -> object:
        """Map one control-plane ``op`` to its hub method + (de)serialisation.

        Long but flat by design: one place to see the entire wire
        control surface. Each branch deserialises ``params`` into the
        hub method's arguments and serialises the return value back to
        a JSON-compatible shape.
        """
        # ── Registration / identity ──────────────────────────────────
        if op == "register":
            passport = Passport.from_dict(params["passport"])
            resume = Resume.from_dict(params["resume"])
            rule = Rule.from_dict(params["rule"]) if params.get("rule") is not None else None
            registered = await self.register_identity(
                passport,
                resume,
                skill_md=params.get("skill_md"),
                rule=rule,
            )
            # Bind the calling connection to the freshly-stamped identity
            # so dispatched notifies route back to this endpoint.
            self.bind_endpoint(endpoint.endpoint_id, registered.agent_id)
            return registered.to_dict()
        if op == "get_agent":
            return (await self.get_agent(params["name_or_id"])).to_dict()
        if op == "get_resume":
            return (await self.get_resume(params["agent_id"])).to_dict()
        if op == "get_skill":
            return await self.get_skill(params["agent_id"])
        if op == "get_rule":
            return (await self.get_rule(params["agent_id"])).to_dict()
        if op == "find_agent_id":
            return self.find_agent_id(params["name"])
        if op == "names_for":
            return {aid: self.name_for(aid) for aid in params["agent_ids"]}
        if op == "list_agents":
            agents = await self.list_agents(
                capability=params.get("capability"),
                query=params.get("query"),
                kind=params.get("kind"),
                sort_by=params.get("sort_by"),
                limit=params.get("limit", 50),
            )
            return [p.to_dict() for p in agents]
        if op == "set_resume":
            await self.set_resume(params["agent_id"], Resume.from_dict(params["resume"]))
            return None
        if op == "set_skill":
            await self.set_skill(params["agent_id"], params.get("skill_md"))
            return None
        if op == "set_rule":
            await self.set_rule(params["agent_id"], Rule.from_dict(params["rule"]))
            return None
        if op == "unregister":
            await self.unregister(params["agent_id"])
            return None

        # ── Channels ─────────────────────────────────────────────────
        if op == "create_channel":
            metadata = await self.create_channel(
                creator_id=params["creator_id"],
                manifest_type=params["manifest_type"],
                manifest_version=params.get("manifest_version", 1),
                participants=params["participants"],
                required_acks=params.get("required_acks"),
                ttl=params.get("ttl"),
                knobs=params.get("knobs"),
                intent=params.get("intent"),
                labels=params.get("labels"),
            )
            return metadata.to_dict()
        if op == "get_channel":
            return (await self.get_channel(params["channel_id"])).to_dict()
        if op == "list_channels":
            channels = await self.list_channels(
                agent_id=params.get("agent_id"),
                limit=params.get("limit", 50),
            )
            return [m.to_dict() for m in channels]
        if op == "close_channel":
            metadata = await self.close_channel(params["channel_id"], reason=params.get("reason", ""))
            return metadata.to_dict()
        if op == "post_envelope":
            return await self.post_envelope(Envelope.from_dict(params["envelope"]))
        if op == "read_wal":
            wal = await self.read_wal(
                params["channel_id"],
                since=params.get("since", 0),
                until=params.get("until"),
            )
            return [e.to_dict() for e in wal]
        if op == "find_envelope_by_causation":
            envelope = await self.find_envelope_by_causation(
                params["channel_id"],
                sender_id=params["sender_id"],
                causation_id=params["causation_id"],
            )
            return envelope.to_dict() if envelope is not None else None
        if op == "can_send":
            return self.can_send(
                params["channel_id"],
                params["sender_id"],
                event_type=params.get("event_type"),
            )
        if op == "pending_turns_for":
            turns = await self.pending_turns_for(params["agent_id"])
            return [dataclasses.asdict(t) for t in turns]
        if op == "report_turn_failure":
            await self.report_turn_failure(
                channel_id=params["channel_id"],
                agent_id=params["agent_id"],
                envelope_id=params["envelope_id"],
                exc=RuntimeError(params.get("error", "")),
            )
            return None

        # ── Tasks (observe-only) ─────────────────────────────────────
        if op == "get_task":
            return (await self.get_task(params["task_id"])).to_dict()
        if op == "list_tasks":
            state = TaskState(params["state"]) if params.get("state") is not None else None
            tasks = await self.list_tasks(
                agent_id=params.get("agent_id"),
                channel_id=params.get("channel_id"),
                state=state,
                limit=params.get("limit", 50),
            )
            return [t.to_dict() for t in tasks]
        if op == "observe_task":
            await self.observe_task(TaskMetadata.from_dict(params["metadata"]))
            return None
        if op == "update_task":
            state = TaskState(params["state"]) if params.get("state") is not None else None
            await self.update_task(
                params["task_id"],
                state=state,
                progress=params.get("progress"),
                result=params.get("result"),
                error=params.get("error"),
            )
            return None
        if op == "record_observation":
            await self.record_observation(
                owner_id=params["owner_id"],
                capability=params["capability"],
                outcome=TaskState(params["outcome"]),
                latency_ms=params.get("latency_ms"),
                task_id=params.get("task_id"),
            )
            return None
        if op == "fire_task_event":
            await self.fire_task_event(params["task_id"], params["kind"], params.get("payload", {}))
            return None
        if op == "checkpoint_task":
            await self.checkpoint_task(params["task_id"], params["state"])
            return None
        if op == "read_task_checkpoint":
            return await self.read_task_checkpoint(params["task_id"])

        raise ProtocolError(f"unknown control-plane op: {op!r}")

    async def _handle_receipt(self, frame: ReceiptFrame) -> None:
        """Process a delivery receipt from a client.

        ``status="ack"`` advances the recipient's cursor for the acked
        envelope's channel (only if the acked envelope_id sorts strictly
        above that channel's current cursor) and persists it.
        ``status="nack"`` leaves the cursor in place — the envelope will
        be replayed when the recipient reconnects — and is logged for
        operational visibility.

        Receipts whose ``recipient_id`` or ``channel_id`` is empty, or
        whose ``recipient_id`` is unknown, are dropped silently: an
        ack the hub cannot attribute to a (recipient, channel) cursor
        has nothing to advance, and a stale receipt for an unregistered
        agent has nothing to act on.
        """
        recipient_id = frame.recipient_id
        if not recipient_id or recipient_id not in self._passports:
            return
        if frame.status == "ack":
            if frame.envelope_id and frame.channel_id:
                await self._advance_cursor(recipient_id, frame.channel_id, frame.envelope_id)
        elif frame.status == "nack":
            logger.warning(
                "envelope nacked: recipient=%s channel=%s envelope=%s reason=%s",
                recipient_id,
                frame.channel_id,
                frame.envelope_id,
                frame.reason or "",
            )

    async def _advance_cursor(self, agent_id: str, channel_id: str, envelope_id: str) -> None:
        """Move ``agent_id``'s cursor for ``channel_id`` forward to
        ``envelope_id`` if it sorts strictly above the current value,
        then persist the agent's full cursor map. Monotonic per channel
        — an ack for an older id is a no-op."""
        channels = self._inbox_cursors.setdefault(agent_id, {})
        if envelope_id > channels.get(channel_id, ""):
            channels[channel_id] = envelope_id
            await self._store.write(
                inbox_cursor_path(agent_id),
                json.dumps(channels, sort_keys=True),
            )

    def inbox_cursor(self, agent_id: str, channel_id: str) -> str:
        """Last envelope_id ``agent_id`` has acked in ``channel_id``.

        Empty string when the agent has acked nothing in that channel
        (or is unknown). Read-only view of the delivery high-water mark
        a reconnect would replay past."""
        return self._inbox_cursors.get(agent_id, {}).get(channel_id, "")

    async def _replay_for_recipient(self, agent_id: str, since_envelope_id: str) -> None:
        """Re-emit unacked envelopes past the recipient's high-water mark.

        Per channel, ``high_water`` is the larger of that channel's
        persisted cursor and the client-supplied ``since_envelope_id``
        — replay starts strictly above it so the client never re-sees
        an envelope it already acked locally, and an ack in one channel
        never suppresses replay in another. Envelopes are replayed in
        per-channel WAL order across every active channel the agent
        participates in. Failures during replay fire
        ``on_dispatch_failed`` and stop the channel's replay so an
        offline recipient does not block reconnect.
        """
        endpoint = self._endpoint_for(agent_id)
        if endpoint is None:
            return
        cursors = self._inbox_cursors.get(agent_id, {})

        for channel_id, metadata in list(self._active_channels.items()):
            if not any(p.agent_id == agent_id for p in metadata.participants):
                continue
            channel_cursor = cursors.get(channel_id, "")
            high_water = channel_cursor if channel_cursor > since_envelope_id else since_envelope_id
            wal = await self.read_wal(channel_id)
            for envelope in wal:
                if not envelope.envelope_id or envelope.envelope_id <= high_water:
                    continue
                if envelope.sender_id == agent_id:
                    continue
                if envelope.audience is not None and agent_id not in envelope.audience:
                    continue
                try:
                    await endpoint.send_frame(NotifyFrame(envelope=envelope, recipient_id=agent_id))
                except Exception as exc:
                    await self._fan_out("on_dispatch_failed", envelope, agent_id, exc)
                    logger.warning(
                        "replay failed: channel=%s recipient=%s envelope=%s reason=%s",
                        channel_id,
                        agent_id,
                        envelope.envelope_id,
                        exc,
                    )
                    break

    async def find_envelope_by_causation(
        self,
        channel_id: str,
        *,
        sender_id: str,
        causation_id: str,
    ) -> Envelope | None:
        """Look up an envelope previously accepted under this causation key.

        Handlers use this to short-circuit duplicate work after an
        at-least-once redelivery: when the same sender re-posts an
        envelope with the same ``causation_id`` (typical on retry),
        the prior accepted envelope is returned so the handler can
        skip the side effect. Returns ``None`` if no envelope is
        recorded for the key — either it was never accepted or its
        channel has already closed (terminal-channel pruning clears
        the index).
        """
        envelope_id = self._causation_index.get((channel_id, sender_id, causation_id))
        if envelope_id is None:
            return None
        wal = await self.read_wal(channel_id)
        for envelope in wal:
            if envelope.envelope_id == envelope_id:
                return envelope
        return None

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
            # Mint inside the WAL lock so the id is assigned in append
            # order — otherwise a concurrent send on this channel could
            # mint a later id but append it first, breaking sort order.
            async with self._wal_lock(channel_id):
                close_envelope.envelope_id = self._mint_envelope_id()
                close_envelope.created_at = self._clock()
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

            # Causation lookups against a closed channel can never produce
            # a meaningful retry decision, so drop every key for this
            # channel and keep the index bounded by active-channel size.
            stale_keys = [k for k in self._causation_index if k[0] == channel_id]
            for k in stale_keys:
                self._causation_index.pop(k, None)
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
            json.dumps(metadata.to_dict()),
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

        cursor_blob = await self._store.read(inbox_cursor_path(agent_id))
        if cursor_blob:
            try:
                parsed = json.loads(cursor_blob)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                cursors = {str(c): str(e) for c, e in parsed.items() if c and e}
                if cursors:
                    self._inbox_cursors[agent_id] = cursors

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
            # Repopulate the causation index for active channels; terminal
            # channels have already had their entries pruned when they
            # closed and shouldn't reappear here.
            if not metadata.is_terminal() and envelope.causation_id and envelope.envelope_id:
                key = (channel_id, envelope.sender_id, envelope.causation_id)
                self._causation_index[key] = envelope.envelope_id
        self._adapter_states[channel_id] = state

    async def _load_task(self, task_id: str) -> None:
        metadata_data = await self._store.read(task_metadata_path(task_id))
        if metadata_data is None:
            return
        metadata = TaskMetadata.from_dict(json.loads(metadata_data))
        self._tasks[task_id] = metadata
        if metadata.channel_id:
            self._channel_tasks.setdefault(metadata.channel_id, set()).add(task_id)
