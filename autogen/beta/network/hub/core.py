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
    Envelope,
)
from ..errors import (
    AccessDeniedError,
    DuplicateRegistrationError,
    InviteRejectedError,
    LimitExceededError,
    RuleViolationError,
    SessionClosedError,
    SessionError,
    SessionTypeError,
    UnknownActorError,
    UnknownSessionError,
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
from ..transport.frames import (
    AcceptFrame,
    ChunkFrame,
    ErrorFrame,
    EventFrame,
    Frame,
    HelloFrame,
    NotifyFrame,
    ReceiptFrame,
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
        return hub

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

        return stamped

    async def unregister(self, actor_id: str) -> None:
        async with self._lock:
            identity = self._identities.pop(actor_id, None)
            if identity is None:
                raise UnknownActorError(actor_id)
            self._rules.pop(actor_id, None)
            self._active_sessions.pop(actor_id, None)
            self._name_to_id.pop(identity.name, None)

        await self._store.delete(layout.actor_dir(actor_id))
        await self._store.delete(layout.name_pointer(identity.name))
        endpoint = self._endpoints.pop(actor_id, None)
        if endpoint is not None and not endpoint.closed:
            await endpoint.close()

    async def find(self, capability: str | None = None) -> list[ActorIdentity]:
        async with self._lock:
            identities = list(self._identities.values())
        if capability is None:
            return identities
        return [i for i in identities if capability in i.capabilities]

    async def describe(self, name_or_id: str) -> ActorIdentity:
        actor_id = self._resolve_actor(name_or_id)
        return self._identities[actor_id]

    async def get_rule(self, actor_id: str) -> Rule:
        if actor_id not in self._rules:
            raise UnknownActorError(actor_id)
        return self._rules[actor_id]

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
            raise InviteRejectedError(
                f"only {len(pending.acked_ids)}/{quorum_target} invites acked "
                "before timeout"
            )
        except InviteRejectedError:
            await self._rollback_session(metadata, reason="invite_rejected")
            raise
        finally:
            self._pending_invites.pop(metadata.session_id, None)

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

        # User envelopes go through the adapter.
        prior = await self._read_user_envelopes(metadata.session_id)
        adapter = self._adapter_for(metadata.type)

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

        adapter.validate_send(metadata, envelope, prior)

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
        await self._fanout_to_subs(subs_snapshot, metadata, envelope)

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
            await self._append_to_wal(metadata, envelope)
            subs_snapshot = list(self._subscriptions.values())
        await self._deliver(envelope)
        await self._fanout_to_subs(subs_snapshot, metadata, envelope)

    async def _apply_adapter_result(
        self, metadata: SessionMetadata, result: AdapterResult
    ) -> None:
        changed = False
        if result.next_state is not None and metadata.state is not result.next_state:
            metadata.state = result.next_state
            if result.next_state is SessionState.CLOSED:
                metadata.closed_at = self._clock()
                metadata.close_reason = result.close_reason
                for p in metadata.participants:
                    self._active_sessions[p.actor_id].discard(metadata.session_id)
            changed = True
        if changed:
            await self._write_session_metadata(metadata)
            if metadata.state is SessionState.CLOSED:
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
                await self._append_to_wal(metadata, opened)
                subs_snapshot = list(self._subscriptions.values())
            await self._fanout_to_subs(subs_snapshot, metadata, opened)
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
            await self._append_to_wal(metadata, closed)
            subs_snapshot = list(self._subscriptions.values())
        for participant in metadata.participants:
            closed_for = Envelope.from_dict(closed.to_dict())
            closed_for.recipient_id = participant.actor_id
            await self._deliver(closed_for)
        await self._fanout_to_subs(subs_snapshot, metadata, closed)

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
        await self._broadcast_session_closed(metadata)

    async def read_wal(self, session_id: str, *, since: int = 0) -> list[Envelope]:
        wal = await self._store.read_range(layout.session_wal(session_id), since)
        envelopes: list[Envelope] = []
        for line in wal.split("\n"):
            if not line:
                continue
            envelopes.append(Envelope.from_json(line))
        return envelopes

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
        raw = await self._store.read(layout.session_wal(session_id))
        if not raw:
            return []
        envelopes: list[Envelope] = []
        for line in raw.split("\n"):
            if not line:
                continue
            env = Envelope.from_json(line)
            if env.event_type.startswith("ag2.session.") or env.event_type.startswith(
                "ag2.error"
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
        payload = envelope.to_json() + "\n"
        await self._store.append(layout.actor_inbox_log(recipient_id), payload)
        endpoint = self._endpoints.get(recipient_id)
        if endpoint is not None and not endpoint.closed:
            try:
                await endpoint.send_frame(NotifyFrame(envelope=envelope))
            except Exception:  # pragma: no cover
                log.warning("failed to push notify frame", exc_info=True)

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
    ) -> None:
        """Fan-out an envelope to a pre-captured subscription snapshot.

        The snapshot must come from a ``_wal_lock``-protected section in
        :meth:`post_envelope` so subscriptions registered *after* this
        envelope was appended are not double-delivered — they will see
        this envelope through :meth:`_handle_subscribe`'s replay path.
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
                    EventFrame(subscription_id=sub.subscription_id, envelope=envelope)
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
            if actor_id is not None:
                registered = self._endpoints.get(actor_id)
                if registered is endpoint:
                    self._endpoints.pop(actor_id, None)
            for sid, sub in list(self._subscriptions.items()):
                if sub.endpoint is endpoint:
                    self._subscriptions.pop(sid, None)

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

        await endpoint.send_frame(
            WelcomeFrame(actor_id=actor_id, hub_id=self.config.hub_id)
        )
        return actor_id

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
        if idempotency_key is not None:
            self._record_idempotent(
                envelope.session_id, idempotency_key, envelope_id, wal_offset
            )
        await endpoint.send_frame(
            AcceptFrame(envelope_id=envelope_id, wal_offset=wal_offset)
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
                await endpoint.send_frame(
                    ErrorFrame(
                        code="access_denied",
                        message="not permitted to subscribe to this session",
                        request_id=frame.subscription_id,
                    )
                )
                return

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
                prior = await self.read_wal(frame.session_id, since=sub.since)
                self._subscriptions[sub.subscription_id] = sub
        else:
            async with self._wal_lock:
                self._subscriptions[sub.subscription_id] = sub
            prior = []

        for envelope in prior:
            if (
                sub.causation_id is not None
                and envelope.causation_id != sub.causation_id
            ):
                continue
            await endpoint.send_frame(
                EventFrame(subscription_id=sub.subscription_id, envelope=envelope)
            )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def hub_id(self) -> str:
        return self.config.hub_id
