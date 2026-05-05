# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``Hub`` — registry, dispatcher, persistence root.

M1 surface only — no sweepers, no expectations, no audit log, no
session adapter machinery. Sessions are M2; observability is M3.

The hub is the only place that has cross-tenant visibility:

* Validates senders are registered.
* Stamps ``envelope_id`` and ``created_at`` on accept.
* Persists envelopes to ``sessions/{session_id}/wal.jsonl``.
* Dispatches notifies to explicit ``audience`` via ``Link.send_frame``.

Broadcast (``audience=None``) is **deferred to M2** because M1 does not
track session participants. Tests must address envelopes explicitly.

The hub never calls ``Agent.ask``, executes tenant transforms, or
imports tenant modules — the trust boundary runs through ``HubClient``
/ ``AgentClient``.
"""

import asyncio
import fnmatch
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from autogen.beta.knowledge import KnowledgeStore

from ..auth import AuthRegistry, default_registry
from ..envelope import Envelope
from ..errors import AccessDeniedError, NetworkError, NotFoundError
from ..identity import Passport, Resume
from ..ids import make_id
from ..rule import Rule
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
from .layout import (
    agents_root,
    passport_path,
    resume_path,
    rule_path,
    skill_path,
    wal_path,
)

__all__ = ("Hub",)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_ERROR_CODE_MAP: dict[type, str] = {
    NotFoundError: "not_found",
    AccessDeniedError: "access_denied",
}


def _error_code(exc: BaseException) -> str:
    for cls, code in _ERROR_CODE_MAP.items():
        if isinstance(exc, cls):
            return code
    return "error"


def _match_any(name: str, patterns: list[str]) -> bool:
    """True if ``name`` matches any of the glob patterns (``["*"]`` allows all)."""
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


class Hub:
    """In-process registry, dispatcher, and persistence root.

    M1 construction takes a ``KnowledgeStore`` and an optional
    ``AuthRegistry``, ``clock``, and a ``Callable`` to register an
    "on connect" hook with the transport — see :meth:`attach_endpoint`.

    Use :meth:`open` for production (async constructor that hydrates
    from disk before returning); the sync ``__init__`` is for tests
    that bring up a hub with no prior state.
    """

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        auth: AuthRegistry | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        # __init__ stores params; side effects deferred to start()/hydrate().
        self._store = store
        self._auth = auth if auth is not None else default_registry
        self._clock = clock if clock is not None else _utc_now_iso

        # In-memory caches — rebuilt by hydrate() from disk.
        self._passports: dict[str, Passport] = {}
        self._resumes: dict[str, Resume] = {}
        self._rules: dict[str, Rule] = {}
        self._skills: dict[str, str] = {}
        self._name_to_id: dict[str, str] = {}

        # Transport-side state.
        self._endpoints_by_id: dict[str, LinkEndpoint] = {}
        self._agent_to_endpoint: dict[str, str] = {}
        self._endpoint_to_agents: dict[str, set[str]] = {}
        self._endpoint_tasks: set[asyncio.Task[None]] = set()

        # Per-session locks for WAL append + dispatch ordering.
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._registration_lock = asyncio.Lock()

        self._closed = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @classmethod
    async def open(
        cls,
        store: KnowledgeStore,
        *,
        auth: AuthRegistry | None = None,
        clock: Callable[[], str] | None = None,
    ) -> "Hub":
        """Construct + hydrate from disk. Production entry point.

        M1 has no sweepers to start; M3 will spawn them here.
        """
        hub = cls(store, auth=auth, clock=clock)
        await hub.hydrate()
        return hub

    async def hydrate(self) -> None:
        """Walk the store; rebuild identity caches. Idempotent.

        M1 hydrates ``passport`` / ``resume`` / ``rule`` / ``SKILL.md``
        for every registered agent. Session WAL re-folding lands in M2
        when adapters ship.
        """
        self._passports.clear()
        self._resumes.clear()
        self._rules.clear()
        self._skills.clear()
        self._name_to_id.clear()

        children = await self._store.list(agents_root())
        for child in children:
            if not child.endswith("/"):
                continue
            agent_id = child.rstrip("/")
            await self._load_agent(agent_id)

    async def close(self) -> None:
        """Cancel endpoint handler tasks and drain pending frames."""
        if self._closed:
            return
        self._closed = True
        for task in list(self._endpoint_tasks):
            task.cancel()
        if self._endpoint_tasks:
            await asyncio.gather(*self._endpoint_tasks, return_exceptions=True)
        self._endpoint_tasks.clear()

    async def __aenter__(self) -> "Hub":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ── Registration ────────────────────────────────────────────────────────

    async def register(
        self,
        passport: Passport,
        resume: Resume,
        *,
        skill_md: str | None = None,
        rule: Rule | None = None,
    ) -> Passport:
        """Stamp ``agent_id`` (UUID-based), persist records, return passport.

        Validates ``passport.auth`` against the registered ``AuthRegistry``
        before stamping the id. Auth failure raises ``AuthError`` and
        nothing is persisted.
        """
        adapter = self._auth.get(passport.auth.scheme)
        await adapter.validate(passport, passport.auth.claim)

        async with self._registration_lock:
            if passport.name in self._name_to_id:
                # Re-registration with the same name yields a fresh id and
                # supersedes the prior registration. The prior agent_id's
                # files remain on disk for audit; cache pointers move.
                pass

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

        return passport

    async def unregister(self, agent_id: str) -> None:
        """Remove registry entries; closed sessions remain on disk.

        M1 does not touch the agent's session WAL or task records on
        unregister. Inbox cursor / nack / overflow files are deleted in
        Phase 3 when the inbox model lands.
        """
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
        # Lazy load from disk so re-registered agents pick up their SKILL.md
        # without a full hydrate cycle.
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
        sort_by: str | None = None,
        limit: int = 50,
    ) -> list[Passport]:
        """Filter + rank registered agents.

        M1 supports ``capability`` (matches ``claimed_capabilities ∪
        observed.keys()``) and ``query`` (substring on ``Resume.summary``,
        case-insensitive). ``sort_by`` honours ``"name"`` (lex);
        ``"cost"`` and ``"track_record"`` arrive in M3 once
        ``CostProfile`` and ``ObservedStat`` are populated by real
        traffic.
        """
        results: list[Passport] = []
        query_lower = query.lower() if query else None
        for agent_id, passport in self._passports.items():
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
        # "cost" and "track_record" sorts ship in M3 when the data
        # they require is being populated by real traffic.

        return results[:limit]

    # ── Mutation ─────────────────────────────────────────────────────────────

    async def set_resume(self, agent_id: str, resume: Resume) -> None:
        if agent_id not in self._passports:
            raise NotFoundError(f"agent not registered: {agent_id}")
        resume.last_updated = self._clock()
        resume.version = (self._resumes[agent_id].version + 1) if agent_id in self._resumes else resume.version
        await self._persist_resume(agent_id, resume)
        self._resumes[agent_id] = resume

    async def set_skill(self, agent_id: str, skill_md: str | None) -> None:
        if agent_id not in self._passports:
            raise NotFoundError(f"agent not registered: {agent_id}")
        if skill_md is None:
            await self._store.delete(skill_path(agent_id))
            self._skills.pop(agent_id, None)
        else:
            await self._persist_skill(agent_id, skill_md)
            self._skills[agent_id] = skill_md

    async def set_rule(self, agent_id: str, rule: Rule) -> None:
        if agent_id not in self._passports:
            raise NotFoundError(f"agent not registered: {agent_id}")
        rule.version = (self._rules[agent_id].version + 1) if agent_id in self._rules else rule.version
        await self._persist_rule(agent_id, rule)
        self._rules[agent_id] = rule

    # ── Envelope dispatch (M1 simplified — no session machinery) ─────────────

    async def post_envelope(self, envelope: Envelope) -> str:
        """Validate sender + stamp + WAL append + dispatch.

        M1 simplifications:

        * No session adapter validation — the envelope is accepted as-is
          (M2 wires ``adapter.validate_send`` via the per-session
          ``AdapterState`` cache).
        * No broadcast (``audience=None``) — M1 has no session
          participant tracking, so broadcast envelopes are persisted to
          the WAL but not dispatched. M2 fills this in.
        * Access enforcement is sender-side ``access.outbound_to`` only;
          recipient-side ``access.inbound_from`` is checked at dispatch.
        """
        sender = self._passports.get(envelope.sender_id)
        if sender is None:
            raise NotFoundError(f"sender not registered: {envelope.sender_id}")

        sender_rule = self._rules.get(envelope.sender_id, Rule())

        # Sender-side outbound access check.
        if envelope.audience is not None:
            for recipient_id in envelope.audience:
                recipient = self._passports.get(recipient_id)
                if recipient is None:
                    continue  # silently skip unknown recipients (offline)
                if not _match_any(recipient.name, sender_rule.access.outbound_to):
                    raise AccessDeniedError(
                        f"sender {sender.name!r} not permitted to send to {recipient.name!r}"
                    )

        envelope.envelope_id = make_id()
        envelope.created_at = self._clock()

        async with self._wal_lock(envelope.session_id):
            await self._store.append(
                wal_path(envelope.session_id),
                envelope.to_json() + "\n",
            )

        await self._dispatch(envelope)
        return envelope.envelope_id

    async def read_wal(self, session_id: str) -> list[Envelope]:
        """Read all envelopes for a session. M1 helper for tests."""
        body = await self._store.read(wal_path(session_id))
        if not body:
            return []
        envelopes: list[Envelope] = []
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            envelopes.append(Envelope.from_json(line))
        return envelopes

    # ── Endpoint management (transport-facing) ───────────────────────────────

    def attach_endpoint(self, endpoint: LinkEndpoint) -> None:
        """Register a new connection; spawn its frame-processor task.

        Called by ``LocalLink.set_on_connect`` for every new client.
        Endpoint ``agent_id`` is bound separately via
        :meth:`bind_endpoint` once the identity is known.
        """
        if self._closed:
            return
        self._endpoints_by_id[endpoint.endpoint_id] = endpoint
        task = asyncio.create_task(self._handle_endpoint(endpoint))
        self._endpoint_tasks.add(task)
        task.add_done_callback(self._endpoint_tasks.discard)

    def bind_endpoint(self, endpoint_id: str, agent_id: str) -> None:
        """Associate an existing endpoint with a registered identity.

        Multiple agent_ids may share one endpoint (one ``HubClient``
        connection hosts every ``AgentClient`` registered through it).
        """
        if endpoint_id not in self._endpoints_by_id:
            raise NotFoundError(f"endpoint not attached: {endpoint_id}")
        if agent_id not in self._passports:
            raise NotFoundError(f"agent not registered: {agent_id}")
        self._endpoints_by_id[endpoint_id].agent_id = agent_id
        self._agent_to_endpoint[agent_id] = endpoint_id
        self._endpoint_to_agents.setdefault(endpoint_id, set()).add(agent_id)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _wal_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    async def _dispatch(self, envelope: Envelope) -> None:
        """Send NotifyFrames to the explicit audience, honouring inbound access.

        Broadcast (``audience=None``) is M2 — M1 returns silently for
        broadcasts, having already persisted to the WAL.
        """
        if envelope.audience is None:
            return
        sender_passport = self._passports.get(envelope.sender_id)
        sender_name = sender_passport.name if sender_passport is not None else envelope.sender_id
        for recipient_id in envelope.audience:
            recipient_rule = self._rules.get(recipient_id)
            if recipient_rule is not None and not _match_any(
                sender_name, recipient_rule.access.inbound_from
            ):
                continue  # silently drop blocked inbound; M3 may emit ag2.error
            endpoint = self._endpoint_for(recipient_id)
            if endpoint is None:
                continue  # recipient not connected — M2's inbox model will queue
            await endpoint.send_frame(NotifyFrame(envelope=envelope))

    def _endpoint_for(self, agent_id: str) -> LinkEndpoint | None:
        endpoint_id = self._agent_to_endpoint.get(agent_id)
        if endpoint_id is None:
            return None
        return self._endpoints_by_id.get(endpoint_id)

    async def _handle_endpoint(self, endpoint: LinkEndpoint) -> None:
        """Long-running frame loop for one connection."""
        try:
            async for frame in endpoint.frames():
                await self._dispatch_frame(endpoint, frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Swallow — endpoint loops must not propagate to the gather.
            # Production builds should log at this point; M1 keeps quiet.
            pass

    async def _dispatch_frame(self, endpoint: LinkEndpoint, frame: Frame) -> None:
        """Per-frame dispatch. M1 supports hello/ping/send only."""
        if isinstance(frame, SendFrame):
            try:
                envelope_id = await self.post_envelope(frame.envelope)
                await endpoint.send_frame(AcceptFrame(envelope_id=envelope_id))
            except NetworkError as exc:
                await endpoint.send_frame(
                    ErrorFrame(
                        code=_error_code(exc),
                        message=str(exc),
                    )
                )
        elif isinstance(frame, HelloFrame):
            agent_id = self._name_to_id.get(frame.name)
            if agent_id is None:
                await endpoint.send_frame(
                    ErrorFrame(code="not_found", message=f"unknown name: {frame.name}")
                )
                return
            try:
                self.bind_endpoint(endpoint.endpoint_id, agent_id)
            except NetworkError as exc:
                await endpoint.send_frame(
                    ErrorFrame(code=_error_code(exc), message=str(exc))
                )
                return
            await endpoint.send_frame(
                WelcomeFrame(endpoint_id=endpoint.endpoint_id, hub_time=self._clock())
            )
        elif isinstance(frame, PingFrame):
            await endpoint.send_frame(PongFrame())
        # ReceiptFrame (audit) ships in M3.
        # SubscribeFrame / UnsubscribeFrame ship in M3.

    # ── Persistence helpers ──────────────────────────────────────────────────

    async def _persist_passport(self, passport: Passport) -> None:
        assert passport.agent_id is not None
        await self._store.write(
            passport_path(passport.agent_id),
            json.dumps(passport.to_dict()),
        )

    async def _persist_resume(self, agent_id: str, resume: Resume) -> None:
        await self._store.write(
            resume_path(agent_id),
            json.dumps(resume.to_dict()),
        )

    async def _persist_rule(self, agent_id: str, rule: Rule) -> None:
        await self._store.write(
            rule_path(agent_id),
            json.dumps(rule.to_dict()),
        )

    async def _persist_skill(self, agent_id: str, skill_md: str) -> None:
        await self._store.write(skill_path(agent_id), skill_md)

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

        # SKILL.md loaded on demand via get_skill.
