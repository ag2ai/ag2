# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``HubClient`` — one connection to one hub per process.

A ``HubClient`` holds a single ``LinkClient`` connection and
multiplexes any number of registered ``AgentClient``s through it. It
runs in one of two modes, selected by whether an in-process ``Hub``
reference is available:

* **In-process** — constructed as ``HubClient(LocalLink(hub), hub=hub)``
  (or with a ``LocalLink`` whose ``.hub`` is read automatically).
  Control-plane operations (register, discovery, channel lifecycle,
  posting an envelope, task ops) are direct method calls on the hub.

* **Cross-process** — constructed as ``HubClient(WsLink(url))`` with no
  hub. Control-plane operations travel over the wire as
  ``RequestFrame`` RPCs and the matching ``ResponseFrame`` is awaited,
  correlated by ``request_id``. The same public API works identically
  in both modes; only the transport differs.

In both modes the link carries dispatched envelopes inbound as
``NotifyFrame``s; the receiving side acks them with ``ReceiptFrame``s
so the hub's per-(agent, channel) cursor advances and reconnect replay
stays correct.

Because the cross-process notify handler runs adapter code locally
(view projection, round-envelope construction, turn-ownership probes),
the ``HubClient`` keeps a client-side adapter registry (the built-in
adapters are registered automatically) and folds per-channel adapter
state from the WAL on demand — the same deterministic fold the hub
performs on ``hydrate()``. A small name cache lets ``name_for`` resolve
participant names without a round-trip per render.
"""

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

from ag2.agent import Agent
from ag2.task import TaskMetadata, TaskState

from ..adapters.base import ChannelAdapter
from ..channel import ChannelMetadata, ChannelState
from ..envelope import Envelope
from ..errors import AccessDeniedError, AuthError, NetworkError, NotFoundError, ProtocolError
from ..identity import Passport, Resume
from ..ids import make_id
from ..rule import Rule
from ..transport.frames import (
    ErrorFrame,
    HelloFrame,
    NotifyFrame,
    ReceiptFrame,
    RequestFrame,
    ResponseFrame,
    WelcomeFrame,
)
from ..transport.link import LinkClient, LinkFactory
from ..views.base import ViewPolicy
from .agent_client import AgentClient
from .human_client import HumanClient
from .plugin import NetworkPlugin

if TYPE_CHECKING:
    from ..hub import Hub, PendingTurn

__all__ = ("HubClient",)


logger = logging.getLogger(__name__)


_ERROR_CLASSES: dict[str, type[NetworkError]] = {
    "not_found": NotFoundError,
    "access_denied": AccessDeniedError,
    "protocol_error": ProtocolError,
    "auth_failed": AuthError,
}


def _error_from_code(code: str, message: str) -> NetworkError:
    """Re-raise a wire ``error_code`` as the matching ``NetworkError``.

    Inverse of the hub's ``_error_code`` mapping; unknown codes
    surface as a bare :class:`NetworkError` so the failure still
    propagates with its message intact.
    """
    return _ERROR_CLASSES.get(code, NetworkError)(message)


def _default_adapters() -> list[ChannelAdapter]:
    """Construct the built-in adapters for the client-side registry.

    Stateless instances — every routing decision derives from channel
    metadata + WAL-folded state, so one shared instance per type is
    enough. Mirrors the set the hub registers on ``Hub.open``.

    The adapter modules import client-side tool builders (e.g. the
    ``say`` tool), so importing them at this module's top would close
    an ``adapters`` ⇄ ``client`` import cycle. The import is therefore
    deferred to call time — invoked once per ``HubClient`` construction,
    well after both packages have finished loading.
    """
    from ..adapters.consulting import ConsultingAdapter
    from ..adapters.conversation import ConversationAdapter
    from ..adapters.discussion import DiscussionAdapter
    from ..adapters.workflow import WorkflowAdapter

    return [ConsultingAdapter(), ConversationAdapter(), DiscussionAdapter(), WorkflowAdapter()]


class HubClient:
    """One connection to a hub. Multiple ``AgentClient``s register through it.

    Takes a link factory (``LocalLink`` in-process, ``WsLink`` over
    WebSocket) and an optional in-process ``hub`` reference. When the
    factory is a ``LocalLink`` its ``.hub`` is read automatically, so
    ``HubClient(LocalLink(hub))`` and ``HubClient(LocalLink(hub), hub=hub)``
    are equivalent. With no hub the client runs cross-process and
    routes control-plane calls through ``RequestFrame`` RPC.

    A single tenant process should hold one ``HubClient`` per hub it
    connects to.
    """

    def __init__(
        self,
        link: "LinkFactory",
        *,
        hub: "Hub | None" = None,
        rpc_timeout: float = 30.0,
    ) -> None:
        # __init__ stores params + initialises internal state; the
        # connection (and any wire I/O) is deferred to open()/register().
        self._link = link
        self._hub = hub if hub is not None else getattr(link, "hub", None)
        self._rpc_timeout = rpc_timeout
        self._client_link: LinkClient | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._clients: dict[str, AgentClient] = {}
        self._closed = False
        # Notify handlers spawned off the receive loop (remote mode only —
        # see ``_receive_loop``). Tracked so ``close`` can drain them.
        self._notify_tasks: set[asyncio.Task[None]] = set()

        # Control-plane RPC correlation: request_id -> awaiting future.
        self._pending: dict[str, asyncio.Future[ResponseFrame]] = {}
        # FIFO of futures awaiting a handshake (Welcome / Error) reply.
        self._handshake_waiters: list[asyncio.Future[WelcomeFrame]] = []

        # Client-side adapter registry — needed cross-process so the
        # notify handler can resolve adapters and fold state locally.
        # Harmless in-process (adapter resolution still delegates to the
        # hub there for authoritative behaviour).
        self._adapters: dict[tuple[str, int], ChannelAdapter] = {}
        for adapter in _default_adapters():
            self._adapters[(adapter.manifest.type, adapter.manifest.version)] = adapter

        # Caches populated as records cross the wire. ``_channel_meta``
        # backs local adapter / view resolution; the name maps back
        # ``name_for`` / ``name_to_id_map`` without a round-trip.
        self._channel_meta: dict[str, ChannelMetadata] = {}
        self._name_by_id: dict[str, str] = {}
        self._id_by_name: dict[str, str] = {}

    @property
    def remote(self) -> bool:
        """True when there is no in-process hub (operations go over the wire)."""
        return self._hub is None

    # ── Connection ───────────────────────────────────────────────────────────

    async def open(self) -> "HubClient":
        """Open the underlying link (and start the receive loop). Idempotent.

        For an in-process ``LocalLink`` this attaches the endpoint to
        the hub. For a ``WsLink`` it performs the WebSocket connect.
        Callers may rely on lazy connection (``register`` / ``attach``
        open on first use) or call this explicitly.
        """
        await self._ensure_connected_async()
        return self

    async def _ensure_connected_async(self) -> LinkClient:
        """Open the link on first use; subsequent calls reuse the connection."""
        if self._client_link is None:
            client_link = self._link.client()
            await client_link.open()
            self._client_link = client_link
            self._receive_task = asyncio.create_task(self._receive_loop())
        return self._client_link

    async def _receive_loop(self) -> None:
        """Demultiplex inbound frames.

        ``NotifyFrame`` → deliver to the addressed ``AgentClient`` +
        ack. ``ResponseFrame`` → resolve the awaiting RPC future.
        ``WelcomeFrame`` / ``ErrorFrame`` → resolve the oldest pending
        handshake. ``PongFrame`` is ignored (heartbeat).
        """
        assert self._client_link is not None
        try:
            async for frame in self._client_link.frames():
                if isinstance(frame, NotifyFrame):
                    if self._hub is None:
                        # Remote: the notify handler issues control-plane
                        # RPCs whose ``ResponseFrame``s arrive on THIS loop,
                        # so it must not block on the handler or it would
                        # deadlock against its own responses. Dispatch the
                        # handler as a tracked task and keep reading frames.
                        task = asyncio.create_task(self._dispatch_notify_safe(frame))
                        self._notify_tasks.add(task)
                        task.add_done_callback(self._notify_tasks.discard)
                    else:
                        # In-process: the handler's hub calls are direct
                        # (no round-trip through this loop), so awaiting
                        # inline preserves ordered delivery.
                        await self._dispatch_notify_safe(frame)
                elif isinstance(frame, ResponseFrame):
                    fut = self._pending.pop(frame.request_id, None)
                    if fut is not None and not fut.done():
                        fut.set_result(frame)
                elif isinstance(frame, WelcomeFrame):
                    self._resolve_handshake(welcome=frame, error=None)
                elif isinstance(frame, ErrorFrame):
                    # Errors outside the request/response path are
                    # handshake failures (unknown name / bad auth).
                    self._resolve_handshake(welcome=None, error=frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("receive loop terminated unexpectedly")
        finally:
            self._fail_pending(RuntimeError("connection closed"))

    async def _dispatch_notify_safe(self, frame: NotifyFrame) -> None:
        """Run :meth:`_dispatch_notify`, logging (not raising) on failure.

        Wraps the handler so a spawned notify task (remote mode) never
        surfaces as an unretrieved task exception, and so an inline
        dispatch (in-process) cannot break the receive loop.
        """
        try:
            await self._dispatch_notify(frame)
        except Exception:
            logger.exception(
                "receive loop dispatch failed: channel=%s event=%s recipient=%s",
                frame.envelope.channel_id,
                frame.envelope.event_type,
                frame.recipient_id,
            )

    def _resolve_handshake(self, *, welcome: WelcomeFrame | None, error: ErrorFrame | None) -> None:
        """Resolve the oldest outstanding handshake waiter (FIFO)."""
        while self._handshake_waiters:
            fut = self._handshake_waiters.pop(0)
            if fut.done():
                continue
            if error is not None:
                fut.set_exception(_error_from_code(error.code, error.message))
            else:
                assert welcome is not None
                fut.set_result(welcome)
            return

    def _fail_pending(self, exc: BaseException) -> None:
        """Fail every in-flight RPC / handshake when the link drops."""
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()
        for fut in self._handshake_waiters:
            if not fut.done():
                fut.set_exception(exc)
        self._handshake_waiters.clear()

    async def _dispatch_notify(self, frame: NotifyFrame) -> None:
        """Route the envelope to the recipient stamped on the frame.

        The hub sets ``recipient_id`` per delivery so broadcasts
        (``audience=None``) reach the right ``AgentClient`` without the
        demuxer re-walking channel participants. Frames missing a
        ``recipient_id`` fall back to ``audience``-based routing. After
        the handler returns a ``ReceiptFrame`` flows back so the hub's
        cursor advances; handler exceptions produce a ``nack``.
        """
        if frame.recipient_id:
            client = self._clients.get(frame.recipient_id)
            if client is not None:
                await self._deliver_and_ack(client, frame.envelope, frame.recipient_id)
            return
        if frame.envelope.audience is None:
            return
        for recipient_id in frame.envelope.audience:
            client = self._clients.get(recipient_id)
            if client is not None:
                await self._deliver_and_ack(client, frame.envelope, recipient_id)

    async def _deliver_and_ack(
        self,
        client: AgentClient,
        envelope: Envelope,
        recipient_id: str,
    ) -> None:
        status = "ack"
        reason = ""
        try:
            await client.receive(envelope)
        except Exception as exc:
            status = "nack"
            reason = str(exc)
            logger.exception(
                "notify handler raised: channel=%s event=%s recipient=%s",
                envelope.channel_id,
                envelope.event_type,
                recipient_id,
            )
        if not envelope.envelope_id or self._client_link is None:
            return
        try:
            await self._client_link.send_frame(
                ReceiptFrame(
                    envelope_id=envelope.envelope_id,
                    status=status,
                    recipient_id=recipient_id,
                    channel_id=envelope.channel_id,
                    reason=reason,
                )
            )
        except Exception:
            logger.exception(
                "receipt send failed: channel=%s envelope=%s recipient=%s",
                envelope.channel_id,
                envelope.envelope_id,
                recipient_id,
            )

    # ── Control-plane RPC ──────────────────────────────────────────────────────

    async def _rpc(self, op: str, params: dict[str, Any]) -> Any:
        """Send a ``RequestFrame`` and await the correlated ``ResponseFrame``.

        Raises the matching :class:`NetworkError` subclass when the hub
        replies with ``ok=False``. Only used cross-process; the
        in-process paths call the hub directly.
        """
        link = await self._ensure_connected_async()
        loop = asyncio.get_event_loop()
        request_id = make_id()
        fut: asyncio.Future[ResponseFrame] = loop.create_future()
        self._pending[request_id] = fut
        await link.send_frame(RequestFrame(request_id=request_id, op=op, params=params))
        try:
            resp = await asyncio.wait_for(fut, self._rpc_timeout)
        finally:
            self._pending.pop(request_id, None)
        if not resp.ok:
            raise _error_from_code(resp.error_code, resp.error_message)
        return resp.result

    async def _handshake(self, hello: HelloFrame) -> WelcomeFrame:
        """Send a ``HelloFrame`` and await the ``WelcomeFrame`` (or error).

        The hub validates auth, binds this connection's endpoint to the
        named identity, and — when ``hello.since_envelope_id`` is set —
        replays unacked notifies before returning the welcome.
        """
        link = await self._ensure_connected_async()
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[WelcomeFrame] = loop.create_future()
        self._handshake_waiters.append(fut)
        await link.send_frame(hello)
        return await asyncio.wait_for(fut, self._rpc_timeout)

    # ── Registration ─────────────────────────────────────────────────────────

    async def register(
        self,
        agent: Agent,
        passport: Passport | None = None,
        resume: Resume | None = None,
        *,
        skill_md: str | None = None,
        rule: Rule | None = None,
        attach_plugin: bool = True,
    ) -> AgentClient:
        """Register an agent and return its ``AgentClient`` handle.

        In-process this is a direct hub call; cross-process it is a
        ``register`` RPC. Either way the resulting ``agent_id`` is bound
        to this connection's endpoint so dispatched ``NotifyFrame``s
        reach the right ``AgentClient``.

        ``attach_plugin=True`` (default) attaches the ``NetworkPlugin``
        which adds ``say`` and ``delegate`` to ``agent.tools`` and
        appends ``NetworkContextPolicy`` to the assembly chain. Pass
        ``False`` for tests that need a bare agent without LLM tools.

        Rejects ``passport.kind == "human"`` with a guidance error
        pointing at :meth:`register_human`.
        """
        if self._closed:
            raise RuntimeError("HubClient is closed")

        if passport is None:
            passport = Passport(name=agent.name)
        resume = resume or Resume()

        if passport.kind == "human":
            raise ValueError(
                "register() is for agent-kind participants; "
                "use HubClient.register_human(...) for kind='human' passports"
            )

        client_link = await self._ensure_connected_async()
        effective_rule = rule if rule is not None else Rule()

        if self._hub is not None:
            passport = await self._hub.register_identity(passport, resume, skill_md=skill_md, rule=effective_rule)
            assert passport.agent_id is not None
            self._hub.bind_endpoint(client_link.endpoint_id, passport.agent_id)

        else:
            data = await self._rpc(
                "register",
                {
                    "passport": passport.to_dict(),
                    "resume": resume.to_dict(),
                    "skill_md": skill_md,
                    "rule": effective_rule.to_dict(),
                },
            )
            # The register op binds this endpoint to the new identity hub-side.
            passport = Passport.from_dict(data)

        assert passport.agent_id is not None
        self._cache_passport(passport)

        client = AgentClient(
            agent=agent,
            passport=passport,
            resume=resume,
            rule=effective_rule,
            hub_client=self,
        )
        self._clients[passport.agent_id] = client

        if attach_plugin:
            agent._apply_plugin(NetworkPlugin(client))

        return client

    async def attach(
        self,
        agent: Agent,
        name: str,
        *,
        passport: Passport | None = None,
        resume: Resume | None = None,
        rule: Rule | None = None,
        skill_md: str | None = None,
        attach_plugin: bool = True,
        since_envelope_id: str | None = "",
    ) -> AgentClient:
        """Bind ``agent`` to the hub identity named ``name``.

        Reconnect-aware companion to :meth:`register`. If ``name`` is
        already registered the existing ``agent_id`` is re-bound to this
        connection (cross-process via a ``HelloFrame`` handshake that
        also replays unacked notifies past ``since_envelope_id``;
        in-process via a direct re-bind). If ``name`` is not registered,
        falls back to :meth:`register` — ``passport`` and ``resume``
        become required in that path.

        ``since_envelope_id`` (cross-process only) is the reconnect
        high-water mark: ``""`` (default) replays every notify the hub
        has not seen acked, ``None`` skips replay, a specific id replays
        strictly past it. In-process attach ignores it (the endpoint
        re-binds without replay).
        """
        if self._closed:
            raise RuntimeError("HubClient is closed")

        if self._hub is not None:
            return await self._attach_in_process(
                agent,
                name,
                passport=passport,
                resume=resume,
                rule=rule,
                skill_md=skill_md,
                attach_plugin=attach_plugin,
            )
        return await self._attach_remote(
            agent,
            name,
            passport=passport,
            resume=resume,
            rule=rule,
            skill_md=skill_md,
            attach_plugin=attach_plugin,
            since_envelope_id=since_envelope_id,
        )

    async def _attach_in_process(
        self,
        agent: Agent,
        name: str,
        *,
        passport: Passport | None,
        resume: Resume | None,
        rule: Rule | None,
        skill_md: str | None,
        attach_plugin: bool,
    ) -> AgentClient:
        assert self._hub is not None
        existing_agent_id = self._hub.find_agent_id(name)
        if existing_agent_id is None:
            if passport is None or resume is None:
                raise ValueError(
                    f"attach({name!r}): name is not registered; "
                    "passport and resume are required to fall back to register()"
                )
            return await self.register(
                agent, passport, resume, skill_md=skill_md, rule=rule, attach_plugin=attach_plugin
            )

        client_link = await self._ensure_connected_async()
        existing_passport = await self._hub.get_agent(existing_agent_id)
        existing_resume = await self._hub.get_resume(existing_agent_id)
        existing_rule = await self._hub.get_rule(existing_agent_id)
        self._hub.bind_endpoint(client_link.endpoint_id, existing_agent_id)

        self._cache_passport(existing_passport)
        client = AgentClient(
            agent=agent,
            passport=existing_passport,
            resume=existing_resume,
            rule=existing_rule,
            hub_client=self,
        )
        self._clients[existing_agent_id] = client
        if attach_plugin:
            agent._apply_plugin(NetworkPlugin(client))
        return client

    async def _attach_remote(
        self,
        agent: Agent,
        name: str,
        *,
        passport: Passport | None,
        resume: Resume | None,
        rule: Rule | None,
        skill_md: str | None,
        attach_plugin: bool,
        since_envelope_id: str | None,
    ) -> AgentClient:
        await self._ensure_connected_async()
        existing_agent_id = await self._rpc("find_agent_id", {"name": name})
        if existing_agent_id is None:
            if passport is None or resume is None:
                raise ValueError(
                    f"attach({name!r}): name is not registered; "
                    "passport and resume are required to fall back to register()"
                )
            return await self.register(
                agent, passport, resume, skill_md=skill_md, rule=rule, attach_plugin=attach_plugin
            )

        # Fetch the persisted identity and register the AgentClient
        # BEFORE the handshake so replayed notifies (which the hub emits
        # immediately after Welcome) find their client in the demuxer.
        existing_passport = Passport.from_dict(await self._rpc("get_agent", {"name_or_id": existing_agent_id}))
        existing_resume = Resume.from_dict(await self._rpc("get_resume", {"agent_id": existing_agent_id}))
        existing_rule = Rule.from_dict(await self._rpc("get_rule", {"agent_id": existing_agent_id}))
        self._cache_passport(existing_passport)

        client = AgentClient(
            agent=agent,
            passport=existing_passport,
            resume=existing_resume,
            rule=existing_rule,
            hub_client=self,
        )
        self._clients[existing_agent_id] = client
        if attach_plugin:
            agent._apply_plugin(NetworkPlugin(client))

        # Reconnect handshake — binds this endpoint to the identity and
        # replays unacked notifies past the high-water mark.
        auth = passport.auth if passport is not None else existing_passport.auth
        await self._handshake(
            HelloFrame(
                name=name,
                auth_scheme=auth.scheme,
                auth_claim=dict(auth.claim),
                since_envelope_id=since_envelope_id,
            )
        )
        return client

    async def register_human(
        self,
        passport: Passport,
        *,
        resume: Resume | None = None,
        rule: Rule | None = None,
        auto_ack_invites: bool = True,
    ) -> HumanClient:
        """Register a non-LLM participant and return its ``HumanClient`` handle.

        Same UUID7-stamping + persistence path as ``register`` (direct
        in-process, ``register`` RPC cross-process); the passport's
        ``kind`` is forced to ``"human"`` so the participant is
        discoverable via ``list_agents(kind="human")``.

        No ``Agent`` is attached, no plugin is installed.
        ``auto_ack_invites=True`` (default) makes the human auto-accept
        channel invites so adapter-driven handshakes complete without UI
        round-trips.
        """
        if self._closed:
            raise RuntimeError("HubClient is closed")
        if passport.kind not in (None, "human"):
            raise ValueError(f"register_human() requires kind='human' (or None); got {passport.kind!r}")
        passport.kind = "human"

        client_link = await self._ensure_connected_async()
        effective_rule = rule if rule is not None else Rule()
        effective_resume = resume if resume is not None else Resume()

        if self._hub is not None:
            passport = await self._hub.register_identity(passport, effective_resume, rule=effective_rule)
            assert passport.agent_id is not None
            self._hub.bind_endpoint(client_link.endpoint_id, passport.agent_id)
        else:
            data = await self._rpc(
                "register",
                {
                    "passport": passport.to_dict(),
                    "resume": effective_resume.to_dict(),
                    "skill_md": None,
                    "rule": effective_rule.to_dict(),
                },
            )
            passport = Passport.from_dict(data)

        assert passport.agent_id is not None
        self._cache_passport(passport)

        human = HumanClient(
            passport=passport,
            resume=effective_resume,
            rule=effective_rule,
            hub=self._hub,
            hub_client=self,
            auto_ack_invites=auto_ack_invites,
        )
        # ``_clients`` is identity-keyed; ``HumanClient.receive`` matches
        # the signature the demuxer calls, so dispatch needs no branch.
        self._clients[passport.agent_id] = human  # type: ignore[assignment]
        return human

    # ── Hub passthrough ──────────────────────────────────────────────────────
    #
    # Each operation runs against the in-process hub when one is
    # present, or as a control-plane RPC otherwise. The two branches are
    # behaviourally identical — the hub method backs both.

    # — Discovery —

    async def get_agent(self, name_or_id: str) -> Passport:
        if self._hub is not None:
            passport = await self._hub.get_agent(name_or_id)
        else:
            passport = Passport.from_dict(await self._rpc("get_agent", {"name_or_id": name_or_id}))
        self._cache_passport(passport)
        return passport

    async def get_resume(self, agent_id: str) -> Resume:
        if self._hub is not None:
            return await self._hub.get_resume(agent_id)
        return Resume.from_dict(await self._rpc("get_resume", {"agent_id": agent_id}))

    async def get_skill(self, agent_id: str) -> str | None:
        if self._hub is not None:
            return await self._hub.get_skill(agent_id)
        return await self._rpc("get_skill", {"agent_id": agent_id})

    def find_agent_id(self, name: str) -> str | None:
        """Non-raising name → agent_id lookup.

        In-process this hits the hub registry. Cross-process it reads
        the local name cache (populated as passports cross the wire);
        for an authoritative remote lookup use :meth:`get_agent`, which
        round-trips and raises :class:`NotFoundError` when absent.
        """
        if self._hub is not None:
            return self._hub.find_agent_id(name)
        return self._id_by_name.get(name)

    async def get_rule(self, agent_id: str) -> Rule:
        if self._hub is not None:
            return await self._hub.get_rule(agent_id)
        return Rule.from_dict(await self._rpc("get_rule", {"agent_id": agent_id}))

    async def list_agents(
        self,
        *,
        capability: str | None = None,
        query: str | None = None,
        kind: str | None = None,
        sort_by: str | None = None,
        limit: int = 50,
    ) -> list[Passport]:
        if self._hub is not None:
            passports = await self._hub.list_agents(
                capability=capability, query=query, kind=kind, sort_by=sort_by, limit=limit
            )
        else:
            raw = await self._rpc(
                "list_agents",
                {
                    "capability": capability,
                    "query": query,
                    "kind": kind,
                    "sort_by": sort_by,
                    "limit": limit,
                },
            )
            passports = [Passport.from_dict(d) for d in raw]
        for passport in passports:
            self._cache_passport(passport)
        return passports

    # — Identity mutation —

    async def set_resume(self, agent_id: str, resume: Resume) -> None:
        if self._hub is not None:
            await self._hub.set_resume(agent_id, resume)
        else:
            await self._rpc("set_resume", {"agent_id": agent_id, "resume": resume.to_dict()})

    async def set_skill(self, agent_id: str, skill_md: str | None) -> None:
        if self._hub is not None:
            await self._hub.set_skill(agent_id, skill_md)
        else:
            await self._rpc("set_skill", {"agent_id": agent_id, "skill_md": skill_md})

    async def set_rule(self, agent_id: str, rule: Rule) -> None:
        if self._hub is not None:
            await self._hub.set_rule(agent_id, rule)
        else:
            await self._rpc("set_rule", {"agent_id": agent_id, "rule": rule.to_dict()})

    async def unregister_agent(self, agent_id: str) -> None:
        if self._hub is not None:
            await self._hub.unregister(agent_id)
        else:
            await self._rpc("unregister", {"agent_id": agent_id})

    # — Channel control —

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
        if self._hub is not None:
            metadata = await self._hub.create_channel(
                creator_id=creator_id,
                manifest_type=manifest_type,
                manifest_version=manifest_version,
                participants=participants,
                required_acks=required_acks,
                ttl=ttl,
                knobs=knobs,
                intent=intent,
                labels=labels,
            )
        else:
            data = await self._rpc(
                "create_channel",
                {
                    "creator_id": creator_id,
                    "manifest_type": manifest_type,
                    "manifest_version": manifest_version,
                    "participants": participants,
                    "required_acks": required_acks,
                    "ttl": ttl,
                    "knobs": knobs,
                    "intent": intent,
                    "labels": labels,
                },
            )
            metadata = ChannelMetadata.from_dict(data)
        await self._cache_channel(metadata)
        return metadata

    async def get_channel(self, channel_id: str) -> ChannelMetadata:
        if self._hub is not None:
            return await self._hub.get_channel(channel_id)
        metadata = ChannelMetadata.from_dict(await self._rpc("get_channel", {"channel_id": channel_id}))
        await self._cache_channel(metadata)
        return metadata

    async def list_channels(
        self,
        *,
        agent_id: str | None = None,
        include_terminal: bool = False,
        limit: int = 50,
    ) -> list[ChannelMetadata]:
        if self._hub is not None:
            results = await self._hub.list_channels(agent_id=agent_id, limit=limit * 4)
        else:
            raw = await self._rpc("list_channels", {"agent_id": agent_id, "limit": limit * 4})
            results = [ChannelMetadata.from_dict(d) for d in raw]
        if not include_terminal:
            results = [m for m in results if m.state not in (ChannelState.CLOSED, ChannelState.EXPIRED)]
        return results[:limit]

    async def close_channel(self, channel_id: str, *, reason: str = "") -> ChannelMetadata:
        if self._hub is not None:
            metadata = await self._hub.close_channel(channel_id, reason=reason)
        else:
            metadata = ChannelMetadata.from_dict(
                await self._rpc("close_channel", {"channel_id": channel_id, "reason": reason})
            )
        self._channel_meta[channel_id] = metadata
        return metadata

    async def post_envelope(self, envelope: Envelope) -> str:
        if self._hub is not None:
            return await self._hub.post_envelope(envelope)
        return await self._rpc("post_envelope", {"envelope": envelope.to_dict()})

    async def report_turn_failure(
        self,
        *,
        channel_id: str,
        agent_id: str,
        envelope_id: str,
        exc: BaseException,
    ) -> None:
        """Report a notify-handler crash through the hub's observability surface.

        The default notify handler calls this when the substantive path
        raises; the hub fans the failure out to every ``HubListener``
        (including the built-in ``AuditLog``). Cross-process the
        exception is carried as its string form (the original type is
        not reconstructed on the hub side).
        """
        if self._hub is not None:
            await self._hub.report_turn_failure(
                channel_id=channel_id, agent_id=agent_id, envelope_id=envelope_id, exc=exc
            )
        else:
            await self._rpc(
                "report_turn_failure",
                {"channel_id": channel_id, "agent_id": agent_id, "envelope_id": envelope_id, "error": str(exc)},
            )

    async def fire_task_event(self, task_id: str, kind: str, payload: dict) -> None:
        """Fan out an ``on_task_event`` through the hub's listener chain."""
        if self._hub is not None:
            await self._hub.fire_task_event(task_id, kind, payload)
        else:
            await self._rpc("fire_task_event", {"task_id": task_id, "kind": kind, "payload": payload})

    async def read_wal(self, channel_id: str, *, since: int = 0, until: int | None = None) -> list[Envelope]:
        if self._hub is not None:
            return await self._hub.read_wal(channel_id, since=since, until=until)
        raw = await self._rpc("read_wal", {"channel_id": channel_id, "since": since, "until": until})
        return [Envelope.from_dict(d) for d in raw]

    async def find_envelope_by_causation(
        self,
        channel_id: str,
        *,
        sender_id: str,
        causation_id: str,
    ) -> Envelope | None:
        """Return the envelope a sender previously posted under this
        causation key, or ``None``. The default notify handler uses this
        to skip work when an at-least-once redelivery re-triggers a turn
        it has already answered."""
        if self._hub is not None:
            return await self._hub.find_envelope_by_causation(
                channel_id, sender_id=sender_id, causation_id=causation_id
            )
        data = await self._rpc(
            "find_envelope_by_causation",
            {"channel_id": channel_id, "sender_id": sender_id, "causation_id": causation_id},
        )
        return Envelope.from_dict(data) if data is not None else None

    async def pending_turns_for(self, agent_id: str) -> "list[PendingTurn]":
        """Return turns the protocol currently expects from ``agent_id``.

        Backs :meth:`AgentClient.resume_pending_turns` so the reconnect
        cycle works against an in-process or a remote hub identically.
        """
        from ..hub import PendingTurn

        if self._hub is not None:
            return await self._hub.pending_turns_for(agent_id)
        raw = await self._rpc("pending_turns_for", {"agent_id": agent_id})
        return [PendingTurn(**d) for d in raw]

    async def can_send(
        self,
        channel_id: str,
        sender_id: str,
        *,
        event_type: str | None = None,
    ) -> bool:
        """Whether the adapter would accept a substantive send now.

        Async because cross-process it is an authoritative round-trip to
        the hub (whose folded state is the source of truth); in-process
        it wraps the synchronous hub probe.
        """
        if self._hub is not None:
            return self._hub.can_send(channel_id, sender_id, event_type=event_type)
        return bool(
            await self._rpc(
                "can_send",
                {"channel_id": channel_id, "sender_id": sender_id, "event_type": event_type},
            )
        )

    # — Adapter / view / name resolution (client-side) —

    def register_adapter(self, adapter: ChannelAdapter) -> None:
        """Register a custom ``ChannelAdapter`` in the client-side registry.

        Required cross-process for any non-built-in channel type, so the
        notify handler can resolve the adapter and fold state locally.
        In-process the hub's registry is authoritative; registering here
        too keeps the two in sync if both are consulted.
        """
        self._adapters[(adapter.manifest.type, adapter.manifest.version)] = adapter

    def adapter_for_metadata(self, metadata: ChannelMetadata) -> ChannelAdapter:
        """Resolve the adapter for an already-fetched ``ChannelMetadata``.

        Synchronous — no I/O. In-process it delegates to the hub's
        authoritative registry; cross-process it looks up the
        client-side registry by manifest ``(type, version)``.
        """
        if self._hub is not None:
            return self._hub._adapter_for(metadata.manifest.type, metadata.manifest.version)
        key = (metadata.manifest.type, metadata.manifest.version)
        adapter = self._adapters.get(key)
        if adapter is None:
            raise NotFoundError(f"no adapter registered for {key[0]!r}@v{key[1]}")
        return adapter

    def adapter_for(self, channel_id: str) -> ChannelAdapter:
        """Resolve the adapter for ``channel_id``.

        In-process delegates to the hub. Cross-process resolves from the
        metadata cache (populated by :meth:`get_channel` /
        :meth:`create_channel`); call one of those first if the channel
        has not been seen on this connection.
        """
        if self._hub is not None:
            return self._hub.adapter_for(channel_id)
        metadata = self._channel_meta.get(channel_id)
        if metadata is None:
            raise NotFoundError(f"channel metadata not cached: {channel_id} (fetch via get_channel first)")
        return self.adapter_for_metadata(metadata)

    async def adapter_state(self, channel_id: str) -> object | None:
        """Return ``channel_id``'s folded adapter state, or ``None``.

        In-process this reads the hub's cached state. Cross-process it
        re-folds the state from the channel WAL via the client-side
        adapter — the same deterministic fold the hub runs on
        ``hydrate()`` — so the notify handler sees consistent state
        without shipping the (non-serialisable) state object over the
        wire.
        """
        if self._hub is not None:
            return self._hub.adapter_state(channel_id)
        metadata = self._channel_meta.get(channel_id)
        if metadata is None:
            metadata = await self.get_channel(channel_id)
        adapter = self.adapter_for_metadata(metadata)
        state = adapter.initial_state(metadata)
        for envelope in await self.read_wal(channel_id):
            state = adapter.fold(envelope, state)
        return state

    def default_view_policy(self, channel_id: str, participant_id: str) -> ViewPolicy:
        """Return the adapter-declared default view policy for a participant.

        In-process delegates to the hub; cross-process resolves from the
        cached metadata + client-side adapter.
        """
        if self._hub is not None:
            return self._hub.default_view_policy(channel_id, participant_id)
        metadata = self._channel_meta.get(channel_id)
        if metadata is None:
            raise NotFoundError(f"channel metadata not cached: {channel_id} (fetch via get_channel first)")
        return self.adapter_for_metadata(metadata).default_view_policy(metadata, participant_id)

    def name_for(self, agent_id: str, *, default: str | None = None) -> str:
        """Resolve ``agent_id`` to its registered name.

        In-process reads the hub registry. Cross-process reads the local
        name cache (filled as passports / channel participants cross the
        wire); unknown ids fall back to ``default`` (or the id itself).
        """
        if self._hub is not None:
            return self._hub.name_for(agent_id, default=default)
        name = self._name_by_id.get(agent_id)
        if name is not None:
            return name
        return default if default is not None else agent_id

    def name_to_id_map(self) -> dict[str, str]:
        """Snapshot of the ``name → agent_id`` directory for reverse lookup.

        Used by adapters that resolve target names to ids (e.g. the
        workflow adapter's handoff routing). In-process delegates to the
        hub; cross-process returns the local name cache.
        """
        if self._hub is not None:
            return self._hub.name_to_id_map()
        return dict(self._id_by_name)

    # — Task observation (network is one observer) —

    async def get_task(self, task_id: str) -> TaskMetadata:
        if self._hub is not None:
            return await self._hub.get_task(task_id)
        return TaskMetadata.from_dict(await self._rpc("get_task", {"task_id": task_id}))

    async def list_tasks(
        self,
        *,
        agent_id: str | None = None,
        channel_id: str | None = None,
        state: TaskState | None = None,
        limit: int = 50,
    ) -> list[TaskMetadata]:
        if self._hub is not None:
            return await self._hub.list_tasks(agent_id=agent_id, channel_id=channel_id, state=state, limit=limit)
        raw = await self._rpc(
            "list_tasks",
            {
                "agent_id": agent_id,
                "channel_id": channel_id,
                "state": state.value if state is not None else None,
                "limit": limit,
            },
        )
        return [TaskMetadata.from_dict(d) for d in raw]

    async def observe_task(self, metadata: TaskMetadata) -> None:
        if self._hub is not None:
            await self._hub.observe_task(metadata)
        else:
            await self._rpc("observe_task", {"metadata": metadata.to_dict()})

    async def update_task(
        self,
        task_id: str,
        *,
        state: TaskState | None = None,
        progress: dict[str, object] | None = None,
        result: object | None = None,
        error: str | None = None,
    ) -> None:
        if self._hub is not None:
            await self._hub.update_task(task_id, state=state, progress=progress, result=result, error=error)
        else:
            await self._rpc(
                "update_task",
                {
                    "task_id": task_id,
                    "state": state.value if state is not None else None,
                    "progress": progress,
                    "result": result,
                    "error": error,
                },
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
        if self._hub is not None:
            await self._hub.record_observation(
                owner_id=owner_id,
                capability=capability,
                outcome=outcome,
                latency_ms=latency_ms,
                task_id=task_id,
            )
        else:
            await self._rpc(
                "record_observation",
                {
                    "owner_id": owner_id,
                    "capability": capability,
                    "outcome": outcome.value,
                    "latency_ms": latency_ms,
                    "task_id": task_id,
                },
            )

    async def checkpoint_task(self, task_id: str, state: dict[str, object]) -> None:
        """Persist a task checkpoint through the hub's ``CheckpointStore`` path."""
        if self._hub is not None:
            await self._hub.checkpoint_task(task_id, state)
        else:
            await self._rpc("checkpoint_task", {"task_id": task_id, "state": state})

    async def read_task_checkpoint(self, task_id: str) -> dict[str, object] | None:
        """Read back a task checkpoint, or ``None`` if none is stored."""
        if self._hub is not None:
            return await self._hub.read_task_checkpoint(task_id)
        return await self._rpc("read_task_checkpoint", {"task_id": task_id})

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _cache_passport(self, passport: Passport) -> None:
        if passport.agent_id:
            self._name_by_id[passport.agent_id] = passport.name
            self._id_by_name[passport.name] = passport.agent_id

    async def _cache_channel(self, metadata: ChannelMetadata) -> None:
        self._channel_meta[metadata.channel_id] = metadata
        # Cross-process: resolve participant names up front so the notify
        # handler's view projection renders names, not bare ids.
        if self._hub is None:
            await self._ensure_names([p.agent_id for p in metadata.participants])

    async def _ensure_names(self, agent_ids: list[str]) -> None:
        missing = [aid for aid in agent_ids if aid and aid not in self._name_by_id]
        if not missing:
            return
        mapping = await self._rpc("names_for", {"agent_ids": missing})
        for agent_id, name in mapping.items():
            self._name_by_id[agent_id] = name
            self._id_by_name[name] = agent_id

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the connection and stop the receive loop. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._fail_pending(RuntimeError("HubClient is closed"))
        for task in list(self._notify_tasks):
            task.cancel()
        if self._notify_tasks:
            await asyncio.gather(*self._notify_tasks, return_exceptions=True)
        self._notify_tasks.clear()
        if self._client_link is not None:
            await self._client_link.close()
        if self._receive_task is not None:
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._receive_task

    async def shutdown(self) -> None:
        """Unregister every ``AgentClient`` then ``close()``."""
        for client in list(self._clients.values()):
            with contextlib.suppress(Exception):
                await client.unregister()
        self._clients.clear()
        await self.close()

    async def __aenter__(self) -> "HubClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
