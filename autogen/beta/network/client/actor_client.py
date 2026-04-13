# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""ActorClient — per-identity inbound handle.

Responsibilities:

* Own the :class:`Link` client connection on behalf of one identity.
* Run the inbox frame loop (notify / accept / event / error).
* Dispatch notify frames to the handler registry (system events or
  per-session-type defaults). Notify handlers run in **background tasks**
  so the inbox loop stays free to deliver the handlers' own outbound
  accepts and subscription events — otherwise ``session.ask`` and
  ``invite_ack`` would deadlock against themselves.
* Expose a friendly ``open(session_type, target, ...)`` API that returns a
  :class:`Session` handle.
* Run the (empty) Phase 1 transforms pipeline — the seam Phase 5 fills in.

The hub never invokes ``Actor.ask`` — the ActorClient does it, inside the
actor's own address space, so Phase 5 transforms run where they belong.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..envelope import (
    EV_SESSION_CLOSED,
    EV_SESSION_INVITE,
    EV_SESSION_INVITE_ACK,
    EV_SESSION_OPENED,
    Envelope,
)
from ..errors import (
    AccessDeniedError,
    InboxFullError,
    LimitExceededError,
    LinkClosedError,
    NetworkError,
    RuleViolationError,
    SessionClosedError,
    SessionError,
    TransportError,
    UnknownActorError,
)
from ..hub import Hub
from ..identity import ActorIdentity
from ..ids import new_id
from ..rule import Rule
from ..session_types import SessionMetadata, SessionType
from ..transport.frames import (
    AcceptFrame,
    ChunkFrame,
    ErrorFrame,
    EventFrame,
    HelloFrame,
    NotifyFrame,
    ReceiptFrame,
    SendFrame,
    SubscribeFrame,
    UnsubscribeFrame,
    WelcomeFrame,
)
from ..transport.link import Link
from . import handlers as default_handlers
from .session import Session

if TYPE_CHECKING:
    from .hub_client import HubClient


log = logging.getLogger("autogen.beta.network.client.actor_client")


NotifyHandler = Callable[[Envelope, "ActorClient"], Awaitable[None]]


@dataclass
class _ClientSubscription:
    """Per-subscription state on the ``ActorClient`` side.

    Phase 2 stored only an ``asyncio.Queue[Envelope]`` keyed by
    subscription id. Phase 3a wraps the queue together with enough
    replay metadata (session id, causation filter, high-water WAL
    offset) that a reconnecting client can re-subscribe with the
    correct ``since`` cursor and not lose envelopes or deliver
    duplicates across a ``WsLink`` drop.

    ``subscription_id`` is the id under which the hub currently
    knows this subscription. On reconnect :meth:`ActorClient.reconnect`
    rotates every subscription to a fresh id (the old endpoint was
    torn down on the hub side when the connection dropped) and
    updates this field in place — callers holding a reference to the
    queue (e.g. ``Session.ask``) do not need to re-open anything.
    """

    subscription_id: str
    queue: asyncio.Queue[Envelope]
    session_id: str | None
    causation_id: str | None
    since: int = 0


class ActorClient:
    """Per-identity client — inbox, handlers, session handles."""

    def __init__(
        self,
        *,
        actor: Any,
        identity: ActorIdentity,
        rule: Rule,
        hub: Hub,
        link: Link,
        hub_client: HubClient | None = None,
    ) -> None:
        self._actor = actor
        self._identity = identity
        self._rule = rule
        self._hub = hub
        self._link = link
        self._hub_client = hub_client

        self._link_client: Any = None
        self._loop_task: asyncio.Task[None] | None = None
        self._welcome = asyncio.Event()
        self._stopped = False
        self._handler_tasks: set[asyncio.Task[None]] = set()

        # Bounded ring of recent outbound envelope ids — used to suppress
        # the default notify handler for envelopes that are replies to our
        # own sends. The user is driving those via ``session.ask`` /
        # ``session.subscribe`` and does not want an auto-echo here.
        self._recent_sends: deque[str] = deque(maxlen=256)
        self._recent_sends_set: set[str] = set()

        # Accept / error correlation for outbound sends.
        self._send_lock = asyncio.Lock()
        self._pending_accept: asyncio.Future[AcceptFrame | ErrorFrame] | None = None

        # Subscription state keyed by subscription_id.
        # Phase 3a upgrade: stores cursor + filter metadata so
        # :meth:`reconnect` can replay each subscription with the
        # right ``since`` after a transport drop.
        self._subs: dict[str, _ClientSubscription] = {}
        # Pending subscription-accept correlation. The hub's
        # _handle_subscribe emits AcceptFrame(envelope_id="",
        # request_id=<sub_id>) once the replay has finished; the
        # client uses this to know when a subscribe is live and the
        # initial replay has drained. Keyed by subscription_id.
        self._pending_sub_accepts: dict[
            str, asyncio.Future[AcceptFrame | ErrorFrame]
        ] = {}

        # Chunk queues keyed by envelope_id — the in-flight response
        # the chunks belong to. Populated as ``chunk`` frames arrive;
        # drained by ``Session.iter_chunks``.
        self._chunks: dict[str, asyncio.Queue[ChunkFrame]] = {}

        # Handler registries. Keys are session-type names (strings) so
        # operator-registered custom types can plug a handler here with
        # the same ``client.on("tournament")`` surface. We key off the
        # enum member's ``.value`` because ``str(SessionType.X)`` in
        # Python 3.11+ returns the qualified name (``"SessionType.X"``)
        # instead of the underlying string value.
        self._type_handlers: dict[str, NotifyHandler] = {
            SessionType.CONSULTING.value: default_handlers.handle_consulting,
            SessionType.CONVERSATION.value: default_handlers.handle_conversation,
            SessionType.NOTIFICATION.value: default_handlers.handle_notification,
        }
        self._system_handlers: dict[str, NotifyHandler] = {
            EV_SESSION_INVITE: _handle_invite,
            EV_SESSION_INVITE_ACK: _noop,
            EV_SESSION_OPENED: _noop,
            EV_SESSION_CLOSED: _noop,
        }

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def actor(self) -> Any:
        return self._actor

    @property
    def actor_id(self) -> str:
        return self._identity.actor_id or ""

    @property
    def identity(self) -> ActorIdentity:
        return self._identity

    @property
    def rule(self) -> Rule:
        return self._rule

    def lookup_session(self, session_id: str) -> SessionMetadata | None:
        """Look up session metadata the hub currently has in cache."""

        return self._hub.peek_session(session_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _start(self) -> None:
        self._link_client = self._link.client()
        await self._link_client.send_frame(
            HelloFrame(
                identity=self._identity.to_dict(),
                rule=self._rule.to_dict(),
                auth_claim={},
                resume_actor_id=self.actor_id,
            )
        )
        self._loop_task = asyncio.get_event_loop().create_task(self._run_loop())
        await self._welcome.wait()

    async def _run_loop(self) -> None:
        try:
            async for frame in self._link_client.frames():
                await self._dispatch_frame(frame)
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception:  # pragma: no cover
            log.exception("actor client frame loop error")

    async def disconnect(self) -> None:
        """Close the link and stop the inbox loop — identity stays registered.

        This is the "graceful shutdown" path: the actor is temporarily going
        offline but its identity, rule, and WAL state remain on the hub.

        Handler tasks that are still running when disconnect is called are
        **cancelled first, then awaited** — a blocking notify handler (e.g.
        one waiting on a model call or a real WebSocket read) would
        otherwise hang ``disconnect`` indefinitely. Phase 2 left the cancel
        out because ``LocalLink`` handlers always completed promptly; Phase
        3a's ``WsLink`` breaks that assumption and the fix was flagged on
        the Phase 2 lingering-items list.
        """

        if self._stopped:
            return
        self._stopped = True

        pending = list(self._handler_tasks)
        for task in pending:
            if not task.done():
                task.cancel()
        for task in pending:
            try:  # noqa: SIM105
                await task
            except (asyncio.CancelledError, Exception):  # pragma: no cover
                pass

        if self._link_client is not None:
            try:  # noqa: SIM105
                await self._link_client.close()
            except Exception:  # pragma: no cover
                pass
        if self._loop_task is not None:
            try:  # noqa: SIM105
                await self._loop_task
            except Exception:  # pragma: no cover
                pass

    async def reconnect(self) -> None:
        """Tear down the current transport and replay subscriptions on a new one.

        Phase 3a Step 3: the foundation for ``WsLink`` reconnect. The
        client is driven by an underlying :class:`Link` whose
        ``client()`` factory hands out fresh peer connections. On a
        transport drop (``LocalLink.close`` in tests, a real WS
        socket error in production), the hub tears down every
        subscription bound to the old endpoint. A reconnecting
        :class:`ActorClient` must therefore:

        1. Close the stale ``_link_client`` and cancel the frame loop.
        2. Mint a fresh client via ``self._link.client()``.
        3. Re-send ``HelloFrame`` and wait for ``WelcomeFrame``.
        4. Walk every live ``_ClientSubscription``, allocate a new
           ``subscription_id``, re-register it in ``_subs`` under the
           new id, and emit ``SubscribeFrame`` with the saved
           ``since`` cursor so the hub replays only envelopes that
           landed after the drop.

        Callers holding queue references (``Session.ask``,
        ``Session.subscribe``) are **not** disturbed — the queue
        object stays the same across the rotation, only its stashed
        subscription id changes.

        This method is idempotent-ish: calling it on a fully-closed
        client will raise ``LinkClosedError`` from the fresh
        ``self._link.client()`` call, which is the right signal that
        the underlying transport is gone.
        """

        if self._stopped:
            raise LinkClosedError("ActorClient is stopped; cannot reconnect")

        # Step 1: stop the old frame loop and close the stale link
        # client. We do NOT cancel handler tasks — a reconnect is a
        # transport-level event, not a shutdown, so any in-flight
        # handler can keep running until it naturally completes.
        old_client = self._link_client
        old_loop_task = self._loop_task
        if old_client is not None:
            with contextlib.suppress(Exception):
                await old_client.close()
        if old_loop_task is not None:
            with contextlib.suppress(Exception):
                await old_loop_task

        # Step 2-3: fresh connection + hello handshake. Reset the
        # welcome event so ``_start`` can wait on it cleanly.
        self._welcome = asyncio.Event()
        self._link_client = self._link.client()
        await self._link_client.send_frame(
            HelloFrame(
                identity=self._identity.to_dict(),
                rule=self._rule.to_dict(),
                auth_claim={},
                resume_actor_id=self.actor_id,
            )
        )
        self._loop_task = asyncio.get_event_loop().create_task(self._run_loop())
        await self._welcome.wait()

        # Step 4: replay every live subscription with a fresh id +
        # saved cursor. We must re-key ``_subs`` under the new ids
        # so the incoming EventFrames land in the right sub entry.
        # The queue objects are reused — callers holding a reference
        # to the queue (Session.ask, Session.subscribe) do not need
        # to be notified of the rotation.
        #
        # Every rotated subscribe is awaited to completion before
        # this method returns. That closes the ordering gap: if
        # ``reconnect`` returns while a subscribe is still in flight,
        # a subsequent ``session.send`` could race the subscribe and
        # fan-out past the new sub, losing envelopes.
        stale = list(self._subs.values())
        self._subs.clear()
        self._pending_sub_accepts.clear()
        rotated: list[
            tuple[_ClientSubscription, asyncio.Future[AcceptFrame | ErrorFrame]]
        ] = []
        for sub in stale:
            new_sub_id = new_id()
            sub.subscription_id = new_sub_id
            sub.queue.__dict__["subscription_id"] = new_sub_id
            self._subs[new_sub_id] = sub
            fut: asyncio.Future[AcceptFrame | ErrorFrame] = (
                asyncio.get_event_loop().create_future()
            )
            self._pending_sub_accepts[new_sub_id] = fut
            rotated.append((sub, fut))
            await self._link_client.send_frame(
                SubscribeFrame(
                    subscription_id=new_sub_id,
                    session_id=sub.session_id,
                    causation_id=sub.causation_id,
                    since=sub.since,
                )
            )

        for sub, fut in rotated:
            result = await fut
            if isinstance(result, ErrorFrame):
                # The hub refused the replay — drop the sub so the
                # client doesn't keep stale state. Subsequent sends
                # will not reach this subscription; callers that
                # still hold a reference to the queue will stall.
                self._subs.pop(sub.subscription_id, None)
                log.warning(
                    "reconnect: hub rejected subscription replay: %s",
                    result.message,
                )
                continue
            if result.wal_offset > sub.since:
                sub.since = result.wal_offset

    async def unregister(self) -> None:
        """Disconnect AND remove the registration from the hub for good."""

        await self.disconnect()
        try:
            await self._hub.unregister(self.actor_id)
        except UnknownActorError:
            pass
        except Exception:  # pragma: no cover
            log.exception("hub.unregister failed")

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def on(
        self, session_type: SessionType | str
    ) -> Callable[[NotifyHandler], NotifyHandler]:
        key = (
            session_type.value
            if isinstance(session_type, SessionType)
            else str(session_type)
        )

        def register(handler: NotifyHandler) -> NotifyHandler:
            self._type_handlers[key] = handler
            return handler

        return register

    def on_system_event(
        self, event_type: str
    ) -> Callable[[NotifyHandler], NotifyHandler]:
        def register(handler: NotifyHandler) -> NotifyHandler:
            self._system_handlers[event_type] = handler
            return handler

        return register

    # ------------------------------------------------------------------
    # Session API
    # ------------------------------------------------------------------

    async def open(
        self,
        session_type: SessionType | str,
        *,
        target: str | list[str],
        labels: dict[str, Any] | None = None,
    ) -> Session:
        participant_names = [target] if isinstance(target, str) else list(target)
        metadata = await self._hub.create_session(
            creator_id=self.actor_id,
            session_type=session_type,
            participant_names=participant_names,
            labels=labels,
        )
        return Session(client=self, metadata=metadata)

    async def close_session(self, session_id: str) -> None:
        await self._hub.close_session(session_id, requested_by=self.actor_id)

    # ------------------------------------------------------------------
    # Sending envelopes
    # ------------------------------------------------------------------

    async def _send_envelope(self, envelope: Envelope) -> tuple[str, int]:
        """Send an envelope and return ``(envelope_id, wal_offset)``.

        ``wal_offset`` is the session WAL byte position *after* this
        envelope's append — callers who want to wait on a correlated reply
        pass it as ``since`` to :meth:`_open_subscription` so the hub
        replays only envelopes that land after the send.
        """

        envelope.sender_id = self.actor_id
        envelope = await self._apply_pre_send_transforms(envelope)

        async with self._send_lock:
            fut: asyncio.Future[AcceptFrame | ErrorFrame] = (
                asyncio.get_event_loop().create_future()
            )
            self._pending_accept = fut
            try:
                await self._link_client.send_frame(
                    SendFrame(
                        envelope=envelope,
                        idempotency_key=envelope.idempotency_key,
                    )
                )
                result = await fut
            finally:
                self._pending_accept = None
        if isinstance(result, ErrorFrame):
            raise _error_for_code(result)
        self._remember_send(result.envelope_id)
        await self._apply_post_send_transforms(envelope)
        return result.envelope_id, result.wal_offset

    def _remember_send(self, envelope_id: str) -> None:
        if len(self._recent_sends) == self._recent_sends.maxlen:
            oldest = self._recent_sends[0]
            self._recent_sends_set.discard(oldest)
        self._recent_sends.append(envelope_id)
        self._recent_sends_set.add(envelope_id)

    def _is_reply_to_recent_send(self, envelope: Envelope) -> bool:
        return (
            bool(envelope.causation_id)
            and envelope.causation_id in self._recent_sends_set
        )

    async def _post_text_reply(self, original: Envelope, content: str) -> str:
        reply = Envelope.text(
            session_id=original.session_id,
            sender_id=self.actor_id,
            content=content,
            recipient_id=original.sender_id,
            causation_id=original.envelope_id,
        )
        # Propagate delegation depth: a reply is "one hop further" in
        # the causation chain, so the hub's depth enforcement sees a
        # monotonically-increasing counter regardless of who is
        # speaking next.
        reply.depth = original.depth + 1
        envelope_id, _ = await self._send_envelope(reply)
        return envelope_id

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    async def _open_subscription(
        self,
        *,
        session_id: str,
        causation_id: str | None = None,
        since: int = 0,
    ) -> asyncio.Queue[Envelope]:
        sub_id = new_id()
        queue: asyncio.Queue[Envelope] = asyncio.Queue()
        # Stash the subscription id on the queue so ``_close_subscription``
        # can find it back without a reverse lookup. Note that this id
        # can be rotated by :meth:`reconnect` — we re-stash there too.
        queue.__dict__["subscription_id"] = sub_id
        sub = _ClientSubscription(
            subscription_id=sub_id,
            queue=queue,
            session_id=session_id,
            causation_id=causation_id,
            since=since,
        )
        self._subs[sub_id] = sub
        fut: asyncio.Future[AcceptFrame | ErrorFrame] = (
            asyncio.get_event_loop().create_future()
        )
        self._pending_sub_accepts[sub_id] = fut
        try:
            await self._link_client.send_frame(
                SubscribeFrame(
                    subscription_id=sub_id,
                    session_id=session_id,
                    causation_id=causation_id,
                    since=since,
                )
            )
            # Wait for the hub's "subscribe accepted + replay done"
            # acknowledgement so the caller never races the hub. This
            # closes the ordering gap between ``_open_subscription``
            # returning and the subscription actually being live on
            # the hub side, which mattered for Phase 3a's reconnect
            # flow and for anything that immediately sends after
            # opening a sub.
            result = await fut
        except BaseException:
            # Clean up on cancellation or send failure — the
            # subscription was never confirmed, so drop the local
            # tracking to avoid a leaked entry on reconnect.
            self._pending_sub_accepts.pop(sub_id, None)
            self._subs.pop(sub_id, None)
            raise
        if isinstance(result, ErrorFrame):
            self._subs.pop(sub_id, None)
            raise _error_for_code(result)
        # The hub stamped the replay-end offset on the accept; seed
        # the cursor so a subsequent reconnect replays from the right
        # position even if no events were delivered in between.
        if result.wal_offset > sub.since:
            sub.since = result.wal_offset
        return queue

    async def _close_subscription(self, queue: asyncio.Queue[Envelope]) -> None:
        sub_id = queue.__dict__.get("subscription_id")
        if sub_id is None:
            return
        self._subs.pop(sub_id, None)
        with contextlib.suppress(LinkClosedError):
            await self._link_client.send_frame(UnsubscribeFrame(subscription_id=sub_id))

    # ------------------------------------------------------------------
    # Chunk streaming (Phase 2)
    # ------------------------------------------------------------------

    async def _send_chunk(
        self,
        *,
        envelope_id: str,
        session_id: str,
        chunk_index: int,
        content: str,
        recipient_id: str | None = None,
        final: bool = False,
    ) -> None:
        """Emit a :class:`ChunkFrame` for an in-flight response.

        Chunks don't go through the ``_send_lock`` accept-correlation
        path because the hub does not accept-reply to chunks — they
        are fire-and-forget relays. The recipient's ActorClient
        routes them into its per-envelope chunk queue.
        """

        await self._link_client.send_frame(
            ChunkFrame(
                envelope_id=envelope_id,
                chunk_index=chunk_index,
                content=content,
                session_id=session_id,
                sender_id=self.actor_id,
                recipient_id=recipient_id,
                final=final,
            )
        )

    def _chunk_queue_for(self, envelope_id: str) -> asyncio.Queue[ChunkFrame]:
        """Get-or-create the chunk queue for ``envelope_id``.

        Clients that want to listen for an upcoming stream call this
        *before* sending the envelope whose reply they expect to
        stream — otherwise fast replies can race ahead of the queue
        creation and drop their chunks on the floor.
        """

        queue = self._chunks.get(envelope_id)
        if queue is None:
            queue = asyncio.Queue()
            self._chunks[envelope_id] = queue
        return queue

    def _discard_chunk_queue(self, envelope_id: str) -> None:
        self._chunks.pop(envelope_id, None)

    # ------------------------------------------------------------------
    # Transforms (empty in Phase 1)
    # ------------------------------------------------------------------

    async def _apply_pre_send_transforms(self, envelope: Envelope) -> Envelope:
        return envelope

    async def _apply_post_send_transforms(self, envelope: Envelope) -> None:
        return None

    async def _apply_pre_receive_transforms(
        self, envelope: Envelope
    ) -> Envelope | None:
        return envelope

    async def _apply_post_receive_transforms(self, envelope: Envelope) -> None:
        return None

    # ------------------------------------------------------------------
    # Inbound frame dispatch
    # ------------------------------------------------------------------

    async def _dispatch_frame(self, frame: Any) -> None:
        if isinstance(frame, WelcomeFrame):
            self._welcome.set()
            return
        if isinstance(frame, NotifyFrame):
            # Run notify handlers in a background task. The handler may
            # call ``client._send_envelope(...)`` which awaits its own
            # AcceptFrame — that frame flows through THIS loop, so calling
            # the handler inline would deadlock.
            task = asyncio.get_event_loop().create_task(self._on_notify(frame.envelope))
            self._handler_tasks.add(task)
            task.add_done_callback(self._handler_tasks.discard)
            return
        if isinstance(frame, EventFrame):
            sub = self._subs.get(frame.subscription_id)
            if sub is not None:
                # Checkpoint the cursor. ``wal_offset`` is the byte
                # position AFTER this envelope in the WAL, so a
                # reconnect that re-subscribes with ``since=<offset>``
                # will see every envelope appended after this one.
                if frame.wal_offset > sub.since:
                    sub.since = frame.wal_offset
                await sub.queue.put(frame.envelope)
            return
        if isinstance(frame, ChunkFrame):
            queue = self._chunks.get(frame.envelope_id)
            if queue is None:
                queue = asyncio.Queue()
                self._chunks[frame.envelope_id] = queue
            await queue.put(frame)
            return
        if isinstance(frame, AcceptFrame):
            # Subscription-accept frames have ``envelope_id=""`` and
            # ``request_id=<subscription_id>``. Route them to the
            # per-sub pending future if one is waiting. Otherwise
            # they fall through to the outbound-send accept path.
            if frame.request_id and frame.request_id in self._pending_sub_accepts:
                fut = self._pending_sub_accepts.pop(frame.request_id)
                if not fut.done():
                    fut.set_result(frame)
                return
            if self._pending_accept is not None and not self._pending_accept.done():
                self._pending_accept.set_result(frame)
            return
        if isinstance(frame, ErrorFrame):
            # ErrorFrames targeting a pending subscribe carry the
            # subscription_id in ``request_id``. Route those to the
            # sub-accept future so ``_open_subscription`` raises.
            if frame.request_id and frame.request_id in self._pending_sub_accepts:
                fut = self._pending_sub_accepts.pop(frame.request_id)
                if not fut.done():
                    fut.set_result(frame)
                return
            if self._pending_accept is not None and not self._pending_accept.done():
                self._pending_accept.set_result(frame)
            return

    async def _on_notify(self, envelope: Envelope) -> None:
        transformed = await self._apply_pre_receive_transforms(envelope)
        if transformed is None:
            await self._nack(envelope, reason="transform_rejected")
            return
        envelope = transformed

        if envelope.event_type.startswith("ag2.session."):
            handler = self._system_handlers.get(envelope.event_type, _noop)
            try:
                await handler(envelope, self)
            except Exception:  # pragma: no cover
                log.exception("system handler error for %s", envelope.event_type)
            await self._ack(envelope)
            await self._apply_post_receive_transforms(envelope)
            return

        # Suppress the default handler for replies to our own sends — the
        # user is driving those via ``session.ask`` or an explicit
        # subscription. Still ack so the hub's WAL records a clean receipt.
        if self._is_reply_to_recent_send(envelope):
            await self._ack(envelope)
            await self._apply_post_receive_transforms(envelope)
            return

        session_type = self._session_type_for(envelope.session_id)
        if session_type is None:
            await self._nack(envelope, reason="unknown_session")
            return
        handler = self._type_handlers.get(session_type)
        if handler is None:
            await self._nack(envelope, reason=f"no handler for {session_type}")
            return
        try:
            await handler(envelope, self)
        except Exception:  # pragma: no cover
            log.exception("handler error for %s", envelope.event_type)
            await self._nack(envelope, reason="handler_error")
            return
        await self._ack(envelope)
        await self._apply_post_receive_transforms(envelope)

    def _session_type_for(self, session_id: str) -> str | None:
        metadata = self._hub.peek_session(session_id)
        return metadata.type if metadata is not None else None

    async def _ack(self, envelope: Envelope) -> None:
        if envelope.envelope_id is None:
            return
        with contextlib.suppress(LinkClosedError):
            await self._link_client.send_frame(
                ReceiptFrame(envelope_id=envelope.envelope_id, status="ack")
            )

    async def _nack(self, envelope: Envelope, *, reason: str) -> None:
        if envelope.envelope_id is None:
            return
        with contextlib.suppress(LinkClosedError):
            await self._link_client.send_frame(
                ReceiptFrame(
                    envelope_id=envelope.envelope_id, status="nack", reason=reason
                )
            )


# ---------------------------------------------------------------------------
# Built-in handlers
# ---------------------------------------------------------------------------


async def _handle_invite(envelope: Envelope, client: ActorClient) -> None:
    ack = Envelope(
        session_id=envelope.session_id,
        sender_id=client.actor_id,
        recipient_id=envelope.sender_id,
        event_type=EV_SESSION_INVITE_ACK,
        event_data={"session_id": envelope.session_id},
        causation_id=envelope.envelope_id,
    )
    try:
        await client._send_envelope(ack)
    except NetworkError:  # pragma: no cover
        log.warning("failed to ack invite", exc_info=True)


async def _noop(envelope: Envelope, client: ActorClient) -> None:
    return None


def _error_for_code(frame: ErrorFrame) -> Exception:
    code = frame.code
    if code == "access_denied":
        return AccessDeniedError(frame.message)
    if code == "limit_exceeded":
        return LimitExceededError(frame.message)
    if code == "rule_violation":
        return RuleViolationError(frame.message)
    if code == "session_closed":
        return SessionClosedError(frame.message)
    if code == "session_error":
        return SessionError(frame.message)
    if code == "unknown_actor":
        return UnknownActorError(frame.message)
    if code == "inbox_full":
        return InboxFullError(frame.message)
    return TransportError(f"{code}: {frame.message}")
