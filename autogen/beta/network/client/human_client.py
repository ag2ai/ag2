# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``HumanClient`` — non-LLM participant in a network.

Implements the ``NetworkClient`` Protocol directly (no ``Agent``, no
``NetworkPlugin``, no assembly policies). External code drives it via
two surfaces:

* **Push** — register a coroutine via ``on_envelope(callback)``; it
  fires once per inbound envelope. Multiple callbacks compose;
  exceptions are logged and never propagate to dispatch.
* **Pull** — ``await next_envelope(...)`` blocks until the next
  matching envelope arrives; ``async for env in client.envelopes(): ...``
  streams every inbound envelope until ``disconnect()``.

Outbound sends use the same primitives as ``AgentClient``: ``send`` for
``EV_TEXT`` convenience, ``open`` to initiate a channel, ``post_envelope``
as the escape hatch for adapter-shaped envelopes (e.g. workflow
``EV_PACKET``).

Application embedders attach whatever UI they want (CLI loop, web app,
WebSocket bridge) — the framework provides the participant primitive,
not the input modality.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING

from ..channel import ChannelMetadata
from ..envelope import EV_CHANNEL_INVITE, EV_CHANNEL_INVITE_ACK, EV_TEXT, Envelope
from ..identity import Passport, Resume
from ..rule import Rule
from .channel import Channel

if TYPE_CHECKING:
    from ..hub import Hub
    from .hub_client import HubClient

__all__ = ("HumanClient",)


logger = logging.getLogger(__name__)


EnvelopeCallback = Callable[[Envelope], Awaitable[None]]
EnvelopePredicate = Callable[[Envelope], bool]


class HumanClient:
    """Per-registration handle for a non-LLM participant.

    Satisfies the ``NetworkClient`` Protocol. Constructed by
    ``HubClient.register_human(...)``; not intended for direct
    instantiation.
    """

    def __init__(
        self,
        *,
        passport: Passport,
        resume: Resume,
        rule: Rule,
        hub: "Hub | None",
        hub_client: "HubClient",
        auto_ack_invites: bool = True,
    ) -> None:
        # __init__ stores params; no side effects.
        self._passport = passport
        self._resume = resume
        self._rule = rule
        self._hub = hub
        self._hub_client = hub_client
        self._auto_ack_invites = auto_ack_invites
        self._disconnected = False

        # Push callbacks. Run in registration order; exceptions are
        # logged and do not propagate (a buggy UI callback cannot break
        # the receive loop).
        self._callbacks: list[EnvelopeCallback] = []

        # Pull queue — every inbound envelope is also enqueued here for
        # ``next_envelope`` / ``envelopes()`` consumers. Unbounded by
        # design — the embedder controls drain rate via its UI. If the
        # queue grows pathologically the application has a UI bug to fix.
        # ``None`` is the disconnect sentinel: ``disconnect()`` enqueues it
        # so blocked consumers wake up immediately. Each consumer that
        # observes the sentinel re-enqueues it so concurrent waiters all
        # see end-of-stream.
        self._inbox: asyncio.Queue[Envelope | None] = asyncio.Queue()

        # Per-channel queues for ``wait_for_channel_event``-style waits.
        # Symmetric with ``AgentClient`` so ``Channel`` helpers that need
        # to await a specific reply also work for humans.
        self._channel_inboxes: dict[str, asyncio.Queue[Envelope | None]] = {}

    # ── Properties (NetworkClient surface) ───────────────────────────────────

    @property
    def agent_id(self) -> str:
        if self._passport.agent_id is None:
            raise RuntimeError("HumanClient has unstamped passport (not registered)")
        return self._passport.agent_id

    @property
    def passport(self) -> Passport:
        return self._passport

    @property
    def resume(self) -> Resume:
        return self._resume

    @property
    def rule(self) -> Rule:
        return self._rule

    # ── NetworkClient impl ───────────────────────────────────────────────────

    async def receive(self, envelope: Envelope) -> None:
        """Hub delivery → fan out to callbacks + pull queue.

        Channel invites are auto-acked by default — without this, every
        channel an agent opens to a human would time out at the
        hub-side invite_ack_timeout. Embedders that want to gate
        channel joins (e.g. show an "accept invite?" UI prompt) pass
        ``auto_ack_invites=False`` to ``register_human`` and emit the
        ack themselves.

        Push callbacks run sequentially in registration order. Each
        callback's exception is logged but does not abort dispatch to
        siblings or to the pull queue — the framework treats the UI
        layer as best-effort.
        """
        if self._disconnected:
            return

        # Auto-ack the invite BEFORE fanning out, so the channel can
        # reach ACTIVE before the UI even sees the inbound — same
        # behavior the default agent handler provides.
        if self._auto_ack_invites and envelope.event_type == EV_CHANNEL_INVITE and envelope.sender_id != self.agent_id:
            ack = Envelope(
                channel_id=envelope.channel_id,
                sender_id=self.agent_id,
                audience=None,
                event_type=EV_CHANNEL_INVITE_ACK,
                event_data={"channel_id": envelope.channel_id},
                causation_id=envelope.envelope_id,
            )
            with contextlib.suppress(Exception):
                await self._hub_client.post_envelope(ack)

        # Per-channel inbox (symmetric with AgentClient.ensure_channel_inbox).
        inbox = self._ensure_channel_inbox(envelope.channel_id)
        await inbox.put(envelope)

        # Pull queue — global, no filtering.
        await self._inbox.put(envelope)

        # Push callbacks — defensive against buggy UI code.
        for callback in self._callbacks:
            try:
                await callback(envelope)
            except Exception:
                logger.exception(
                    "human callback raised: human=%s channel=%s event=%s",
                    self._passport.name,
                    envelope.channel_id,
                    envelope.event_type,
                )

    async def disconnect(self) -> None:
        """Stop accepting deliveries and wake any blocked consumers. Idempotent.

        Pull-mode consumers (``next_envelope`` / ``envelopes`` /
        ``wait_for_channel_event``) park on ``Queue.get()``. Without a
        wakeup they would block forever after disconnect — this method
        broadcasts an end-of-stream sentinel into every queue so each
        waiter unblocks and either raises (``next_envelope``) or breaks
        (``envelopes``).
        """
        if self._disconnected:
            return
        self._disconnected = True
        # Sentinel into the global pull queue and every per-channel queue.
        # Each consumer that observes ``None`` re-enqueues it so multiple
        # concurrent waiters all see the disconnect.
        self._inbox.put_nowait(None)
        for inbox in self._channel_inboxes.values():
            inbox.put_nowait(None)

    async def receive_chunk(
        self,
        delta: object,
        *,
        channel_id: str,
        parent_envelope_id: str,
    ) -> None:
        """Streaming chunks are LLM-output flow control; humans see the
        final envelope. No-op for the V1 ``HumanClient`` surface.

        Embedders that want token-level streaming display can subclass
        and forward ``delta`` to their UI; the framework neither
        requires nor blocks that.
        """
        # ``delta`` is typed loosely so subclasses don't need to import
        # ``ChunkDelta`` if they don't override.
        return None

    # ── Push surface ─────────────────────────────────────────────────────────

    def on_envelope(self, callback: EnvelopeCallback) -> None:
        """Register a coroutine fired per inbound envelope.

        Multiple callbacks compose in registration order. Exceptions
        raised by a callback are logged at ``ERROR`` and never
        propagate to the hub's dispatch path.
        """
        self._callbacks.append(callback)

    def remove_envelope_callback(self, callback: EnvelopeCallback) -> None:
        """Detach a previously-registered callback. No-op if absent."""
        with contextlib.suppress(ValueError):
            self._callbacks.remove(callback)

    # ── Pull surface ─────────────────────────────────────────────────────────

    async def next_envelope(
        self,
        *,
        predicate: EnvelopePredicate | None = None,
        timeout: float | None = None,
    ) -> Envelope:
        """Block until the next matching envelope arrives.

        ``predicate=None`` returns the very next envelope. ``timeout``
        in seconds; raises ``asyncio.TimeoutError`` if exceeded.
        Envelopes that don't match the predicate are discarded — pass
        them via ``on_envelope`` if you want both behaviors at once.

        Raises ``RuntimeError`` if the client is (or becomes) disconnected.
        """
        if self._disconnected:
            raise RuntimeError("HumanClient is disconnected")
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout if timeout is not None else None
        while True:
            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                envelope = await asyncio.wait_for(self._inbox.get(), timeout=remaining)
            else:
                envelope = await self._inbox.get()
            if envelope is None:
                # Disconnect sentinel — re-enqueue so concurrent waiters
                # also wake up, then surface the disconnect.
                self._inbox.put_nowait(None)
                raise RuntimeError("HumanClient is disconnected")
            if predicate is None or predicate(envelope):
                return envelope

    async def envelopes(self) -> AsyncIterator[Envelope]:
        """Stream every inbound envelope until disconnect.

        Yields envelopes in arrival order. The iterator terminates when
        ``disconnect()`` is called (consumers blocked on ``get()`` are
        woken via the disconnect sentinel).
        """
        while True:
            envelope = await self._inbox.get()
            if envelope is None:
                # Disconnect — re-enqueue sentinel for concurrent iterators
                # and exit cleanly.
                self._inbox.put_nowait(None)
                return
            yield envelope

    # ── Outbound surface ─────────────────────────────────────────────────────

    async def send(
        self,
        channel_id: str,
        text: str,
        *,
        audience: list[str] | None = None,
        causation_id: str | None = None,
    ) -> str:
        """Post an ``EV_TEXT`` envelope into ``channel_id``. Returns envelope_id.

        Convenience wrapper for the common case. For non-text events
        or adapter-shaped envelopes (e.g. workflow ``EV_PACKET``), build
        the ``Envelope`` directly and call ``post_envelope``.
        """
        envelope = Envelope(
            channel_id=channel_id,
            sender_id=self.agent_id,
            audience=audience,
            event_type=EV_TEXT,
            event_data={"text": text},
            causation_id=causation_id,
        )
        return await self.post_envelope(envelope)

    async def post_envelope(self, envelope: Envelope) -> str:
        """Post an arbitrary envelope through the hub.

        Escape hatch for adapter-shaped envelopes (workflow ``EV_PACKET``
        seeds, etc.) constructed via Layer-2 envelope helpers.
        ``envelope.sender_id`` is stamped if blank.
        """
        if self._disconnected:
            raise RuntimeError("HumanClient is disconnected")
        if envelope.sender_id == "":
            envelope.sender_id = self.agent_id
        return await self._hub_client.post_envelope(envelope)

    async def open(
        self,
        *,
        type: str,
        target: str | list[str],
        ttl: str | int | None = None,
        knobs: dict[str, object] | None = None,
        intent: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> Channel:
        """Open a channel as the initiator and return a ``Channel`` handle.

        ``target`` accepts peer names or agent_ids; resolution goes
        through the bound ``HubClient``.
        """
        if self._disconnected:
            raise RuntimeError("HumanClient is disconnected")

        targets = [target] if isinstance(target, str) else list(target)
        target_ids: list[str] = []
        for t in targets:
            passport = await self._hub_client.get_agent(t)
            if passport.agent_id is None:
                raise RuntimeError(f"target {t!r} has no agent_id")
            target_ids.append(passport.agent_id)

        metadata = await self._hub_client.create_channel(
            creator_id=self.agent_id,
            manifest_type=type,
            participants=target_ids,
            ttl=ttl,
            knobs=knobs,
            intent=intent,
            labels=labels,
        )
        self._ensure_channel_inbox(metadata.channel_id)
        return Channel(metadata=metadata, client=self)  # type: ignore[arg-type]

    async def close_channel(self, channel_id: str, reason: str = "human_closed") -> ChannelMetadata:
        """Close a channel this participant is in."""
        if self._disconnected:
            raise RuntimeError("HumanClient is disconnected")
        return await self._hub_client.close_channel(channel_id, reason=reason)

    # ── Channel.send-compat surface ──────────────────────────────────────────
    #
    # ``Channel`` is back-pointed to its client and calls ``client.agent_id``
    # plus ``client.send_envelope(envelope)``. Mirror the AgentClient names so
    # the Channel handle returned by ``open()`` works without conditional logic.

    async def send_envelope(self, envelope: Envelope) -> str:
        """Channel-compatible alias for ``post_envelope``."""
        return await self.post_envelope(envelope)

    def ensure_channel_inbox(self, channel_id: str) -> "asyncio.Queue[Envelope | None]":
        """Public alias for the per-channel inbox helper.

        ``Channel`` doesn't currently call this on humans, but the
        method exists so any future helper that does inherit the
        AgentClient pattern (ensure → send → wait) works uniformly.
        """
        return self._ensure_channel_inbox(channel_id)

    async def wait_for_channel_event(
        self,
        *,
        channel_id: str,
        predicate: EnvelopePredicate,
        timeout: float = 300.0,
    ) -> Envelope:
        """Block until a channel-scoped envelope matches.

        Symmetric with ``AgentClient.wait_for_channel_event``; useful
        for UIs that want to await a specific reply (e.g. "did the
        consulted agent finish?") without setting up a callback.
        """
        if self._disconnected:
            raise RuntimeError("HumanClient is disconnected")
        inbox = self._ensure_channel_inbox(channel_id)
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            envelope = await asyncio.wait_for(inbox.get(), timeout=remaining)
            if envelope is None:
                inbox.put_nowait(None)
                raise RuntimeError("HumanClient is disconnected")
            if predicate(envelope):
                return envelope

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _ensure_channel_inbox(self, channel_id: str) -> "asyncio.Queue[Envelope | None]":
        inbox = self._channel_inboxes.get(channel_id)
        if inbox is None:
            inbox = asyncio.Queue()
            self._channel_inboxes[channel_id] = inbox
            # If we're already disconnected, immediately seed the sentinel
            # so any new wait_for_channel_event call fails cleanly instead
            # of blocking forever on a fresh queue.
            if self._disconnected:
                inbox.put_nowait(None)
        return inbox

    # Read-only access to the hub_client back-reference for advanced
    # callers (e.g. tests building bespoke envelopes that need adapter
    # state). Mirrors ``AgentClient._hub_client`` exposure pattern.

    @property
    def hub_client(self) -> "HubClient":
        return self._hub_client

    @property
    def hub(self) -> "Hub | None":
        return self._hub
