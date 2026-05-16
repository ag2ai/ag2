# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``AgentClient`` — per-registration tenant handle.

Surface:

* Properties (agent, passport, resume, agent_id).
* ``receive`` (NetworkClient impl) — routes envelopes to the optional
  per-channel inbox queue (used by ``delegate``) AND to the registered
  notify-handler callback (default = ``handlers.default_handler``,
  which auto-acks invites and runs ``Agent.ask`` on text envelopes).
* ``send_envelope`` — direct ``Hub.post_envelope`` call.
* ``open(type=..., target=..., ...)`` — create a channel via the hub;
  returns a :class:`Channel` handle.
* ``wait_for_channel_event`` — block until an inbound envelope on a
  channel matches a predicate; used by ``delegate`` to await replies.
* Tenant-driven mutation (``set_resume`` / ``set_skill`` / ``set_rule``).
* ``on_envelope(callback)`` — override the default notify handler
  (testing seam).

The ``NetworkPlugin`` is attached at registration by ``HubClient`` so
``agent.tools`` includes ``say`` / ``delegate`` and the assembly chain
includes ``NetworkContextPolicy``.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from autogen.beta.agent import Agent

from ..envelope import Envelope
from ..identity import Passport, Resume, ResumeExample
from ..rule import Rule
from .channel import Channel
from .handlers import default_handler

if TYPE_CHECKING:
    from ..hub import Hub
    from .hub_client import HubClient

__all__ = ("AgentClient",)


EnvelopeHandler = Callable[[Envelope], Awaitable[None]]
EnvelopePredicate = Callable[[Envelope], bool]


class AgentClient:
    """Tenant-side handle for one ``(Agent, identity, hub)`` registration."""

    def __init__(
        self,
        *,
        agent: Agent,
        passport: Passport,
        resume: Resume,
        rule: Rule,
        hub: "Hub",
        hub_client: "HubClient",
        attach_default_handler: bool = True,
    ) -> None:
        # __init__ stores params; no side effects.
        self._agent = agent
        self._passport = passport
        self._resume = resume
        self._rule = rule
        self._hub = hub
        self._hub_client = hub_client
        self._on_envelope: EnvelopeHandler | None = self._run_default_handler if attach_default_handler else None
        self._disconnected = False

        # Per-channel inbox queues for ``wait_for_channel_event``
        # (used by the ``delegate`` tool to await consulting replies).
        self._channel_inboxes: dict[str, asyncio.Queue[Envelope]] = {}

        # Channels where the default notify handler should NOT run —
        # used by ``delegate`` while it owns the channel lifecycle.
        self._handler_suppressed_channels: set[str] = set()

        # Stack of envelopes currently being handled. The top of the
        # stack is the envelope this agent is processing right now;
        # ``delegate`` reads its ``depth`` to stamp the outgoing prompt
        # for delegation-depth enforcement (Rule.limits.delegation_depth).
        self._handling_envelope_stack: list[Envelope] = []

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def passport(self) -> Passport:
        return self._passport

    @property
    def resume(self) -> Resume:
        return self._resume

    @property
    def rule(self) -> Rule:
        return self._rule

    @property
    def agent_id(self) -> str:
        if self._passport.agent_id is None:
            raise RuntimeError("AgentClient has unstamped passport (not registered)")
        return self._passport.agent_id

    # ── NetworkClient impl ───────────────────────────────────────────────────

    async def receive(self, envelope: Envelope) -> None:
        """Hub delivery → fan out to inbox + (suppressible) handler."""
        inbox = self.ensure_channel_inbox(envelope.channel_id)
        await inbox.put(envelope)
        if envelope.channel_id in self._handler_suppressed_channels:
            return
        if self._on_envelope is not None:
            await self._on_envelope(envelope)

    def on_envelope(self, callback: EnvelopeHandler) -> None:
        """Override the default notify handler with a custom callback.

        Calling with the default handler restores it: pass
        ``self._run_default_handler`` (or simply construct without
        ``attach_default_handler=False``).
        """
        self._on_envelope = callback

    async def disconnect(self) -> None:
        self._disconnected = True
        self._on_envelope = None

    async def _run_default_handler(self, envelope: Envelope) -> None:
        """Bound-method wrapper around :func:`handlers.default_handler`.

        Pushes the inbound envelope onto the handling stack so any
        ``delegate``/``channels.open`` invoked from inside the LLM turn
        can stamp ``Envelope.depth = outer.depth + 1`` and the hub can
        enforce ``Rule.limits.delegation_depth``.
        """
        self._handling_envelope_stack.append(envelope)
        try:
            await default_handler(envelope, self)
        finally:
            self._handling_envelope_stack.pop()

    @property
    def current_handling_depth(self) -> int:
        """Depth of the envelope this agent is currently handling.

        Returns ``0`` when no handler is on the stack (i.e. the agent
        initiated the call from outside any inbound delivery). Used by
        ``delegate`` to stamp ``Envelope.depth = current + 1``.
        """
        if not self._handling_envelope_stack:
            return 0
        return self._handling_envelope_stack[-1].depth

    # ── Channel lifecycle ────────────────────────────────────────────────────

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
        """Open a channel via the hub and return its :class:`Channel` handle.

        ``target`` accepts peer **names** or agent_ids; resolution goes
        through the bound :class:`HubClient` so in-process and any
        future cross-process transport take the same code path.
        """
        if self._disconnected:
            raise RuntimeError("AgentClient is disconnected")

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
        self.ensure_channel_inbox(metadata.channel_id)
        return Channel(metadata=metadata, client=self)

    def ensure_channel_inbox(self, channel_id: str) -> "asyncio.Queue[Envelope]":
        """Create (or fetch) the per-channel inbox queue.

        Callers that send first and then ``wait_for_channel_event`` MUST
        call this BEFORE the send. Otherwise a fast reply (e.g. via
        ``LocalLink`` where dispatch happens on the same event-loop tick)
        can be delivered to ``receive`` before the wait creates the
        inbox — the envelope would then be dropped silently.

        Idempotent: returns the existing queue if one is already bound.
        """
        inbox = self._channel_inboxes.get(channel_id)
        if inbox is None:
            inbox = asyncio.Queue()
            self._channel_inboxes[channel_id] = inbox
        return inbox

    def discard_channel_inbox(self, channel_id: str) -> None:
        """Drop the per-channel inbox queue.

        Callers should invoke this after they've finished waiting on a
        channel so the per-client memory footprint doesn't grow with
        every consulted channel.
        """
        self._channel_inboxes.pop(channel_id, None)

    async def wait_for_channel_event(
        self,
        *,
        channel_id: str,
        predicate: EnvelopePredicate,
        timeout: float = 300.0,
    ) -> Envelope:
        """Block until an inbound envelope on ``channel_id`` matches.

        Used by ``delegate`` to await the consulting respondent's
        reply. The inbox is created on demand and shared across waits;
        callers should not hold multiple concurrent waits on the same
        channel.

        Raises ``asyncio.TimeoutError`` on timeout.
        """
        inbox = self.ensure_channel_inbox(channel_id)

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            envelope = await asyncio.wait_for(inbox.get(), timeout=remaining)
            if predicate(envelope):
                return envelope

    def _suppress_handler(self, channel_id: str) -> None:
        """Internal: stop running the default notify handler for ``channel_id``.

        Used by ``delegate`` to own the channel lifecycle while waiting
        for the respondent's reply — the default handler would
        otherwise try to ``Agent.ask`` on every inbound EV_TEXT.
        """
        self._handler_suppressed_channels.add(channel_id)

    def _unsuppress_handler(self, channel_id: str) -> None:
        self._handler_suppressed_channels.discard(channel_id)

    # ── Envelope send ────────────────────────────────────────────────────────

    async def send_envelope(self, envelope: Envelope) -> str:
        """Post an envelope through the hub. Returns the stamped envelope_id."""
        if self._disconnected:
            raise RuntimeError("AgentClient is disconnected")
        if envelope.sender_id == "":
            envelope.sender_id = self.agent_id
        if envelope.trace_id is None:
            try:
                from opentelemetry import trace as _otel_trace

                sc = _otel_trace.get_current_span().get_span_context()
                if sc.is_valid:
                    envelope.trace_id = f"00-{sc.trace_id:032x}-{sc.span_id:016x}-{sc.trace_flags:02x}"
            except ImportError:
                pass
        return await self._hub_client.post_envelope(envelope)

    # ── Tenant-driven mutation ───────────────────────────────────────────────

    async def set_resume(self, resume: Resume) -> None:
        await self._hub_client.set_resume(self.agent_id, resume)
        # Refresh local cache so subsequent reads see the bumped version.
        self._resume = await self._hub_client.get_resume(self.agent_id)

    async def add_example(self, example: ResumeExample) -> None:
        """Append a ``ResumeExample`` to this agent's resume.

        Fetches the latest resume from the hub first so concurrent
        ``set_resume`` / ``record_observation`` updates don't get
        clobbered.
        """
        current = await self._hub_client.get_resume(self.agent_id)
        current.examples.append(example)
        await self.set_resume(current)

    async def set_skill(self, skill_md: str | None) -> None:
        await self._hub_client.set_skill(self.agent_id, skill_md)

    async def set_rule(self, rule: Rule) -> None:
        await self._hub_client.set_rule(self.agent_id, rule)
        self._rule = rule

    async def unregister(self) -> None:
        if not self._disconnected:
            await self._hub_client.unregister_agent(self.agent_id)
            self._disconnected = True

    async def __aenter__(self) -> "AgentClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.unregister()
