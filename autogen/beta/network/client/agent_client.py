# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``AgentClient`` — per-registration tenant handle.

M1 surface: properties (agent, passport, resume, agent_id), receive
callback (M1 testing seam — M2 wires the per-session-type notify
handler registry), tenant-driven mutation (``set_resume`` /
``set_skill`` / ``set_rule``), unregister, disconnect, and a direct
``send_envelope`` helper that bypasses the link's ``SendFrame`` path
for in-process simplicity.

M2 will attach the ``NetworkPlugin`` at registration so the LLM verbs
become ``agent.tools``; M1 keeps the agent untouched.
"""

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from autogen.beta.agent import Agent

from ..envelope import Envelope
from ..identity import Passport, Resume
from ..rule import Rule

if TYPE_CHECKING:
    from ..hub import Hub
    from .hub_client import HubClient

__all__ = ("AgentClient",)


EnvelopeHandler = Callable[[Envelope], Awaitable[None]]


class AgentClient:
    """Tenant-side handle for one ``(Agent, identity, hub)`` registration.

    M1 ships the bare bones — properties, receive callback (testing
    seam), envelope post helper, and tenant-driven mutation passthroughs
    to the hub. The notify-handler registry, ``open(...)`` for sessions,
    and ``NetworkPlugin`` attachment all arrive in M2.
    """

    def __init__(
        self,
        *,
        agent: Agent,
        passport: Passport,
        resume: Resume,
        rule: Rule,
        hub: "Hub",
        hub_client: "HubClient",
    ) -> None:
        # __init__ stores params; no side effects.
        self._agent = agent
        self._passport = passport
        self._resume = resume
        self._rule = rule
        self._hub = hub
        self._hub_client = hub_client
        self._on_envelope: EnvelopeHandler | None = None
        self._disconnected = False

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
        """Hub delivers an envelope (via ``HubClient`` demux).

        M1 routes to ``on_envelope`` callback for testing. M2 dispatches
        to the per-session-type notify handler registry, which in turn
        calls the default handler (read WAL → project view → ask agent
        → send reply).
        """
        if self._on_envelope is not None:
            await self._on_envelope(envelope)

    def on_envelope(self, callback: EnvelopeHandler) -> None:
        """Register a callback for incoming envelopes (M1 testing seam).

        M2 replaces this with the ``@client.on(session_type)`` registry.
        Multiple registrations overwrite; one callback at a time in M1.
        """
        self._on_envelope = callback

    async def disconnect(self) -> None:
        """Drop the AgentClient's local state. Idempotent.

        Does not unregister the identity from the hub — call
        :meth:`unregister` for that. Does not close the underlying
        link — that's owned by ``HubClient``.
        """
        self._disconnected = True
        self._on_envelope = None

    # ── Envelope send (M1 helper — direct hub call) ──────────────────────────

    async def send_envelope(self, envelope: Envelope) -> str:
        """Post an envelope through the hub. Returns the stamped ``envelope_id``.

        M1 simplification: bypasses the link's ``SendFrame`` path and
        calls ``Hub.post_envelope`` directly. This works because V1 is
        in-process; M2 / Phase 3 add the round-trip ``SendFrame`` →
        ``AcceptFrame`` flow with request/response correlation.
        """
        if self._disconnected:
            raise RuntimeError("AgentClient is disconnected")
        if envelope.sender_id == "":
            envelope.sender_id = self.agent_id
        return await self._hub.post_envelope(envelope)

    # ── Tenant-driven mutation ───────────────────────────────────────────────

    async def set_resume(self, resume: Resume) -> None:
        await self._hub.set_resume(self.agent_id, resume)
        self._resume = resume

    async def set_skill(self, skill_md: str | None) -> None:
        await self._hub.set_skill(self.agent_id, skill_md)

    async def set_rule(self, rule: Rule) -> None:
        await self._hub.set_rule(self.agent_id, rule)
        self._rule = rule

    async def unregister(self) -> None:
        """Unregister this identity from the hub.

        After this returns, subsequent ``send_envelope`` calls will fail
        with ``NotFoundError`` (sender not registered). The
        ``AgentClient`` instance becomes inert.
        """
        if not self._disconnected:
            await self._hub.unregister(self.agent_id)
            self._disconnected = True
