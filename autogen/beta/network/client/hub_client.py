# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``HubClient`` — one connection to one hub per process.

M1 surface: lazy-connects the underlying ``LinkClient`` on first
``register``; demultiplexes inbound frames (only ``NotifyFrame`` in M1)
to the appropriate ``AgentClient``; provides discovery passthroughs.

In V1 (LocalLink only) the ``HubClient`` is constructed with both the
``link`` and a direct ``hub`` reference — frames carry the wire
contract but discovery / register paths cut through to the hub
in-process. Phase 3 adds ``WsLink`` + HTTP discovery, where ``hub`` is
``None`` and every operation goes through frames.
"""

import asyncio
from typing import TYPE_CHECKING

from autogen.beta.agent import Agent

from ..envelope import Envelope
from ..identity import Passport, Resume
from ..rule import Rule
from ..transport.frames import NotifyFrame
from ..transport.local import LocalLink, LocalLinkClient
from .agent_client import AgentClient

if TYPE_CHECKING:
    from ..hub import Hub

__all__ = ("HubClient",)


class HubClient:
    """One connection to a hub. Multiple ``AgentClient``s register through it.

    M1 takes both ``link`` (V1 ``LocalLink`` only) and an explicit
    ``hub`` reference. The link carries dispatched envelopes via
    ``NotifyFrame``; the direct hub reference is used for register /
    discovery / mutation calls (cuts through wire serialisation when
    we're in-process).

    A single tenant process should hold one ``HubClient`` per hub it
    connects to.
    """

    def __init__(self, link: LocalLink, *, hub: "Hub | None" = None) -> None:
        # __init__ stores params; side effects deferred to register()/close().
        self._link = link
        self._hub = hub if hub is not None else link.hub
        self._client_link: LocalLinkClient | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._clients: dict[str, AgentClient] = {}
        self._closed = False

    # ── Connection ───────────────────────────────────────────────────────────

    def _ensure_connected(self) -> LocalLinkClient:
        """Open the link on first use; subsequent calls reuse the connection."""
        if self._client_link is None:
            self._client_link = self._link.client()
            self._receive_task = asyncio.create_task(self._receive_loop())
        return self._client_link

    async def _receive_loop(self) -> None:
        """Demultiplex inbound frames to the appropriate ``AgentClient``."""
        assert self._client_link is not None
        try:
            async for frame in self._client_link.frames():
                if isinstance(frame, NotifyFrame):
                    await self._dispatch_notify(frame.envelope)
                # Other frame kinds (Accept/Error/Pong/Event) are M3 routes —
                # M1's send path goes direct via Hub.post_envelope so AcceptFrame
                # is unused here.
        except asyncio.CancelledError:
            raise
        except Exception:
            # Receive loops must not propagate; M3 audit logs the cause.
            pass

    async def _dispatch_notify(self, envelope: Envelope) -> None:
        if envelope.audience is None:
            # M1 doesn't broadcast (Hub doesn't dispatch broadcasts);
            # M2 wires participant tracking.
            return
        for recipient_id in envelope.audience:
            client = self._clients.get(recipient_id)
            if client is not None:
                await client.receive(envelope)

    # ── Registration ─────────────────────────────────────────────────────────

    async def register(
        self,
        agent: Agent,
        passport: Passport,
        resume: Resume,
        *,
        skill_md: str | None = None,
        rule: Rule | None = None,
    ) -> AgentClient:
        """Register an agent and return its ``AgentClient`` handle.

        M1 simplification: register goes direct to the hub (in-process),
        then the resulting ``agent_id`` is bound to this connection's
        endpoint so dispatched ``NotifyFrame``s reach the right
        ``AgentClient``. M3 / Phase 3 will swap to a ``HelloFrame``-driven
        bind for cross-process correctness.
        """
        if self._closed:
            raise RuntimeError("HubClient is closed")

        client_link = self._ensure_connected()

        effective_rule = rule if rule is not None else Rule()
        passport = await self._hub.register(passport, resume, skill_md=skill_md, rule=effective_rule)
        assert passport.agent_id is not None
        self._hub.bind_endpoint(client_link.endpoint_id, passport.agent_id)

        client = AgentClient(
            agent=agent,
            passport=passport,
            resume=resume,
            rule=effective_rule,
            hub=self._hub,
            hub_client=self,
        )
        self._clients[passport.agent_id] = client
        return client

    # ── Discovery passthrough ────────────────────────────────────────────────

    async def get_agent(self, name_or_id: str) -> Passport:
        return await self._hub.get_agent(name_or_id)

    async def get_resume(self, agent_id: str) -> Resume:
        return await self._hub.get_resume(agent_id)

    async def get_skill(self, agent_id: str) -> str | None:
        return await self._hub.get_skill(agent_id)

    async def list_agents(
        self,
        *,
        capability: str | None = None,
        query: str | None = None,
        sort_by: str | None = None,
        limit: int = 50,
    ) -> list[Passport]:
        return await self._hub.list_agents(
            capability=capability,
            query=query,
            sort_by=sort_by,
            limit=limit,
        )

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the connection and stop the receive loop. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._client_link is not None:
            await self._client_link.close()
        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass

    async def shutdown(self) -> None:
        """Unregister every ``AgentClient`` then ``close()``."""
        for client in list(self._clients.values()):
            try:
                await client.unregister()
            except Exception:
                pass
        self._clients.clear()
        await self.close()
