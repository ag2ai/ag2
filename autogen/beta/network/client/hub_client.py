# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``HubClient`` — one connection to one hub per process.

Lazy-connects the underlying ``LinkClient`` on first ``register``;
demultiplexes inbound ``NotifyFrame``s to the appropriate
``AgentClient``; provides discovery passthroughs.

With ``LocalLink``, the ``HubClient`` holds an explicit in-process
``hub`` reference so register / discovery / mutation cut through wire
serialisation. A cross-process transport keeps ``hub=None`` and runs
every operation through frames.
"""

import asyncio
import contextlib
from typing import TYPE_CHECKING

from autogen.beta.agent import Agent
from autogen.beta.task import TaskMetadata, TaskState

from ..envelope import Envelope
from ..identity import Passport, Resume
from ..rule import Rule
from ..session import SessionMetadata, SessionState
from ..transport.frames import NotifyFrame
from ..transport.local import LocalLink, LocalLinkClient
from ..views.base import ViewPolicy
from .agent_client import AgentClient

if TYPE_CHECKING:
    from ..hub import Hub

__all__ = ("HubClient",)


class HubClient:
    """One connection to a hub. Multiple ``AgentClient``s register through it.

    Takes a ``link`` (currently ``LocalLink``) and an explicit ``hub``
    reference. The link carries dispatched envelopes via
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
                    await self._dispatch_notify(frame)
                # Other frame kinds (Accept/Error/Pong/Event) bypass the
                # demuxer — the in-process send path goes direct via
                # ``Hub.post_envelope`` so ``AcceptFrame`` is unused here.
        except asyncio.CancelledError:
            raise
        except Exception:
            # Receive loops must not propagate.
            pass

    async def _dispatch_notify(self, frame: NotifyFrame) -> None:
        """Route the envelope to the recipient stamped on the frame.

        The hub sets ``recipient_id`` per delivery so broadcasts
        (``audience=None``) reach the right ``AgentClient`` without
        the demuxer re-walking session participants. Frames missing a
        ``recipient_id`` fall back to ``audience``-based routing.
        """
        if frame.recipient_id:
            client = self._clients.get(frame.recipient_id)
            if client is not None:
                await client.receive(frame.envelope)
            return
        if frame.envelope.audience is None:
            return
        for recipient_id in frame.envelope.audience:
            client = self._clients.get(recipient_id)
            if client is not None:
                await client.receive(frame.envelope)

    # ── Registration ─────────────────────────────────────────────────────────

    async def register(
        self,
        agent: Agent,
        passport: Passport,
        resume: Resume,
        *,
        skill_md: str | None = None,
        rule: Rule | None = None,
        attach_plugin: bool = True,
    ) -> AgentClient:
        """Register an agent and return its ``AgentClient`` handle.

        Direct hub call for register (in-process); the resulting
        ``agent_id`` is bound to this connection's endpoint so
        dispatched ``NotifyFrame``s reach the right ``AgentClient``. A
        cross-process transport binds via ``HelloFrame`` instead.

        ``attach_plugin`` is accepted for forward compatibility but
        does nothing here — the LLM-facing tool surface that it
        attaches lives in a layer that is not part of this module.
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

    # ── Hub control-plane passthrough ────────────────────────────────────────
    #
    # Forwards directly to the in-process hub. A cross-process transport
    # would replace these bodies with frame-based RPC; the call sites on
    # ``AgentClient`` / handlers stay the same.

    # — Discovery —

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

    # — Identity mutation —

    async def set_resume(self, agent_id: str, resume: Resume) -> None:
        await self._hub.set_resume(agent_id, resume)

    async def set_skill(self, agent_id: str, skill_md: str | None) -> None:
        await self._hub.set_skill(agent_id, skill_md)

    async def set_rule(self, agent_id: str, rule: Rule) -> None:
        await self._hub.set_rule(agent_id, rule)

    async def unregister_agent(self, agent_id: str) -> None:
        await self._hub.unregister(agent_id)

    # — Session control —

    async def create_session(
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
    ) -> SessionMetadata:
        return await self._hub.create_session(
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

    async def get_session(self, session_id: str) -> SessionMetadata:
        return await self._hub.get_session(session_id)

    async def list_sessions(
        self,
        *,
        agent_id: str | None = None,
        include_terminal: bool = False,
        limit: int = 50,
    ) -> list[SessionMetadata]:
        results = await self._hub.list_sessions(agent_id=agent_id, limit=limit * 4)
        if not include_terminal:
            results = [m for m in results if m.state not in (SessionState.CLOSED, SessionState.EXPIRED)]
        return results[:limit]

    async def close_session(self, session_id: str, *, reason: str = "") -> SessionMetadata:
        return await self._hub.close_session(session_id, reason=reason)

    async def post_envelope(self, envelope: Envelope) -> str:
        return await self._hub.post_envelope(envelope)

    async def read_wal(self, session_id: str, *, since: int = 0, until: int | None = None) -> list[Envelope]:
        return await self._hub.read_wal(session_id, since=since, until=until)

    def can_send(
        self,
        session_id: str,
        sender_id: str,
        *,
        event_type: str | None = None,
    ) -> bool:
        return self._hub.can_send(session_id, sender_id, event_type=event_type)

    def default_view_policy(self, session_id: str, participant_id: str) -> ViewPolicy:
        return self._hub.default_view_policy(session_id, participant_id)

    # — Task observation (network is one observer) —

    async def get_task(self, task_id: str) -> TaskMetadata:
        return await self._hub.get_task(task_id)

    async def list_tasks(
        self,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        state: TaskState | None = None,
        limit: int = 50,
    ) -> list[TaskMetadata]:
        return await self._hub.list_tasks(
            agent_id=agent_id,
            session_id=session_id,
            state=state,
            limit=limit,
        )

    async def observe_task(self, metadata: TaskMetadata) -> None:
        await self._hub.observe_task(metadata)

    async def update_task(
        self,
        task_id: str,
        *,
        state: TaskState | None = None,
        progress: dict[str, object] | None = None,
        result: object | None = None,
        error: str | None = None,
    ) -> None:
        await self._hub.update_task(
            task_id,
            state=state,
            progress=progress,
            result=result,
            error=error,
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
        await self._hub.record_observation(
            owner_id=owner_id,
            capability=capability,
            outcome=outcome,
            latency_ms=latency_ms,
            task_id=task_id,
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
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._receive_task

    async def shutdown(self) -> None:
        """Unregister every ``AgentClient`` then ``close()``."""
        for client in list(self._clients.values()):
            with contextlib.suppress(Exception):
                await client.unregister()
        self._clients.clear()
        await self.close()
