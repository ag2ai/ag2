# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Test harness utilities.

These helpers live outside the public package so Phase 1c tests can exercise
the Hub's Link-driven code paths without building the full ActorClient
(which is Phase 1d). The harness spins up a bare Link client, sends hello,
and optionally auto-acks invites or echoes replies.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from autogen.beta.network.envelope import (
    EV_SESSION_CLOSED,
    EV_SESSION_INVITE,
    EV_SESSION_INVITE_ACK,
    EV_TEXT,
    Envelope,
)
from autogen.beta.network.hub import Hub
from autogen.beta.network.transport import (
    AcceptFrame,
    ErrorFrame,
    EventFrame,
    HelloFrame,
    LocalLink,
    NotifyFrame,
    ReceiptFrame,
    SendFrame,
    SubscribeFrame,
    WelcomeFrame,
)


HandlerFn = Callable[["FakeClient", Envelope], Awaitable[None]]


def attach_hub_to_link(hub: Hub) -> LocalLink:
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    return link


@dataclass
class FakeClient:
    """Bare Link-connected client used by Phase 1c tests.

    It speaks the wire vocabulary directly — ``hello`` on start, then an
    inbox loop that delivers every inbound frame into a queue. Tests can
    subscribe to notify frames and trigger behaviors from the outside, or
    register a handler callback that runs inside the loop.
    """

    hub: Hub
    link: LocalLink
    actor_id: str
    handler: HandlerFn | None = None

    welcome: asyncio.Event = field(default_factory=asyncio.Event)
    notify_queue: asyncio.Queue[Envelope] = field(default_factory=asyncio.Queue)
    event_queue: asyncio.Queue[Envelope] = field(default_factory=asyncio.Queue)
    accept_map: dict[str, str] = field(default_factory=dict)
    _client_handle: object = None
    _loop_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._client_handle = self.link.client()
        await self._client_handle.send_frame(  # type: ignore[union-attr]
            HelloFrame(identity={}, rule={}, resume_actor_id=self.actor_id)
        )
        self._loop_task = asyncio.get_event_loop().create_task(self._loop())
        # Wait for welcome before returning.
        await self.welcome.wait()

    async def stop(self) -> None:
        if self._client_handle is not None:
            await self._client_handle.close()  # type: ignore[union-attr]
        if self._loop_task is not None:
            try:
                await self._loop_task
            except Exception:
                pass

    async def _loop(self) -> None:
        assert self._client_handle is not None
        async for frame in self._client_handle.frames():  # type: ignore[union-attr]
            if isinstance(frame, WelcomeFrame):
                self.welcome.set()
                continue
            if isinstance(frame, NotifyFrame):
                await self.notify_queue.put(frame.envelope)
                if self.handler is not None:
                    await self.handler(self, frame.envelope)
                continue
            if isinstance(frame, EventFrame):
                await self.event_queue.put(frame.envelope)
                continue
            if isinstance(frame, AcceptFrame):
                # Correlate via the envelope_id we just sent.
                if frame.request_id:
                    self.accept_map[frame.request_id] = frame.envelope_id
                continue
            if isinstance(frame, ErrorFrame):
                continue

    async def send_text(
        self,
        *,
        session_id: str,
        content: str,
        recipient_id: str | None = None,
        causation_id: str | None = None,
    ) -> None:
        assert self._client_handle is not None
        env = Envelope.text(
            session_id=session_id,
            sender_id=self.actor_id,
            content=content,
            recipient_id=recipient_id,
            causation_id=causation_id,
        )
        await self._client_handle.send_frame(SendFrame(envelope=env))  # type: ignore[union-attr]

    async def ack_invite(self, invite: Envelope) -> None:
        assert self._client_handle is not None
        ack = Envelope(
            session_id=invite.session_id,
            sender_id=self.actor_id,
            recipient_id=invite.sender_id,
            event_type=EV_SESSION_INVITE_ACK,
            event_data={"session_id": invite.session_id},
            causation_id=invite.envelope_id,
        )
        await self._client_handle.send_frame(SendFrame(envelope=ack))  # type: ignore[union-attr]

    async def subscribe(
        self,
        *,
        session_id: str | None = None,
        causation_id: str | None = None,
        since: int = 0,
    ) -> str:
        assert self._client_handle is not None
        from autogen.beta.network.ids import new_id

        sub_id = new_id()
        await self._client_handle.send_frame(  # type: ignore[union-attr]
            SubscribeFrame(
                subscription_id=sub_id,
                session_id=session_id,
                causation_id=causation_id,
                since=since,
            )
        )
        return sub_id

    async def next_event(self) -> Envelope:
        return await self.event_queue.get()


async def auto_ack_and_reply(client: FakeClient, envelope: Envelope) -> None:
    """Handler that auto-acks invites and echoes text content back.

    Useful for tests that need a responder without constructing a full
    ActorClient.
    """

    if envelope.event_type == EV_SESSION_INVITE:
        await client.ack_invite(envelope)
        return
    if envelope.event_type == EV_TEXT:
        await client.send_text(
            session_id=envelope.session_id,
            content=f"echo: {envelope.event_data['content']}",
            recipient_id=envelope.sender_id,
            causation_id=envelope.envelope_id,
        )
        return
    if envelope.event_type == EV_SESSION_CLOSED:
        return


async def auto_ack_only(client: FakeClient, envelope: Envelope) -> None:
    if envelope.event_type == EV_SESSION_INVITE:
        await client.ack_invite(envelope)
