# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""HttpChannel — cross-process HTTP transport for envelopes.

At-least-once delivery with configurable retry. Validates Envelope wire format.
Enables multi-process agent network deployments.

Usage::

    # Process A: receiver
    channel = HttpChannel(host="0.0.0.0", port=8900)
    channel.subscribe(my_handler)
    await channel.start_server()

    # Process B: sender
    channel = HttpChannel(peers=["http://localhost:8900"])
    await channel.send(envelope)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from uuid import uuid4

from aiohttp import ClientSession, ClientTimeout, web

from autogen.beta.context import Context, SubId
from autogen.beta.events.conditions import Condition
from autogen.beta.stream import MemoryStream

from ..primitives.envelope import Envelope

logger = logging.getLogger(__name__)

ChannelCallback = Any  # Callable[[Envelope, Context], Awaitable[None]]


class HttpChannel:
    """HTTP-based cross-process transport for envelopes.

    Operates in two modes (can be both simultaneously):
    - **Server mode**: Receives envelopes via HTTP POST and delivers to subscribers
    - **Client mode**: Sends envelopes to peer URLs via HTTP POST

    Delivery semantics: at-least-once with configurable retry.

    Example::

        # Full-duplex node
        channel = HttpChannel(
            host="0.0.0.0", port=8900,
            peers=["http://peer1:8900", "http://peer2:8900"],
        )
        channel.subscribe(my_handler)
        await channel.start_server()
        await channel.send(envelope)  # Sends to all peers
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8900,
        peers: list[str] | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        self._host = host
        self._port = port
        self._peers = list(peers or [])
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._timeout = timeout

        self._subscribers: dict[SubId, tuple[Condition | None, ChannelCallback]] = {}
        self._stream = MemoryStream()

        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._session: ClientSession | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Server mode
    # ------------------------------------------------------------------

    async def start_server(self) -> None:
        """Start the HTTP server to receive envelopes."""
        self._app = web.Application()
        self._app.router.add_post("/envelope", self._handle_receive)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        logger.info("HttpChannel server started on %s:%d", self._host, self._port)

    async def _handle_receive(self, request: web.Request) -> web.Response:
        """Handle incoming envelope POST."""
        try:
            data = await request.json()
            envelope = Envelope.from_dict(data)

            # Check TTL
            if envelope.is_expired:
                return web.json_response(
                    {"status": "dropped", "reason": "expired"},
                    status=200,
                )

            # Deliver to subscribers
            ctx = Context(stream=self._stream)
            for condition, callback in tuple(self._subscribers.values()):
                if condition and not condition(envelope.event):
                    continue
                await callback(envelope, ctx)

            return web.json_response({"status": "ok"})

        except Exception as e:
            logger.exception("Failed to process incoming envelope")
            return web.json_response(
                {"status": "error", "reason": str(e)},
                status=400,
            )

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response(
            {
                "status": "healthy",
                "subscribers": len(self._subscribers),
                "peers": len(self._peers),
            }
        )

    # ------------------------------------------------------------------
    # Client mode
    # ------------------------------------------------------------------

    async def send(self, envelope: Envelope) -> None:
        """Send an envelope to all peers. At-least-once with retry."""
        if self._closed:
            raise RuntimeError("Channel is closed")

        if not self._peers:
            # Local-only: deliver to subscribers directly
            ctx = Context(stream=self._stream)
            for condition, callback in tuple(self._subscribers.values()):
                if condition and not condition(envelope.event):
                    continue
                await callback(envelope, ctx)
            return

        data = envelope.to_dict()
        payload = json.dumps(data)

        if self._session is None:
            self._session = ClientSession(timeout=ClientTimeout(total=self._timeout))

        # Send to all peers concurrently
        await asyncio.gather(
            *(self._send_to_peer(peer, payload) for peer in self._peers),
            return_exceptions=True,
        )

    async def _send_to_peer(self, peer_url: str, payload: str) -> None:
        """Send to a single peer with retry."""
        url = f"{peer_url.rstrip('/')}/envelope"
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                assert self._session is not None
                async with self._session.post(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    if resp.status == 200:
                        return
                    body = await resp.text()
                    last_error = RuntimeError(f"HTTP {resp.status}: {body}")
            except Exception as e:
                last_error = e

            if attempt < self._max_retries:
                await asyncio.sleep(self._retry_delay * (attempt + 1))

        logger.error(
            "Failed to send envelope to %s after %d attempts: %s",
            peer_url,
            self._max_retries + 1,
            last_error,
        )

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(
        self,
        callback: ChannelCallback,
        *,
        condition: Condition | None = None,
    ) -> SubId:
        """Subscribe to incoming envelopes."""
        sub_id = uuid4()
        self._subscribers[sub_id] = (condition, callback)
        return sub_id

    def unsubscribe(self, sub_id: SubId) -> None:
        self._subscribers.pop(sub_id, None)

    # ------------------------------------------------------------------
    # Peer management
    # ------------------------------------------------------------------

    def add_peer(self, url: str) -> None:
        """Add a peer URL for sending."""
        if url not in self._peers:
            self._peers.append(url)

    def remove_peer(self, url: str) -> None:
        """Remove a peer URL."""
        self._peers = [p for p in self._peers if p != url]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Graceful shutdown."""
        self._closed = True
        if self._session:
            await self._session.close()
            self._session = None
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        self._subscribers.clear()
        logger.info("HttpChannel closed")

    @property
    def url(self) -> str:
        """This node's URL for peers to connect to."""
        return f"http://{self._host}:{self._port}"
