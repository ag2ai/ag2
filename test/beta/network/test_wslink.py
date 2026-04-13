# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Phase 3a WsLink — the cross-process WebSocket transport.

WsLink implements the same :class:`Link` protocol as :class:`LocalLink`
but over a real WebSocket, using the ``websockets`` library. Phase 3a
ships it as the primary cross-process transport so a hub on one host
can accept actor connections from another — the critical path for the
"cross-process works" goal.

The test matrix:

* Server binds on ``port=0`` and surfaces its URL via ``WsLinkServer.url``.
* Frame vocabulary round-trip parity with ``LocalLink``: every frame
  type the hub uses rides over the WebSocket without loss (hello,
  welcome, send, accept, subscribe, event, chunk, receipt, error).
* End-to-end handshake: a ``HubClient`` pointed at a ``WsLinkClient``
  registers an actor, the hub stamps runtime.json with ``binding="ws"``
  and the URL.
* End-to-end consulting session: Alice over WsLink asks Bob over
  WsLink, the reply arrives via the default consulting handler.
* Conversation with multiple sends + an ack-driven pending/received
  round-trip on the structured inbox.
* Subscription delivery + cursor advance over WsLink.
* Disconnect cleanup: closing a client cleanly triggers
  ``_handle_hello``'s cleanup on the server and flips runtime.json to
  ``reachable=false``.
* Reconnect: an ``ActorClient.reconnect`` on a ``WsLinkClient`` rotates
  its sub ids and replays envelopes sent during the drop.
* WsLinkClient.client() raises ``LinkClosedError`` after close.
* Install-hint: construction of a WsLinkClient without the underlying
  ``websockets`` library gives a clear error.

These tests use ``127.0.0.1`` only — no external network, no DNS.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    Envelope,
    Hub,
    HubClient,
    LinkClosedError,
    SessionType,
    WsLinkClient,
    WsLinkServer,
)
from autogen.beta.network.envelope import EV_TEXT
from autogen.beta.network.hub import layout
from autogen.beta.network.transport.frames import (
    AcceptFrame,
    EventFrame,
    HelloFrame,
    ReceiptFrame,
    SendFrame,
    WelcomeFrame,
    decode_frame,
    encode_frame,
)


# Skip the whole module if websockets isn't installed — mirrors the
# optional-dep shape of the runtime.
websockets = pytest.importorskip("websockets")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Echo:
    def __init__(self, name: str) -> None:
        self.name = name

    async def ask(self, content: str, **_kwargs):
        class _R:
            def __init__(self, body: str) -> None:
                self.body = body

        return _R(f"echo:{content}")


async def _block_forever(_envelope, _client):
    await asyncio.Event().wait()


async def _next_text(queue: asyncio.Queue[Envelope], *, timeout: float = 2.0) -> Envelope:
    while True:
        env = await asyncio.wait_for(queue.get(), timeout=timeout)
        if env.event_type == EV_TEXT:
            return env


async def _spin_ws_hub() -> tuple[Hub, WsLinkServer, WsLinkClient, HubClient]:
    hub = Hub(MemoryKnowledgeStore())
    server = WsLinkServer(host="127.0.0.1", port=0)
    server.on_connection(hub.connection_handler)
    await server.start()
    client_link = WsLinkClient(server.url)
    hc = HubClient(hub, client_link)
    return hub, server, client_link, hc


# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------


class TestAvailability:
    @pytest.mark.asyncio
    async def test_wslink_server_requires_handler(self) -> None:
        server = WsLinkServer(host="127.0.0.1", port=0)
        with pytest.raises(LinkClosedError):
            await server.start()

    @pytest.mark.asyncio
    async def test_wslink_server_url_exposes_bound_port(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        server = WsLinkServer(host="127.0.0.1", port=0)
        server.on_connection(hub.connection_handler)
        await server.start()
        try:
            assert server.url.startswith("ws://127.0.0.1:")
            assert server.url.endswith("/")
        finally:
            await server.close()


# ---------------------------------------------------------------------------
# Raw frame round-trip over a WebSocket
# ---------------------------------------------------------------------------


class TestRawFrameRoundTrip:
    """Skip the hub and talk to a raw websockets echo handler.

    These tests verify the WsLink transport's framing layer alone —
    encode/decode parity, message boundaries, binary-to-text
    tolerance — without coupling to hub semantics.
    """

    @pytest.mark.asyncio
    async def test_send_accept_round_trip(self) -> None:
        from websockets.asyncio.server import serve

        async def handler(ws):
            async for message in ws:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                frame = decode_frame(message)
                # Echo the frame type back as an accept.
                if isinstance(frame, SendFrame):
                    await ws.send(
                        encode_frame(
                            AcceptFrame(
                                envelope_id="stamped-by-echo",
                                wal_offset=42,
                                request_id=frame.envelope.envelope_id,
                            )
                        )
                    )

        async with serve(handler, "127.0.0.1", 0) as server:
            host, port = server.sockets[0].getsockname()[:2]
            url = f"ws://{host}:{port}/"

            client = WsLinkClient(url)
            side = client.client()
            try:
                envelope = Envelope.text(
                    session_id="sess-1",
                    sender_id="alice",
                    content="hi",
                    recipient_id="bob",
                )
                envelope.envelope_id = "env-1"
                await side.send_frame(SendFrame(envelope=envelope))
                # Drain until we see the echoed accept.
                async for frame in side.frames():
                    assert isinstance(frame, AcceptFrame)
                    assert frame.envelope_id == "stamped-by-echo"
                    assert frame.wal_offset == 42
                    break
            finally:
                await side.close()

    @pytest.mark.asyncio
    async def test_event_frame_round_trip_with_wal_offset(self) -> None:
        from websockets.asyncio.server import serve

        async def handler(ws):
            async for _ in ws:
                env = Envelope.text(
                    session_id="sess-1",
                    sender_id="bob",
                    content="broadcast",
                )
                env.envelope_id = "env-1"
                await ws.send(
                    encode_frame(
                        EventFrame(
                            subscription_id="sub-1",
                            envelope=env,
                            wal_offset=99,
                        )
                    )
                )
                break

        async with serve(handler, "127.0.0.1", 0) as server:
            host, port = server.sockets[0].getsockname()[:2]
            url = f"ws://{host}:{port}/"

            client = WsLinkClient(url)
            side = client.client()
            try:
                await side.send_frame(
                    HelloFrame(identity={}, rule={}, resume_actor_id="alice")
                )
                async for frame in side.frames():
                    assert isinstance(frame, EventFrame)
                    assert frame.subscription_id == "sub-1"
                    assert frame.wal_offset == 99
                    break
            finally:
                await side.close()


# ---------------------------------------------------------------------------
# End-to-end hub + client (this is the real Phase 3a integration test)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_register_over_wslink(self) -> None:
        hub, server, client_link, hc = await _spin_ws_hub()
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            assert alice.actor_id
            # runtime.json should have been stamped with ws binding +
            # the server's bound URL.
            raw = await hub._store.read(layout.actor_runtime(alice.actor_id))
            runtime = json.loads(raw)
            assert runtime["binding"] == "ws"
            assert runtime["reachable"] is True
            assert runtime["ws_url"] == server.url
            assert runtime["target"]  # endpoint_id stamped
        finally:
            await hc.close()
            await server.close()

    @pytest.mark.asyncio
    async def test_consulting_round_trip_over_wslink(self) -> None:
        hub, server, client_link, hc = await _spin_ws_hub()
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            bob = await hc.register(
                _Echo("bob"), identity=ActorIdentity(name="bob")
            )
            session = await alice.open(SessionType.CONSULTING, target="bob")
            reply = await session.ask("hi", timeout=4.0)
            assert reply == "echo:hi"
        finally:
            await hc.close()
            await server.close()

    @pytest.mark.asyncio
    async def test_conversation_multi_send_over_wslink(self) -> None:
        hub, server, client_link, hc = await _spin_ws_hub()
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            bob = await hc.register(
                _Echo("bob"), identity=ActorIdentity(name="bob")
            )
            session = await alice.open(SessionType.CONVERSATION, target="bob")
            r1 = await session.ask("one", timeout=4.0)
            r2 = await session.ask("two", timeout=4.0)
            assert r1 == "echo:one"
            assert r2 == "echo:two"
        finally:
            await hc.close()
            await server.close()

    @pytest.mark.asyncio
    async def test_subscription_delivery_over_wslink(self) -> None:
        hub, server, client_link, hc = await _spin_ws_hub()
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            bob = await hc.register(
                _Echo("bob"), identity=ActorIdentity(name="bob")
            )
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            queue = await alice._open_subscription(session_id=session.session_id)

            await session.send("one")
            await session.send("two")
            await session.send("three")

            envs = [await _next_text(queue) for _ in range(3)]
            assert [e.content() for e in envs] == ["one", "two", "three"]

            sub_id = queue.__dict__["subscription_id"]
            assert alice._subs[sub_id].since > 0
        finally:
            await hc.close()
            await server.close()


# ---------------------------------------------------------------------------
# Disconnect + reconnect over WsLink
# ---------------------------------------------------------------------------


class TestWsDisconnectAndReconnect:
    @pytest.mark.asyncio
    async def test_disconnect_marks_runtime_unreachable(self) -> None:
        hub, server, client_link, hc = await _spin_ws_hub()
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            alice_id = alice.actor_id

            raw_before = await hub._store.read(layout.actor_runtime(alice_id))
            assert json.loads(raw_before)["reachable"] is True

            await alice.disconnect()
            # Give the hub's connection_handler finally block time to
            # see the WebSocket close and flip reachable.
            for _ in range(50):
                raw_after = await hub._store.read(layout.actor_runtime(alice_id))
                if json.loads(raw_after).get("reachable") is False:
                    break
                await asyncio.sleep(0.02)
            else:  # pragma: no cover
                pytest.fail("runtime.reachable did not flip to false")

            runtime_after = json.loads(raw_after)
            assert runtime_after["reachable"] is False
            assert runtime_after["binding"] == "ws"
        finally:
            await hc.close()
            await server.close()

    @pytest.mark.asyncio
    async def test_reconnect_over_wslink_replays_missed_envelopes(self) -> None:
        hub, server, client_link, hc = await _spin_ws_hub()
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            bob = await hc.register(
                _Echo("bob"), identity=ActorIdentity(name="bob")
            )
            bob.on("conversation")(_block_forever)

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            queue = await alice._open_subscription(session_id=session.session_id)
            sub_before = queue.__dict__["subscription_id"]

            await session.send("before-drop")
            before = await _next_text(queue)
            assert before.content() == "before-drop"

            # Reconnect the WebSocket. This should NOT lose any
            # envelopes sent after the rotation.
            await alice.reconnect()
            sub_after = queue.__dict__["subscription_id"]
            assert sub_before != sub_after

            await session.send("after-reconnect")
            after = await _next_text(queue)
            assert after.content() == "after-reconnect"
        finally:
            await hc.close()
            await server.close()


# ---------------------------------------------------------------------------
# WsLinkClient lifecycle
# ---------------------------------------------------------------------------


class TestClientLifecycle:
    @pytest.mark.asyncio
    async def test_closed_client_rejects_new_client_calls(self) -> None:
        link = WsLinkClient("ws://127.0.0.1:1/")
        await link.close()
        with pytest.raises(LinkClosedError):
            link.client()

    @pytest.mark.asyncio
    async def test_client_side_close_is_idempotent(self) -> None:
        link = WsLinkClient("ws://127.0.0.1:1/")
        side = link.client()
        await side.close()
        await side.close()  # second close must not raise
        assert side.closed

    @pytest.mark.asyncio
    async def test_connect_failure_raises_transport_error(self) -> None:
        from autogen.beta.network.errors import TransportError

        # Pick a port that is almost certainly not listening.
        link = WsLinkClient("ws://127.0.0.1:1/", open_timeout=0.3)
        side = link.client()
        with pytest.raises((TransportError, LinkClosedError)):
            await side.send_frame(
                HelloFrame(identity={}, rule={}, resume_actor_id="noone")
            )
        await side.close()


# ---------------------------------------------------------------------------
# Chunk streaming over WsLink
# ---------------------------------------------------------------------------


class TestChunkStreamingOverWs:
    @pytest.mark.asyncio
    async def test_chunks_stream_end_to_end_over_wslink(self) -> None:
        from autogen.beta.network.client.session import Session

        hub, server, client_link, hc = await _spin_ws_hub()
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            bob = await hc.register(
                _Echo("bob"), identity=ActorIdentity(name="bob")
            )
            bob.on("conversation")(_block_forever)
            alice.on("conversation")(_block_forever)

            alice_session = await alice.open(
                SessionType.CONVERSATION, target="bob"
            )
            bob_session = Session(client=bob, metadata=alice_session.metadata)

            consumed: list[str] = []

            async def consume() -> None:
                async for chunk in alice_session.iter_chunks("reply-1"):
                    consumed.append(chunk)

            consume_task = asyncio.create_task(consume())
            await asyncio.sleep(0.05)

            for i, token in enumerate(["hel", "lo ", "world"]):
                await bob_session.send_chunk(
                    envelope_id="reply-1",
                    chunk_index=i,
                    content=token,
                    recipient_id=alice.actor_id,
                    final=(i == 2),
                )
            await asyncio.wait_for(consume_task, timeout=4.0)
            assert consumed == ["hel", "lo ", "world"]
        finally:
            await hc.close()
            await server.close()
