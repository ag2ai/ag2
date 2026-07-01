# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end exercises for the WebSocket transport.

Each test starts a real ``serve_ws`` server on a loopback port and
connects a :class:`WsLinkClient` to it. The wire round-trip uses the
same Frame vocabulary as ``LocalLink`` plus real JSON encoding so any
serialisation issue surfaces here rather than in production.
"""

import asyncio

import pytest

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    ApiKeyAuth,
    AuthBlock,
    AuthRegistry,
    Envelope,
    ErrorFrame,
    HelloFrame,
    Hub,
    HubClient,
    LocalLink,
    NoAuth,
    NotifyFrame,
    Passport,
    ReceiptFrame,
    Resume,
    WelcomeFrame,
    WsLinkClient,
    serve_ws,
)

from ._helpers import ScriptedConfig, wait_for_text_count


def _agent(name: str) -> Agent:
    return Agent(name=name, config=ScriptedConfig())


async def _next_frame(client, timeout: float = 2.0):
    async def first():
        async for frame in client.frames():
            return frame
        raise AssertionError("ws closed before a frame arrived")

    return await asyncio.wait_for(first(), timeout)


async def _new_hub() -> Hub:
    return await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )


def _bound_port(server) -> int:
    return server.sockets[0].getsockname()[1]


class TestHandshake:
    @pytest.mark.asyncio
    async def test_hello_welcome_round_trip(self) -> None:
        hub = await _new_hub()
        await hub.register_identity(Passport(name="alice"), Resume())

        async with serve_ws(hub, "127.0.0.1", 0) as server:
            port = _bound_port(server)
            client = WsLinkClient(f"ws://127.0.0.1:{port}")
            try:
                await client.open()
                await client.send_frame(HelloFrame(name="alice"))
                welcome = await _next_frame(client)
                assert isinstance(welcome, WelcomeFrame)
                assert welcome.endpoint_id != ""
                # The client cached the server-assigned endpoint id.
                assert client.endpoint_id == welcome.endpoint_id
            finally:
                await client.close()
        await hub.close()

    @pytest.mark.asyncio
    async def test_hello_rejects_unknown_name(self) -> None:
        hub = await _new_hub()
        async with serve_ws(hub, "127.0.0.1", 0) as server:
            port = _bound_port(server)
            client = WsLinkClient(f"ws://127.0.0.1:{port}")
            try:
                await client.open()
                await client.send_frame(HelloFrame(name="ghost"))
                frame = await _next_frame(client)
                assert isinstance(frame, ErrorFrame)
                assert frame.code == "not_found"
            finally:
                await client.close()
        await hub.close()


class TestAuthOverWire:
    @pytest.mark.asyncio
    async def test_api_key_happy_path(self) -> None:
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            auth=AuthRegistry([NoAuth(), ApiKeyAuth(keys={"alice": "k-alice"})]),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        await hub.register_identity(
            Passport(name="alice", auth=AuthBlock(scheme="api_key", claim={"token": "k-alice"})),
            Resume(),
        )

        async with serve_ws(hub, "127.0.0.1", 0) as server:
            port = _bound_port(server)
            client = WsLinkClient(f"ws://127.0.0.1:{port}")
            try:
                await client.open()
                await client.send_frame(
                    HelloFrame(name="alice", auth_scheme="api_key", auth_claim={"token": "k-alice"})
                )
                frame = await _next_frame(client)
                assert isinstance(frame, WelcomeFrame)
            finally:
                await client.close()
        await hub.close()

    @pytest.mark.asyncio
    async def test_api_key_wrong_token_returns_auth_failed(self) -> None:
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            auth=AuthRegistry([NoAuth(), ApiKeyAuth(keys={"alice": "k-alice"})]),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        passport = await hub.register_identity(
            Passport(name="alice", auth=AuthBlock(scheme="api_key", claim={"token": "k-alice"})),
            Resume(),
        )

        async with serve_ws(hub, "127.0.0.1", 0) as server:
            port = _bound_port(server)
            client = WsLinkClient(f"ws://127.0.0.1:{port}")
            try:
                await client.open()
                await client.send_frame(
                    HelloFrame(name="alice", auth_scheme="api_key", auth_claim={"token": "k-wrong"})
                )
                frame = await _next_frame(client)
                assert isinstance(frame, ErrorFrame)
                assert frame.code == "auth_failed"
                # Endpoint MUST NOT be bound on auth failure.
                assert passport.agent_id not in hub._agent_to_endpoint
            finally:
                await client.close()
        await hub.close()


async def _setup_active_channel(hub, channel_type: str = "discussion"):
    """Create an active alice/bob channel via the in-process HubClient
    flow (so invite acks land cleanly), then drop both HubClients so
    follow-on tests can attach raw WS clients without binding conflicts.
    Returns (alice_id, bob_id, channel)."""
    local_link = LocalLink(hub)
    alice_hc = HubClient(local_link, hub=hub)
    bob_hc = HubClient(local_link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())
    channel = await alice.open(type=channel_type, target=["bob"])
    await alice_hc.close()
    await bob_hc.close()
    # Give the LocalLink endpoints a tick to drop their bindings via
    # the hub's cleanup-on-handle-exit path.
    await asyncio.sleep(0.05)
    return alice.agent_id, bob.agent_id, channel


class TestEnvelopePostAndDispatch:
    @pytest.mark.asyncio
    async def test_dispatched_envelope_delivers_over_ws(self) -> None:
        """Alice posts a task envelope through the hub; bob — connected
        only via WS — receives it as a NotifyFrame and the receipt he
        sends back advances his cursor."""
        hub = await _new_hub()
        alice_id, bob_id, channel = await _setup_active_channel(hub)

        async with serve_ws(hub, "127.0.0.1", 0) as server:
            port = _bound_port(server)
            bob_ws = WsLinkClient(f"ws://127.0.0.1:{port}")
            try:
                await bob_ws.open()
                await bob_ws.send_frame(HelloFrame(name="bob"))
                welcome = await _next_frame(bob_ws)
                assert isinstance(welcome, WelcomeFrame)
                assert hub._agent_to_endpoint[bob_id] == welcome.endpoint_id

                # Task envelope bypasses the discussion adapter's round-robin.
                missed = Envelope(
                    channel_id=channel.channel_id,
                    sender_id=alice_id,
                    audience=[bob_id],
                    event_type="ag2.task.progress",
                    event_data={"step": "delivered-over-ws"},
                )
                await hub.post_envelope(missed)

                notify = await _next_frame(bob_ws)
                assert isinstance(notify, NotifyFrame)
                assert notify.recipient_id == bob_id
                assert notify.envelope.event_data.get("step") == "delivered-over-ws"

                await bob_ws.send_frame(
                    ReceiptFrame(
                        envelope_id=notify.envelope.envelope_id,
                        status="ack",
                        recipient_id=bob_id,
                        channel_id=channel.channel_id,
                    )
                )
                await asyncio.sleep(0.1)
                assert hub.inbox_cursor(bob_id, channel.channel_id) == notify.envelope.envelope_id
            finally:
                await bob_ws.close()
        await hub.close()

    @pytest.mark.asyncio
    async def test_in_channel_text_send_acks_and_dispatches_in_local_flow(self) -> None:
        """End-to-end text round-trip via the in-process HubClient stack
        keeps working unchanged after the WS surface is wired."""
        hub = await _new_hub()
        local_link = LocalLink(hub)
        alice_hc = HubClient(local_link, hub=hub)
        bob_hc = HubClient(local_link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())
        try:
            channel = await alice.open(type="discussion", target=["bob"])
            await channel.send("hello", audience=[bob.agent_id])
            await wait_for_text_count(hub, channel.channel_id, 1)
        finally:
            await alice_hc.close()
            await bob_hc.close()
            await hub.close()


class TestReplayOverWire:
    @pytest.mark.asyncio
    async def test_since_envelope_id_replays_missed_envelopes(self) -> None:
        hub = await _new_hub()
        alice_id, bob_id, channel = await _setup_active_channel(hub)

        # Post a task envelope addressed to bob while bob is offline.
        missed = Envelope(
            channel_id=channel.channel_id,
            sender_id=alice_id,
            audience=[bob_id],
            event_type="ag2.task.progress",
            event_data={"step": "missed-while-offline"},
        )
        await hub.post_envelope(missed)

        async with serve_ws(hub, "127.0.0.1", 0) as server:
            port = _bound_port(server)
            client = WsLinkClient(f"ws://127.0.0.1:{port}")
            try:
                await client.open()
                # since_envelope_id="" replays anything past the empty
                # cursor — i.e. everything bob hasn't yet acked.
                await client.send_frame(HelloFrame(name="bob", since_envelope_id=""))
                welcome = await _next_frame(client)
                assert isinstance(welcome, WelcomeFrame)

                seen_missed = False
                deadline = asyncio.get_event_loop().time() + 2.0
                while asyncio.get_event_loop().time() < deadline:
                    try:
                        frame = await asyncio.wait_for(_next_frame(client), 0.5)
                    except asyncio.TimeoutError:
                        break
                    if (
                        isinstance(frame, NotifyFrame)
                        and frame.envelope.event_data.get("step") == "missed-while-offline"
                    ):
                        seen_missed = True
                        break
                assert seen_missed
            finally:
                await client.close()
        await hub.close()


class TestConnectionLifecycle:
    @pytest.mark.asyncio
    async def test_client_close_drops_server_endpoint(self) -> None:
        hub = await _new_hub()
        passport = await hub.register_identity(Passport(name="alice"), Resume())
        async with serve_ws(hub, "127.0.0.1", 0) as server:
            port = _bound_port(server)
            client = WsLinkClient(f"ws://127.0.0.1:{port}")
            await client.open()
            await client.send_frame(HelloFrame(name="alice"))
            welcome = await _next_frame(client)
            assert isinstance(welcome, WelcomeFrame)
            assert welcome.endpoint_id in hub._endpoints_by_id
            assert hub._agent_to_endpoint[passport.agent_id] == welcome.endpoint_id

            await client.close()
            await asyncio.sleep(0.2)
            # Endpoint and agent binding both released on disconnect.
            assert welcome.endpoint_id not in hub._endpoints_by_id
            assert passport.agent_id not in hub._agent_to_endpoint
        await hub.close()
