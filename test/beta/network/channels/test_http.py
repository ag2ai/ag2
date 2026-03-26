# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from autogen.beta.events import ModelMessage
from autogen.beta.network.primitives.envelope import Envelope

aiohttp = pytest.importorskip("aiohttp")

from autogen.beta.network.channels.http import HttpChannel  # noqa: E402


class TestHttpChannelLocalOnly:
    """Tests without starting a server — local subscriber delivery."""

    @pytest.mark.asyncio
    async def test_local_send_and_receive(self) -> None:
        channel = HttpChannel()
        received: list = []

        async def handler(envelope, ctx):
            received.append(envelope)

        channel.subscribe(handler)

        env = Envelope(event=ModelMessage(content="hello"), sender="a")
        await channel.send(env)

        assert len(received) == 1
        assert received[0].sender == "a"

    @pytest.mark.asyncio
    async def test_unsubscribe(self) -> None:
        channel = HttpChannel()
        received: list = []

        async def handler(envelope, ctx):
            received.append(envelope)

        sub_id = channel.subscribe(handler)
        await channel.send(Envelope(event=ModelMessage(content="m1"), sender="a"))
        channel.unsubscribe(sub_id)
        await channel.send(Envelope(event=ModelMessage(content="m2"), sender="a"))

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_close_rejects(self) -> None:
        channel = HttpChannel()
        await channel.close()

        with pytest.raises(RuntimeError, match="closed"):
            await channel.send(Envelope(event=ModelMessage(content="m"), sender="a"))


class TestHttpChannelCrossProcess:
    """Tests with real HTTP server and client."""

    @pytest.mark.asyncio
    async def test_send_and_receive_via_http(self) -> None:
        received: list = []

        async def handler(envelope, ctx):
            received.append(envelope)

        # Start receiver
        receiver = HttpChannel(host="127.0.0.1", port=18901)
        receiver.subscribe(handler)
        await receiver.start_server()

        try:
            # Sender targets receiver
            sender = HttpChannel(peers=["http://127.0.0.1:18901"])

            env = Envelope(
                event=ModelMessage(content="cross-process hello"),
                sender="process-a",
                recipient="process-b",
            )
            await sender.send(env)

            # Wait for delivery
            await asyncio.sleep(0.1)

            assert len(received) == 1
            assert received[0].sender == "process-a"
            assert received[0].event.content == "cross-process hello"

            await sender.close()
        finally:
            await receiver.close()

    @pytest.mark.asyncio
    async def test_expired_envelope_dropped(self) -> None:
        received: list = []

        async def handler(envelope, ctx):
            received.append(envelope)

        receiver = HttpChannel(host="127.0.0.1", port=18902)
        receiver.subscribe(handler)
        await receiver.start_server()

        try:
            sender = HttpChannel(peers=["http://127.0.0.1:18902"])

            # Create already-expired envelope
            env = Envelope(
                event=ModelMessage(content="expired"),
                sender="a",
                ttl=0.0,
                timestamp=0.0,
            )
            await sender.send(env)
            await asyncio.sleep(0.1)

            assert len(received) == 0  # Dropped by receiver

            await sender.close()
        finally:
            await receiver.close()

    @pytest.mark.asyncio
    async def test_health_endpoint(self) -> None:
        from aiohttp import ClientSession

        receiver = HttpChannel(host="127.0.0.1", port=18903)
        await receiver.start_server()

        try:
            async with ClientSession() as session, session.get("http://127.0.0.1:18903/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "healthy"
        finally:
            await receiver.close()

    @pytest.mark.asyncio
    async def test_peer_management(self) -> None:
        channel = HttpChannel()
        assert len(channel._peers) == 0

        channel.add_peer("http://localhost:9000")
        assert len(channel._peers) == 1

        channel.add_peer("http://localhost:9000")  # Duplicate
        assert len(channel._peers) == 1

        channel.remove_peer("http://localhost:9000")
        assert len(channel._peers) == 0
