# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""LocalLink tests — bidirectional frame delivery + close semantics."""

from __future__ import annotations

import asyncio

import pytest

from autogen.beta.network.envelope import Envelope
from autogen.beta.network.errors import LinkClosedError
from autogen.beta.network.transport import (
    HelloFrame,
    LocalLink,
    NotifyFrame,
    WelcomeFrame,
)
from autogen.beta.network.transport.link import LinkClient, LinkEndpoint


def test_local_link_client_to_server_delivery() -> None:
    async def run() -> list[str]:
        link = LocalLink()
        received_names: list[str] = []

        async def handler(endpoint) -> None:  # type: ignore[no-untyped-def]
            async for frame in endpoint.frames():
                if isinstance(frame, HelloFrame):
                    received_names.append(frame.identity["name"])
                    await endpoint.send_frame(WelcomeFrame(actor_id="act-1", hub_id="hub-1"))
                    break

        link.on_connection(handler)
        client = link.client()
        await client.send_frame(HelloFrame(identity={"name": "alice"}, rule={}))
        async for frame in client.frames():
            assert isinstance(frame, WelcomeFrame)
            assert frame.actor_id == "act-1"
            break
        await client.close()
        await link.close()
        return received_names

    assert asyncio.run(run()) == ["alice"]


def test_local_link_bidirectional_notify_and_receipt() -> None:
    async def run() -> list[str]:
        link = LocalLink()
        receipts: list[str] = []

        async def handler(endpoint) -> None:  # type: ignore[no-untyped-def]
            async for frame in endpoint.frames():
                if isinstance(frame, HelloFrame):
                    env = Envelope.text(session_id="s", sender_id="hub", content="ping")
                    env.envelope_id = "env-1"
                    await endpoint.send_frame(NotifyFrame(envelope=env))
                else:
                    if hasattr(frame, "envelope_id"):
                        receipts.append(frame.envelope_id)
                        break

        link.on_connection(handler)
        client = link.client()
        await client.send_frame(HelloFrame(identity={"name": "bob"}, rule={}))

        async for frame in client.frames():
            assert isinstance(frame, NotifyFrame)
            from autogen.beta.network.transport import ReceiptFrame

            await client.send_frame(ReceiptFrame(envelope_id=frame.envelope.envelope_id or ""))
            break

        # Give handler a chance to process the receipt before we close.
        for _ in range(50):
            if receipts:
                break
            await asyncio.sleep(0)

        await client.close()
        await link.close()
        return receipts

    assert asyncio.run(run()) == ["env-1"]


def test_local_link_client_close_raises_on_subsequent_send() -> None:
    async def run() -> None:
        link = LocalLink()

        async def handler(endpoint) -> None:  # type: ignore[no-untyped-def]
            async for _ in endpoint.frames():
                pass

        link.on_connection(handler)
        client = link.client()
        await client.send_frame(HelloFrame(identity={}, rule={}))
        await client.close()
        with pytest.raises(LinkClosedError):
            await client.send_frame(HelloFrame(identity={}, rule={}))
        await link.close()

    asyncio.run(run())


def test_local_link_client_iteration_stops_on_close() -> None:
    async def run() -> list[object]:
        link = LocalLink()

        async def handler(endpoint) -> None:  # type: ignore[no-untyped-def]
            async for _ in endpoint.frames():
                pass

        link.on_connection(handler)
        client = link.client()
        collected: list[object] = []

        async def drain() -> None:
            async for frame in client.frames():
                collected.append(frame)

        drain_task = asyncio.get_event_loop().create_task(drain())
        await asyncio.sleep(0)
        await client.close()
        await drain_task
        await link.close()
        return collected

    assert asyncio.run(run()) == []


def test_local_link_without_handler_refuses_client() -> None:
    async def run() -> None:
        link = LocalLink()
        with pytest.raises(LinkClosedError):
            link.client()
        await link.close()

    asyncio.run(run())


def test_local_link_sides_conform_to_protocols() -> None:
    """_ClientSide and _EndpointSide must satisfy the Link protocols.

    This is what keeps Phase 3 WsLink additive — a new transport only has
    to satisfy LinkClient + LinkEndpoint; nothing above the transport changes.
    """

    async def run() -> None:
        link = LocalLink()
        captured: list[LinkEndpoint] = []

        async def handler(endpoint: LinkEndpoint) -> None:
            captured.append(endpoint)
            async for _ in endpoint.frames():
                break

        link.on_connection(handler)
        client = link.client()
        assert isinstance(client, LinkClient)
        await client.send_frame(HelloFrame(identity={"name": "x"}, rule={}))
        # Give the handler a chance to run so captured is populated.
        for _ in range(10):
            if captured:
                break
            await asyncio.sleep(0)
        assert captured, "handler never ran"
        assert isinstance(captured[0], LinkEndpoint)
        await client.close()
        await link.close()

    asyncio.run(run())


def test_local_link_two_clients_are_isolated() -> None:
    async def run() -> tuple[str, str]:
        link = LocalLink()

        async def handler(endpoint) -> None:  # type: ignore[no-untyped-def]
            async for frame in endpoint.frames():
                if isinstance(frame, HelloFrame):
                    await endpoint.send_frame(
                        WelcomeFrame(
                            actor_id=f"act-{frame.identity['name']}",
                            hub_id="hub",
                        )
                    )
                    break

        link.on_connection(handler)

        client_a = link.client()
        client_b = link.client()
        await client_a.send_frame(HelloFrame(identity={"name": "alice"}, rule={}))
        await client_b.send_frame(HelloFrame(identity={"name": "bob"}, rule={}))

        async def read_one(c) -> str:  # type: ignore[no-untyped-def]
            async for frame in c.frames():
                assert isinstance(frame, WelcomeFrame)
                return frame.actor_id
            return ""

        a = await read_one(client_a)
        b = await read_one(client_b)

        await client_a.close()
        await client_b.close()
        await link.close()
        return a, b

    assert asyncio.run(run()) == ("act-alice", "act-bob")
