# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import anyio
import httpx
import pytest
from mcp.client.client import Client
from mcp.server.lowlevel import NotificationOptions
from mcp.shared.subscriptions import ResourceUpdated
from mcp.types import ResourceUpdatedNotification

from ag2 import Agent
from ag2.mcp import MCPServer, Resource, ResourceNotifier, ResourceTemplate
from ag2.mcp.errors import MCPResourceNotFoundError
from ag2.mcp.testing import connect, serve
from ag2.testing import TestConfig

_MODERN = "2026-07-28"


def _collect_updates(into: list[str], arrived: anyio.Event | None = None):
    """A ``ClientSession`` message handler recording every resource-updated URI."""

    async def on_message(message: object) -> None:
        if isinstance(message, ResourceUpdatedNotification):
            into.append(str(message.params.uri))
            if arrived is not None:
                arrived.set()

    return on_message


def _agent() -> Agent:
    return Agent("greeter", config=TestConfig("hi"))


def _resource(body: str = "v1") -> Resource:
    return Resource(uri="mem://counter", name="counter", read=lambda: body)


_HANDSHAKE_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


async def _handshake_session(client: httpx.AsyncClient) -> dict[str, str]:
    """Open a handshake-era session and return the headers that address it."""
    response = await client.post(
        "/mcp",
        headers=_HANDSHAKE_HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
    )
    headers = _HANDSHAKE_HEADERS | {"mcp-session-id": response.headers["mcp-session-id"]}
    await client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    return headers


async def _subscribe(client: httpx.AsyncClient, headers: dict[str, str], uri: str) -> dict:
    response = await client.post(
        "/mcp",
        headers=headers,
        json={"jsonrpc": "2.0", "id": "sub", "method": "resources/subscribe", "params": {"uri": uri}},
    )
    return response.json()


async def _unsubscribe(client: httpx.AsyncClient, headers: dict[str, str], uri: str) -> dict:
    response = await client.post(
        "/mcp",
        headers=headers,
        json={"jsonrpc": "2.0", "id": "unsub", "method": "resources/unsubscribe", "params": {"uri": uri}},
    )
    return response.json()


class TestCapability:
    def test_not_advertised_without_a_notifier(self) -> None:
        server = MCPServer(_agent(), resources=[_resource()]).server

        assert server.get_capabilities(NotificationOptions(), {}).resources.subscribe is False

    def test_advertised_in_both_eras(self) -> None:
        server = MCPServer(_agent(), resources=[_resource()], subscriptions=ResourceNotifier()).server

        assert server.get_capabilities(NotificationOptions(), {}).resources.subscribe is True
        modern = server.get_capabilities(NotificationOptions(), {}, protocol_version=_MODERN)
        assert modern.resources.subscribe is True

    def test_stateless_keeps_only_the_modern_half(self) -> None:
        # Handshake delivery pushes into a session an earlier request opened, and
        # a stateless transport keeps none: advertising it would be a promise the
        # server cannot keep, so the handshake handlers go unregistered.
        server = MCPServer(_agent(), resources=[_resource()], subscriptions=ResourceNotifier(), stateless=True).server

        assert server.get_capabilities(NotificationOptions(), {}).resources.subscribe is False
        assert "resources/subscribe" not in server._request_handlers
        # `subscriptions/listen` is carved out of stateless/JSON mode by the SDK
        # and still streams, so the modern era keeps working.
        assert "subscriptions/listen" in server._request_handlers
        modern = server.get_capabilities(NotificationOptions(), {}, protocol_version=_MODERN)
        assert modern.resources.subscribe is True

    def test_list_changed_stays_false_in_the_handshake_era(self) -> None:
        # The resource set is fixed at construction, so there is no list change to
        # announce. The modern era cannot express that — every notification kind
        # rides one stream there — but the handshake era can, and does.
        server = MCPServer(_agent(), resources=[_resource()], subscriptions=ResourceNotifier()).server

        assert server.get_capabilities(NotificationOptions(), {}).resources.list_changed is False


class TestNotifierWiring:
    def test_requires_resources(self) -> None:
        with pytest.raises(ValueError, match="resources"):
            MCPServer(_agent(), subscriptions=ResourceNotifier())

    def test_one_notifier_per_server(self) -> None:
        notifier = ResourceNotifier()
        MCPServer(_agent(), resources=[_resource()], subscriptions=notifier)

        with pytest.raises(ValueError, match="already attached"):
            MCPServer(_agent(), resources=[_resource()], subscriptions=notifier)

    def test_servers_may_share_a_bus(self) -> None:
        first = ResourceNotifier()
        second = ResourceNotifier(bus=first.bus)

        MCPServer(_agent(), resources=[_resource()], subscriptions=first)
        MCPServer(_agent(), resources=[_resource()], subscriptions=second)

        assert second.bus is first.bus

    def test_exposes_the_notifier(self) -> None:
        notifier = ResourceNotifier()

        assert MCPServer(_agent(), resources=[_resource()], subscriptions=notifier).subscriptions is notifier

    def test_no_notifier_when_unconfigured(self) -> None:
        assert MCPServer(_agent(), resources=[_resource()]).subscriptions is None


@pytest.mark.asyncio
class TestPublishValidation:
    async def test_rejects_an_unserved_uri(self) -> None:
        server = MCPServer(_agent(), resources=[_resource()], subscriptions=ResourceNotifier())

        # Strict towards our own code: a typo here would otherwise show up only
        # as notifications that never arrive.
        with pytest.raises(MCPResourceNotFoundError):
            await server.notify_resource_updated("mem://typo")

    async def test_accepts_a_template_match(self) -> None:
        server = MCPServer(
            _agent(),
            resource_templates=[ResourceTemplate("weather://{city}", "weather", lambda v: v["city"])],
            subscriptions=ResourceNotifier(),
        )

        await server.notify_resource_updated("weather://London")

    async def test_unattached_notifier_publishes_anything(self) -> None:
        # Nothing to check a URI against until a server adopts the notifier.
        await ResourceNotifier().notify_resource_updated("mem://whatever")

    async def test_server_without_subscriptions_refuses_to_notify(self) -> None:
        server = MCPServer(_agent(), resources=[_resource()])

        with pytest.raises(ValueError, match="serves no subscriptions"):
            await server.notify_resource_updated("mem://counter")


@pytest.mark.asyncio
class TestHandshakeDelivery:
    async def test_subscriber_is_notified(self) -> None:
        notifier = ResourceNotifier()
        server = MCPServer(_agent(), resources=[_resource()], subscriptions=notifier)
        updated: list[str] = []
        arrived = anyio.Event()

        async with connect(server, message_handler=_collect_updates(updated, arrived)) as session:
            await session.subscribe_resource("mem://counter")
            await server.notify_resource_updated("mem://counter")
            with anyio.fail_after(5):
                await arrived.wait()

        assert updated == ["mem://counter"]

    async def test_unsubscribed_uri_is_not_delivered(self) -> None:
        server = MCPServer(
            _agent(),
            resources=[_resource(), Resource(uri="mem://other", name="other", read=lambda: "o")],
            subscriptions=ResourceNotifier(),
        )
        updated: list[str] = []
        arrived = anyio.Event()

        async with connect(server, message_handler=_collect_updates(updated, arrived)) as session:
            await session.subscribe_resource("mem://counter")
            await server.notify_resource_updated("mem://other")
            # Publishing a subscribed URI second makes the assertion deterministic:
            # events are delivered in order, so its arrival proves the unsubscribed
            # one was already handled — and dropped.
            await server.notify_resource_updated("mem://counter")
            with anyio.fail_after(5):
                await arrived.wait()

        assert updated == ["mem://counter"]

    async def test_unsubscribe_stops_delivery(self) -> None:
        server = MCPServer(
            _agent(),
            resources=[_resource(), Resource(uri="mem://other", name="other", read=lambda: "o")],
            subscriptions=ResourceNotifier(),
        )
        updated: list[str] = []
        arrived = anyio.Event()

        async with connect(server, message_handler=_collect_updates(updated, arrived)) as session:
            await session.subscribe_resource("mem://counter")
            await session.subscribe_resource("mem://other")
            await session.unsubscribe_resource("mem://counter")
            await server.notify_resource_updated("mem://counter")
            # Still-subscribed, published second: its arrival is the barrier that
            # proves the dropped one was not delivered.
            await server.notify_resource_updated("mem://other")
            with anyio.fail_after(5):
                await arrived.wait()

        assert updated == ["mem://other"]

    async def test_unserved_uri_may_be_subscribed(self) -> None:
        # The protocol honors a subscription to a resource that does not exist;
        # it simply never fires. Lenient towards the wire, unlike publishing.
        server = MCPServer(_agent(), resources=[_resource()], subscriptions=ResourceNotifier())

        async with connect(server) as session:
            await session.subscribe_resource("mem://nothing-here")


@pytest.mark.asyncio
class TestSubscriberBound:
    async def test_refuses_a_subscriber_past_the_bound(self) -> None:
        app = MCPServer(
            _agent(),
            resources=[_resource()],
            subscriptions=ResourceNotifier(max_subscribers=1),
            json_response=True,
        )

        async with serve(app) as client:
            first = await _handshake_session(client)
            second = await _handshake_session(client)

            assert "error" not in await _subscribe(client, first, "mem://counter")
            # A connection is only discovered dead when a send to it fails, so the
            # bound refuses rather than evicting someone who may still be listening.
            refused = await _subscribe(client, second, "mem://counter")

        assert refused["error"]["message"] == "Subscription limit reached"

    async def test_a_freed_slot_is_reusable(self) -> None:
        app = MCPServer(
            _agent(),
            resources=[_resource()],
            subscriptions=ResourceNotifier(max_subscribers=1),
            json_response=True,
        )

        async with serve(app) as client:
            first = await _handshake_session(client)
            second = await _handshake_session(client)
            await _subscribe(client, first, "mem://counter")
            await _unsubscribe(client, first, "mem://counter")

            # The bound tracks live subscribers, not connections that ever asked.
            reused = await _subscribe(client, second, "mem://counter")

        assert "error" not in reused


@pytest.mark.asyncio
class TestModernDelivery:
    async def test_listen_stream_receives_the_update(self) -> None:
        app = MCPServer(_agent(), resources=[_resource()], subscriptions=ResourceNotifier())

        async with (
            Client(app.server, mode=_MODERN) as client,
            client.listen(resource_subscriptions=["mem://counter"]) as subscription,
        ):
            await app.notify_resource_updated("mem://counter")
            event = await anext(aiter(subscription))

        assert event == ResourceUpdated(uri="mem://counter")

    async def test_listen_stream_honors_its_filter(self) -> None:
        app = MCPServer(
            _agent(),
            resources=[_resource(), Resource(uri="mem://other", name="other", read=lambda: "o")],
            subscriptions=ResourceNotifier(),
        )
        received: list[object] = []

        async with (
            Client(app.server, mode=_MODERN) as client,
            client.listen(resource_subscriptions=["mem://counter"]) as subscription,
        ):
            await app.notify_resource_updated("mem://other")
            with anyio.move_on_after(0.3):
                async for event in subscription:
                    received.append(event)

        assert received == []

    async def test_survives_a_stateless_server(self) -> None:
        # `subscriptions/listen` is carved out of stateless/JSON mode by the SDK
        # and always streams, so suppressing the handshake half costs the modern
        # era nothing.
        app = MCPServer(
            _agent(),
            resources=[_resource()],
            subscriptions=ResourceNotifier(),
            stateless=True,
            json_response=True,
        )

        async with (
            Client(app.server, mode=_MODERN) as client,
            client.listen(resource_subscriptions=["mem://counter"]) as subscription,
        ):
            await app.notify_resource_updated("mem://counter")
            event = await anext(aiter(subscription))

        assert event == ResourceUpdated(uri="mem://counter")
