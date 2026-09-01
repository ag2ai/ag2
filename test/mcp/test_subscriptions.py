# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import json
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

import anyio
import httpx
import pytest
from dirty_equals import IsPartialDict
from mcp.client.client import Client
from mcp.server.lowlevel import NotificationOptions
from mcp.server.subscriptions import ListenHandler
from mcp.shared.exceptions import MCPError
from mcp.shared.subscriptions import ResourceUpdated
from mcp.types import ResourceUpdatedNotification, ServerCapabilities
from mcp_types.version import LATEST_MODERN_VERSION

from ag2 import Agent
from ag2.mcp import MCPServer, Prompt, Resource, ResourceNotifier, ResourceTemplate
from ag2.mcp.errors import MCPResourceNotFoundError
from ag2.mcp.security import AccessToken, TokenVerifier, oauth2_scheme, require
from ag2.mcp.testing import connect, serve
from ag2.testing import TestConfig

_MODERN = LATEST_MODERN_VERSION


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


def _other() -> Resource:
    return Resource(uri="mem://other", name="other", read=lambda: "o")


class _Verifier(TokenVerifier):
    """A token verifier stub: the failing check under test never consults it."""

    async def verify_token(self, token: str) -> AccessToken | None:
        return None


def _prompt() -> Prompt:
    return Prompt(name="greet", render=lambda _: "hello")


def _capabilities(server: MCPServer, *, modern: bool = False) -> ServerCapabilities:
    """The capability block a client would see at ``initialize``, for one era.

    The one assertion target in this suite that reads the server rather than the
    wire, because this block *is* what the wire carries at initialize.
    """
    if modern:
        return server.server.get_capabilities(NotificationOptions(), {}, protocol_version=_MODERN)
    return server.server.get_capabilities(NotificationOptions(), {})


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


def _reply(response: httpx.Response) -> dict:
    """The JSON-RPC reply in ``response``, whether the transport chose JSON or SSE.

    ``json_response=True`` answers a POST with a JSON body; the default answers
    it with a one-event SSE stream. Both carry the same reply, and which one a
    test gets should not decide how it reads it.
    """
    if response.headers["content-type"].startswith("text/event-stream"):
        for line in response.text.splitlines():
            if line.startswith("data: "):
                return json.loads(line.removeprefix("data: "))
        raise AssertionError(f"no data frame in SSE response: {response.text!r}")
    return response.json()


async def _subscribe(client: httpx.AsyncClient, headers: dict[str, str], uri: str) -> dict:
    response = await client.post(
        "/mcp",
        headers=headers,
        json={"jsonrpc": "2.0", "id": "sub", "method": "resources/subscribe", "params": {"uri": uri}},
    )
    return _reply(response)


@asynccontextmanager
async def _standalone_stream(
    app: MCPServer, headers: dict[str, str], *, wedge: bool = False
) -> AsyncGenerator[asyncio.Queue[bytes]]:
    """Open the connection's standalone SSE stream and yield its arriving chunks.

    ``httpx.ASGITransport`` buffers a response whole, so it cannot read a stream
    that never ends. Driving the ASGI app directly is the only way to observe
    what a handshake-era client actually receives — which is the seam that proves
    connection identity, since an in-memory client session has no session id.

    ``wedge=True`` accepts the response but never completes a body write, which
    is what a client that stopped reading looks like from the server's side.
    """
    chunks: asyncio.Queue[bytes] = asyncio.Queue()
    disconnected = asyncio.Event()
    opened = asyncio.Event()

    async def receive() -> dict[str, object]:
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        if message["type"] == "http.response.start":
            opened.set()
            return
        if message["type"] != "http.response.body":
            return
        if wedge:
            await disconnected.wait()
            return
        await chunks.put(message.get("body", b""))  # type: ignore[arg-type]

    scope = {
        "type": "http",
        "asgi": {"spec_version": "2.3", "version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        # The MCP endpoint is mounted, so the unslashed form only redirects.
        "path": "/mcp/",
        "raw_path": b"/mcp/",
        "query_string": b"",
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in {**headers, "Accept": "text/event-stream"}.items()],
        "client": ("127.0.0.1", 32767),
        "server": ("test", 80),
    }
    stream = asyncio.ensure_future(app(scope, receive, send))
    try:
        # An announcement published before the transport registered this stream
        # would be dropped, so hand it over only once the response has begun.
        with anyio.fail_after(5):
            await opened.wait()
        yield chunks
    finally:
        disconnected.set()
        stream.cancel()
        with suppress(asyncio.CancelledError):
            await stream


async def _next_updated_uri(chunks: "asyncio.Queue[bytes]", timeout: float = 5.0) -> str:
    """The URI of the next ``notifications/resources/updated`` to arrive on ``chunks``."""
    buffered = ""
    with anyio.fail_after(timeout):
        while True:
            # SSE frames end in a blank CRLF line; normalise so one split works.
            buffered += (await chunks.get()).decode().replace("\r\n", "\n")
            *events, buffered = buffered.split("\n\n")
            for event in events:
                for line in event.splitlines():
                    if not line.startswith("data: "):
                        continue
                    message = json.loads(line.removeprefix("data: "))
                    if message.get("method") == "notifications/resources/updated":
                        return str(message["params"]["uri"])


async def _unsubscribe(client: httpx.AsyncClient, headers: dict[str, str], uri: str) -> dict:
    response = await client.post(
        "/mcp",
        headers=headers,
        json={"jsonrpc": "2.0", "id": "unsub", "method": "resources/unsubscribe", "params": {"uri": uri}},
    )
    return _reply(response)


class TestCapability:
    def test_not_advertised_without_a_notifier(self) -> None:
        server = MCPServer(_agent(), resources=[_resource()])

        assert _capabilities(server).resources.subscribe is False
        assert _capabilities(server, modern=True).resources.subscribe is False

    def test_advertised_in_both_eras(self) -> None:
        server = MCPServer(_agent(), resources=[_resource()], subscriptions=ResourceNotifier())

        assert _capabilities(server).resources.subscribe is True
        assert _capabilities(server, modern=True).resources.subscribe is True

    def test_stateless_keeps_only_the_modern_half(self) -> None:
        # Handshake delivery pushes into a session an earlier request opened, and
        # a stateless transport keeps none: advertising it would be a promise the
        # server cannot keep, so handshake-era clients are told to poll. The
        # modern era's listen stream is carved out of stateless mode by the SDK
        # and still streams, so it goes on advertising the capability.
        with pytest.warns(RuntimeWarning):
            server = MCPServer(_agent(), resources=[_resource()], subscriptions=ResourceNotifier(), stateless=True)

        assert _capabilities(server).resources.subscribe is False
        assert _capabilities(server, modern=True).resources.subscribe is True

    def test_list_changed_is_false_in_both_eras_without_a_notifier(self) -> None:
        server = MCPServer(_agent(), resources=[_resource()], prompts=[_prompt()])

        for caps in (_capabilities(server), _capabilities(server, modern=True)):
            assert caps.resources.list_changed is False
            assert caps.tools.list_changed is False
            assert caps.prompts.list_changed is False

    def test_list_changed_stays_false_in_the_handshake_era(self) -> None:
        # The resource, tool and prompt sets are all fixed at construction, so
        # there is no list change to announce — and here the flags say so.
        server = MCPServer(_agent(), resources=[_resource()], prompts=[_prompt()], subscriptions=ResourceNotifier())

        caps = _capabilities(server)
        assert caps.resources.list_changed is False
        assert caps.tools.list_changed is False
        assert caps.prompts.list_changed is False

    def test_list_changed_rides_the_listen_handler_in_the_modern_era(self) -> None:
        # Accepted imprecision, pinned so a change in it is caught. At 2026-07-28
        # every change notification rides the one subscription stream, so the SDK
        # derives `resources.subscribe` *and* all three list-changed flags from
        # the single fact that the listen handler is registered. The sets are
        # still fixed at construction, so these three promise events that will
        # never fire. Correcting them would mean subclassing the SDK's server to
        # change a flag that changes no behaviour, so it is documented instead.
        server = MCPServer(_agent(), resources=[_resource()], prompts=[_prompt()], subscriptions=ResourceNotifier())

        caps = _capabilities(server, modern=True)
        assert caps.resources.list_changed is True
        assert caps.tools.list_changed is True
        assert caps.prompts.list_changed is True


class TestStatelessWarning:
    def test_warns_on_the_intersection(self) -> None:
        with pytest.warns(RuntimeWarning, match="handshake-era clients") as caught:
            MCPServer(_agent(), resources=[_resource()], subscriptions=ResourceNotifier(), stateless=True)

        message = str(caught[0].message)
        assert "resources.subscribe" in message and "poll" in message
        # Attributed to the caller's own line rather than a frame inside
        # `ag2.mcp`, so `-W error::RuntimeWarning` points the author at the
        # `MCPServer(...)` they wrote.
        assert caught[0].filename == __file__

    def test_silent_for_a_stateless_server_without_a_notifier(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            MCPServer(_agent(), resources=[_resource()], stateless=True)

    def test_silent_for_a_notifier_without_a_stateless_transport(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            MCPServer(_agent(), resources=[_resource()], subscriptions=ResourceNotifier())

    def test_silenceable_through_the_standard_filter(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            MCPServer(_agent(), resources=[_resource()], subscriptions=ResourceNotifier(), stateless=True)


class TestBoundDefaults:
    def test_default_matches_the_sdk_listen_handler(self) -> None:
        # The handshake registry and the modern listen streams should fall over
        # at the same scale, not at two. The constant is a copy — the SDK exports
        # it only as a parameter default — so pin it here.
        sdk_default = inspect.signature(ListenHandler.__init__).parameters["max_subscriptions"].default

        assert ResourceNotifier().max_subscribers == sdk_default

    def test_backlog_default_matches_the_sdk_listen_handler(self) -> None:
        sdk_default = inspect.signature(ListenHandler.__init__).parameters["max_buffered_events"].default

        assert ResourceNotifier().max_buffered_events == sdk_default

    @pytest.mark.parametrize("bound", [0, -5])
    def test_rejects_a_bound_that_admits_nobody(self, bound: int) -> None:
        # A server that advertises `subscribe: true` and then refuses every
        # subscription is the silent lie the stateless warning exists to avoid.
        with pytest.raises(ValueError, match="max_subscribers must be >= 1"):
            ResourceNotifier(max_subscribers=bound)

    @pytest.mark.parametrize("bound", [0, -5])
    def test_rejects_a_backlog_that_buffers_nothing(self, bound: int) -> None:
        # A zero-length buffer drops every subscriber on its first announcement,
        # which is the same silent lie one step later.
        with pytest.raises(ValueError, match="max_buffered_events must be >= 1"):
            ResourceNotifier(max_buffered_events=bound)


class TestNotifierWiring:
    def test_requires_resources(self) -> None:
        with pytest.raises(ValueError, match="resources"):
            MCPServer(_agent(), subscriptions=ResourceNotifier())

    def test_one_notifier_per_server(self) -> None:
        notifier = ResourceNotifier()
        MCPServer(_agent(), resources=[_resource()], subscriptions=notifier)

        with pytest.raises(ValueError, match="already attached"):
            MCPServer(_agent(), resources=[_resource()], subscriptions=notifier)

    def test_a_failed_construction_leaves_the_notifier_reusable(self) -> None:
        # Adoption is the last thing `__init__` does, so a check that fails
        # after it was reached does not brick the caller's notifier: a corrected
        # retry with the same one has to work.
        notifier = ResourceNotifier()
        with pytest.raises(ValueError, match="must match the MCP endpoint path"):
            MCPServer(
                _agent(),
                resources=[_resource()],
                subscriptions=notifier,
                path="/mcp",
                security=require(
                    oauth2_scheme(url="https://auth.test"),
                    resource_url="http://test/elsewhere",
                    verifier=_Verifier(),
                ),
            )

        server = MCPServer(_agent(), resources=[_resource()], subscriptions=notifier)

        assert server.subscriptions is notifier

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

    async def test_unattached_notifier_refuses_to_publish(self) -> None:
        # No resource set to check the URI against and nobody subscribed to hear
        # it: publishing would be the silent no-op the strictness rules out.
        with pytest.raises(ValueError, match="not attached to an MCPServer"):
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
            resources=[_resource(), _other()],
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
            resources=[_resource(), _other()],
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

    async def test_unsubscribing_without_a_subscription_is_accepted(self) -> None:
        # Nothing to forget, and no way for the connection to know that: the
        # registry drops entries on its own (a wedged subscriber, a serving that
        # ended), so a client's unsubscribe has to be a no-op rather than an error.
        server = MCPServer(_agent(), resources=[_resource()], subscriptions=ResourceNotifier())

        async with connect(server) as session:
            await session.unsubscribe_resource("mem://counter")

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

        assert refused == IsPartialDict({"error": IsPartialDict({"message": "Subscription limit reached"})})

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

    async def test_refuses_uris_past_the_bound_on_one_connection(self) -> None:
        # Subscribing to a URI nobody serves is accepted, which is exactly what
        # makes this bound necessary: one connection could otherwise invent them
        # without end and never spend more than its single subscriber slot.
        app = MCPServer(
            _agent(),
            resources=[_resource()],
            subscriptions=ResourceNotifier(max_subscribers=2),
            json_response=True,
        )

        async with serve(app) as client:
            headers = await _handshake_session(client)
            assert "error" not in await _subscribe(client, headers, "mem://counter")
            assert "error" not in await _subscribe(client, headers, "mem://invented-1")
            refused = await _subscribe(client, headers, "mem://invented-2")

        assert refused == IsPartialDict({
            "error": IsPartialDict({"message": "Subscription limit reached for this connection"})
        })

    async def test_releasing_a_uri_frees_room_on_the_same_connection(self) -> None:
        app = MCPServer(
            _agent(),
            resources=[_resource(), _other()],
            subscriptions=ResourceNotifier(max_subscribers=2),
            json_response=True,
        )

        async with serve(app) as client:
            headers = await _handshake_session(client)
            await _subscribe(client, headers, "mem://counter")
            await _subscribe(client, headers, "mem://other")
            await _unsubscribe(client, headers, "mem://other")

            reused = await _subscribe(client, headers, "mem://third")

        assert "error" not in reused

    async def test_resubscribing_to_a_held_uri_is_not_refused_at_the_bound(self) -> None:
        # Re-sending a subscription the connection already holds adds no URI, so
        # it cannot be what tips the registry over.
        app = MCPServer(
            _agent(),
            resources=[_resource(), _other()],
            subscriptions=ResourceNotifier(max_subscribers=2),
            json_response=True,
        )

        async with serve(app) as client:
            headers = await _handshake_session(client)
            await _subscribe(client, headers, "mem://counter")
            await _subscribe(client, headers, "mem://other")

            again = await _subscribe(client, headers, "mem://counter")

        assert "error" not in again

    async def test_a_refused_uri_leaves_the_held_ones_working(self) -> None:
        server = MCPServer(
            _agent(),
            resources=[_resource(), _other()],
            subscriptions=ResourceNotifier(max_subscribers=2),
        )
        updated: list[str] = []
        arrived = anyio.Event()

        async with connect(server, message_handler=_collect_updates(updated, arrived)) as session:
            await session.subscribe_resource("mem://counter")
            await session.subscribe_resource("mem://other")
            with pytest.raises(MCPError, match="limit reached for this connection"):
                await session.subscribe_resource("mem://invented")

            # Refusal is not eviction: what the connection already holds still fires.
            await server.notify_resource_updated("mem://counter")
            with anyio.fail_after(5):
                await arrived.wait()

        assert updated == ["mem://counter"]


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
            resources=[_resource(), _other()],
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
        # era nothing — the warning informs the author rather than blocking them.
        with pytest.warns(RuntimeWarning):
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


@pytest.mark.asyncio
class TestHandshakeDeliveryOverHTTP:
    """The seam that exercises connection identity, the registry and delivery together.

    The in-memory client session files every subscription under one constant
    key, so only the HTTP transport can show that a subscription belongs to a
    *connection* — which is the whole reason the registry is keyed the way it is.
    """

    async def test_announcement_reaches_the_subscribing_connection_only(self) -> None:
        app = MCPServer(_agent(), resources=[_resource(), _other()], subscriptions=ResourceNotifier())

        async with serve(app) as client:
            subscriber = await _handshake_session(client)
            bystander = await _handshake_session(client)

            async with (
                _standalone_stream(app, subscriber) as subscriber_events,
                _standalone_stream(app, bystander) as bystander_events,
            ):
                assert "error" not in await _subscribe(client, subscriber, "mem://counter")
                await app.notify_resource_updated("mem://counter")

                assert await _next_updated_uri(subscriber_events) == "mem://counter"
                # The other connection never asked. The opt-in contract is
                # per connection, not per server.
                with pytest.raises(TimeoutError):
                    await _next_updated_uri(bystander_events, timeout=0.3)

    async def test_a_wedged_subscriber_is_dropped_and_its_peers_keep_receiving(self) -> None:
        # A client that stopped reading must cost its own subscription and
        # nothing else: not the publisher, and not the other subscribers.
        backlog = 8
        app = MCPServer(
            _agent(),
            resources=[_resource(), _other()],
            subscriptions=ResourceNotifier(max_subscribers=2, max_buffered_events=backlog),
        )

        async with serve(app) as client:
            wedged = await _handshake_session(client)
            healthy = await _handshake_session(client)
            latecomer = await _handshake_session(client)

            async with (
                _standalone_stream(app, wedged, wedge=True),
                _standalone_stream(app, healthy) as healthy_events,
            ):
                assert "error" not in await _subscribe(client, wedged, "mem://counter")
                assert "error" not in await _subscribe(client, healthy, "mem://other")
                refused = await _subscribe(client, latecomer, "mem://counter")
                assert refused == IsPartialDict({"error": IsPartialDict({"message": "Subscription limit reached"})})

                # Past its own backlog the wedged subscriber has nothing left
                # worth keeping — the protocol offers no replay.
                for _ in range(backlog + 64):
                    await app.notify_resource_updated("mem://counter")

                await app.notify_resource_updated("mem://other")
                assert await _next_updated_uri(healthy_events) == "mem://other"

                # Dropped, not served forever: the slot it held comes free.
                assert "error" not in await _subscribe(client, latecomer, "mem://counter")
