# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
import pytest
from a2a.extensions.common import HTTP_EXTENSION_HEADER
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCard, AgentExtension, Part, Task, TaskState, TaskStatus

from ag2 import Agent
from ag2.a2a import A2AConfig, A2AServer, build_card
from ag2.a2a.errors import A2AExtensionNotSupportedError
from ag2.a2a.extension import EXTENSION_URI
from ag2.a2a.mappers.messages import build_user_message
from ag2.a2a.testing import make_test_client_factory
from ag2.events import TextInput
from ag2.testing import TestConfig

CUSTOM_URI = "urn:example:custom:v1"
OTHER_URI = "urn:example:other:v1"
URL = "http://test"


def _agent() -> Agent:
    return Agent("ext-server", config=TestConfig("ok"))


def _server_with_card(*exts: AgentExtension) -> tuple[A2AServer, AgentCard]:
    """Build a server plus the card it serves, declaring ``exts`` on top of the AG2 native one."""
    agent = _agent()
    return A2AServer(agent), build_card(agent, url=URL, extensions=list(exts))


def _factory_for(app: object, *, timeout: float = 30.0) -> Callable[[], httpx.AsyncClient]:
    """``httpx_client_factory`` dispatching into an ASGI ``app`` in-process.

    ``make_test_client_factory`` builds the card itself, so it can't serve a
    card carrying custom extensions; this drives ``build_jsonrpc(card=...)``
    through ``httpx.ASGITransport`` directly instead.
    """
    transport = httpx.ASGITransport(app=app)

    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, base_url=URL, timeout=timeout)

    return factory


class _CaptureExtensions:
    """ASGI wrapper recording the extension-activation request header."""

    def __init__(self, app: object) -> None:
        self.app = app
        self.header_values: list[str | None] = []

    async def __call__(self, scope: dict, receive: object, send: object) -> None:
        if scope["type"] == "http":
            headers = {k.decode(): v.decode() for k, v in scope["headers"]}
            self.header_values.append(headers.get(HTTP_EXTENSION_HEADER.lower()))
        await self.app(scope, receive, send)  # type: ignore[operator]


@dataclass(slots=True)
class _Seen:
    """One server-side observation of an incoming request."""

    header_uris: set[str]
    message_uris: list[str]


@dataclass(slots=True)
class _RecordingExecutor(AgentExecutor):
    """Records both activation channels per request, then drives one HITL round-trip.

    First message gets an ``input_required`` prompt, the second completes the
    task — so the recording covers the continuation leg too, not just the
    opening turn.
    """

    seen: list[_Seen] = field(default_factory=list)

    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        msg = request_context.message
        assert msg is not None
        self.seen.append(
            _Seen(
                header_uris=set(request_context.requested_extensions),
                message_uris=list(msg.extensions),
            )
        )
        task_id = msg.task_id or "task-1"
        context_id = msg.context_id or "ctx-1"
        updater = TaskUpdater(event_queue, task_id, context_id)

        if request_context.current_task is None:
            await event_queue.enqueue_event(
                Task(id=task_id, context_id=context_id, status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED))
            )
            await updater.start_work()
            await updater.requires_input(message=updater.new_agent_message(parts=[Part(text="more?")]))
            return

        await updater.complete(message=updater.new_agent_message(parts=[Part(text="ok")]))

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def test_build_card_declares_user_extensions() -> None:
    card = build_card(
        _agent(),
        url="http://test",
        extensions=[AgentExtension(uri=CUSTOM_URI, description="custom", required=False)],
    )
    uris = [ext.uri for ext in card.capabilities.extensions]
    assert uris == [EXTENSION_URI, CUSTOM_URI]


def test_build_card_rejects_duplicate_uris() -> None:
    with pytest.raises(ValueError, match=CUSTOM_URI):
        build_card(
            _agent(),
            url="http://test",
            extensions=[
                AgentExtension(uri=CUSTOM_URI),
                AgentExtension(uri=CUSTOM_URI),
            ],
        )


def test_build_card_rejects_redeclared_client_tools() -> None:
    with pytest.raises(ValueError, match="urn:ag2:client-tools:v1"):
        build_card(_agent(), url="http://test", extensions=[AgentExtension(uri=EXTENSION_URI)])


@pytest.mark.asyncio
async def test_unknown_requested_extension_raises() -> None:
    server = A2AServer(_agent())
    config = A2AConfig(
        card_url=URL,
        httpx_client_factory=make_test_client_factory(server, url=URL),
        extensions=["urn:example:not-advertised:v1"],
    )
    client = Agent("client", config=config)

    with pytest.raises(A2AExtensionNotSupportedError, match="not-advertised"):
        await client.ask("hi")


@pytest.mark.asyncio
async def test_required_extension_not_activated_raises() -> None:
    server, card = _server_with_card(AgentExtension(uri=CUSTOM_URI, required=True))
    config = A2AConfig(
        card_url=URL,
        httpx_client_factory=_factory_for(server.build_jsonrpc(url=URL, card=card)),
    )
    client = Agent("client", config=config)

    with pytest.raises(A2AExtensionNotSupportedError, match=CUSTOM_URI):
        await client.ask("hi")


@pytest.mark.asyncio
async def test_required_extension_activated_connects() -> None:
    server, card = _server_with_card(AgentExtension(uri=CUSTOM_URI, required=True))
    config = A2AConfig(
        card_url=URL,
        httpx_client_factory=_factory_for(server.build_jsonrpc(url=URL, card=card)),
        extensions=[CUSTOM_URI],
    )
    client = Agent("client", config=config)

    reply = await client.ask("hi")

    assert reply.body == "ok"


@pytest.mark.asyncio
async def test_activated_uris_ride_header_and_message() -> None:
    server, card = _server_with_card(AgentExtension(uri=CUSTOM_URI, required=False))
    app = _CaptureExtensions(server.build_jsonrpc(url=URL, card=card))
    config = A2AConfig(
        card_url=URL,
        httpx_client_factory=_factory_for(app),
        extensions=[CUSTOM_URI],
    )
    client = Agent("client", config=config)

    reply = await client.ask("hi")

    assert reply.body == "ok"
    assert any(v and CUSTOM_URI in v for v in app.header_values), (
        f"activated URI must appear in the {HTTP_EXTENSION_HEADER} request header, got {app.header_values!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_server_resolves_activation_on_every_leg(streaming: bool) -> None:
    """The server sees the activation on both channels, opening turn and continuation alike."""
    executor = _RecordingExecutor()
    server_agent = Agent("server-stub", config=TestConfig("unused"))
    server = A2AServer(server_agent, executor=executor)
    card = build_card(server_agent, url=URL, extensions=[AgentExtension(uri=CUSTOM_URI)])

    async def hitl_hook() -> str:
        return "more input"

    client = Agent(
        "client",
        config=A2AConfig(
            card_url=URL,
            httpx_client_factory=_factory_for(server.build_jsonrpc(url=URL, card=card)),
            streaming=streaming,
            extensions=[CUSTOM_URI],
        ),
        hitl_hook=hitl_hook,
    )

    await client.ask("hi")

    assert len(executor.seen) == 2, f"expected an opening turn plus a continuation, got {executor.seen!r}"
    assert all(CUSTOM_URI in seen.header_uris for seen in executor.seen)
    assert all(CUSTOM_URI in seen.message_uris for seen in executor.seen)


@pytest.mark.asyncio
async def test_no_activation_sends_no_extension_header() -> None:
    server, card = _server_with_card(AgentExtension(uri=CUSTOM_URI, required=False))
    app = _CaptureExtensions(server.build_jsonrpc(url=URL, card=card))
    client = Agent("client", config=A2AConfig(card_url=URL, httpx_client_factory=_factory_for(app)))

    await client.ask("hi")

    assert app.header_values and not any(app.header_values)


def test_message_extensions_field_carries_activated_uri() -> None:
    msg = build_user_message([TextInput("hi")], extra_extensions=[CUSTOM_URI, OTHER_URI])

    assert list(msg.extensions) == [CUSTOM_URI, OTHER_URI]


def test_message_extensions_do_not_duplicate_native_uri() -> None:
    msg = build_user_message(
        [TextInput("hi")],
        advertise_extension=True,
        extra_extensions=[EXTENSION_URI, CUSTOM_URI],
    )

    assert list(msg.extensions) == [EXTENSION_URI, CUSTOM_URI]


@pytest.mark.asyncio
async def test_native_extension_required_needs_no_activation() -> None:
    """A card requiring ``urn:ag2:client-tools:v1`` connects without explicit activation."""
    agent = _agent()
    card = build_card(agent, url=URL)
    card.capabilities.extensions[0].required = True
    config = A2AConfig(
        card_url=URL,
        httpx_client_factory=_factory_for(A2AServer(agent).build_jsonrpc(url=URL, card=card)),
    )
    client = Agent("client", config=config)

    reply = await client.ask("hi")

    assert reply.body == "ok"
