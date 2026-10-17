# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A2UI extension negotiation over a real A2A round-trip.

``test_extension.py`` covers ``try_activate_a2ui_extension`` against a
``_StubContext`` that is handed a pre-populated ``requested_extensions``
list. That pins the helper's *logic* but not the wiring underneath it —
nothing there shows that a real client can put a URI in that list, which
is the only way the helper ever fires in production.

The wiring spans three layers: the client's transport (the
``A2A-Extensions`` header, gRPC metadata on that binding), the A2A SDK's
server-side plumbing into ``ServerCallContext.requested_extensions``, and
``RequestContext`` delegating to it. These tests drive all three through
``A2UIAgentExecutor``, which calls the helper on every request
(``ag2/a2ui/a2a/executor.py``).

Client-side activation is expressed as a raw ``A2A-Extensions`` header
because that is the only route available today. ``A2AConfig.extensions``
(ag2ai/ag2#3116) is the supported successor and will slot in here without
changing a single assertion below — see ``_activating``.
"""

import httpx
import pytest
from a2a.server.agent_execution import AgentExecutor as A2AAgentExecutorBase
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import AgentCard

from ag2 import Agent
from ag2.a2a import A2AConfig, A2AServer
from ag2.a2a.extension import EXTENSION_URI
from ag2.a2ui._types import A2UIVersion
from ag2.a2ui.a2a import get_a2ui_agent_extension, get_activated_extensions
from ag2.a2ui.a2a.executor import A2UIAgentExecutor
from ag2.a2ui.constants import A2UI_EXTENSION_URI_BY_VERSION
from ag2.testing import TestConfig

URL = "http://test"
REPLY = "ok"
A2UI_URI = A2UI_EXTENSION_URI_BY_VERSION["v0.9"]
OTHER_URI = "urn:example:unrelated:v1"


class _Recorder(A2AAgentExecutorBase):
    """Wraps ``A2UIAgentExecutor`` and snapshots what each request negotiated.

    Wrapping rather than subclassing keeps the executor under test the
    real one — the activation call site stays
    ``A2UIAgentExecutor.execute``, not something this file arranges.
    The snapshot is taken *after* the inner executor runs so it observes
    whatever activation left behind.
    """

    def __init__(self, agent: Agent, *, protocol_version: A2UIVersion = "v0.9") -> None:
        self.inner = A2UIAgentExecutor(agent, protocol_version=protocol_version)
        self.requested: list[set[str]] = []
        self.activated: list[list[str]] = []

    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        await self.inner.execute(request_context, event_queue)
        self.requested.append(set(request_context.requested_extensions))
        self.activated.append(get_activated_extensions(request_context))

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        await self.inner.cancel(request_context, event_queue)


def _advertising(version: A2UIVersion = "v0.9"):  # type: ignore[no-untyped-def]
    """Card modifier declaring A2UI, as ``user-guide/a2ui/a2a.mdx`` documents."""

    async def modifier(card: AgentCard) -> AgentCard:
        card.capabilities.extensions.append(get_a2ui_agent_extension(version=version))
        return card

    return modifier


def _pair(
    *,
    activate: str | None = None,
    server_version: A2UIVersion = "v0.9",
) -> tuple[Agent, _Recorder]:
    """A client talking to an A2UI-advertising server, optionally activating ``activate``."""
    agent = Agent("ui-server", config=TestConfig(REPLY))
    recorder = _Recorder(agent, protocol_version=server_version)
    server = A2AServer(agent, executor=recorder, card_modifier=_advertising(server_version))
    app = server.build_jsonrpc(url=URL)

    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=URL,
            headers=_activating(activate),
        )

    client = Agent("client", config=A2AConfig(card_url=URL, httpx_client_factory=factory, streaming=False))
    return client, recorder


def _activating(uri: str | None) -> dict[str, str]:
    """Headers that activate ``uri`` on every request, or none at all.

    The single place this file knows *how* activation is expressed. When
    ``A2AConfig.extensions`` lands, this collapses into that field and the
    tests below stay as they are.
    """
    return {"A2A-Extensions": uri} if uri else {}


@pytest.mark.asyncio
class TestCardAdvertisesA2UI:
    async def test_the_client_can_discover_the_a2ui_extension_on_the_card(self) -> None:
        """A client has to see the URI before it can sensibly activate it."""
        agent = Agent("ui-server", config=TestConfig(REPLY))
        server = A2AServer(agent, executor=_Recorder(agent), card_modifier=_advertising())
        app = server.build_jsonrpc(url=URL)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=URL) as http:
            response = await http.get("/.well-known/agent-card.json")

        uris = [ext["uri"] for ext in response.json()["capabilities"]["extensions"]]
        assert uris == [EXTENSION_URI, A2UI_URI]


@pytest.mark.asyncio
class TestActivationReachesTheServer:
    """The gap the stub tests cannot cover: does a real client populate ``requested_extensions``?"""

    async def test_an_activating_client_is_seen_by_the_server(self) -> None:
        client, recorder = _pair(activate=A2UI_URI)

        reply = await client.ask("show me a form")

        assert reply.body == REPLY, "the turn has to actually complete"
        assert recorder.requested == [{A2UI_URI}]

    async def test_a_non_activating_client_leaves_it_empty(self) -> None:
        client, recorder = _pair()

        await client.ask("show me a form")

        assert recorder.requested == [set()]

    async def test_an_unrelated_uri_does_not_look_like_a2ui(self) -> None:
        client, recorder = _pair(activate=OTHER_URI)

        await client.ask("show me a form")

        assert recorder.requested == [{OTHER_URI}]

    @pytest.mark.parametrize("version", ["v0.9", "v0.9.1", "v1.0"])
    async def test_each_protocol_version_negotiates_its_own_uri(self, version: A2UIVersion) -> None:
        uri = A2UI_EXTENSION_URI_BY_VERSION[version]
        client, recorder = _pair(activate=uri, server_version=version)

        await client.ask("show me a form")

        assert recorder.requested == [{uri}]

    async def test_activation_holds_across_a_second_turn(self) -> None:
        """Activation is per-request, so a follow-up turn must re-send it."""
        client, recorder = _pair(activate=A2UI_URI)

        first = await client.ask("show me a form")
        await first.ask("and another")

        assert recorder.requested == [{A2UI_URI}, {A2UI_URI}]


@pytest.mark.asyncio
class TestActivationIsRecorded:
    """The helper's side effect, checked against a real ``RequestContext``.

    This is where the carrier bug lived: activation used to record into
    ``RequestContext.metadata``, a read-only property deriving a fresh
    dict from the request params on every access, so the write landed on
    a throwaway. Every assertion here passed under the old stub and
    failed in production.
    """

    async def test_the_activation_survives_the_request(self) -> None:
        client, recorder = _pair(activate=A2UI_URI)

        await client.ask("show me a form")

        assert recorder.activated == [[A2UI_URI]]

    async def test_nothing_is_recorded_without_activation(self) -> None:
        client, recorder = _pair()

        await client.ask("show me a form")

        assert recorder.activated == [[]]

    async def test_an_unrelated_uri_records_nothing(self) -> None:
        client, recorder = _pair(activate=OTHER_URI)

        await client.ask("show me a form")

        assert recorder.activated == [[]]

    async def test_each_turn_records_independently(self) -> None:
        """``state`` is per-call, so turn two starts from a clean slate."""
        client, recorder = _pair(activate=A2UI_URI)

        first = await client.ask("show me a form")
        await first.ask("and another")

        assert recorder.activated == [[A2UI_URI], [A2UI_URI]]

    async def test_a_version_mismatch_records_nothing(self) -> None:
        """Client activates v0.9, server negotiates v1.0 — no activation, no record."""
        client, recorder = _pair(activate=A2UI_EXTENSION_URI_BY_VERSION["v0.9"], server_version="v1.0")

        await client.ask("show me a form")

        assert recorder.activated == [[]]
