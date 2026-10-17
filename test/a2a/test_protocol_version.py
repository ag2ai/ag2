# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for issue #2904 — AG2 A2A must reject AgentCards whose
selected interface advertises an A2A protocol version < 1.0, while still
accepting interfaces that omit the optional ``protocol_version`` field."""

import pytest
from a2a.client.client_factory import TransportProtocol
from a2a.types import AgentCard, AgentInterface
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT

from ag2 import Agent
from ag2.a2a import A2AConfig, A2AServer, build_card
from ag2.a2a.errors import A2AError, A2AIncompatibleProtocolVersionError
from ag2.a2a.testing import make_test_client_factory
from ag2.a2a.transports import TransportName
from ag2.testing import TestConfig

URL = "http://test"
LEGACY_URL = "http://legacy"
REPLY = "pong"


def _iface(*, url: str, version: str, binding: str = TransportProtocol.JSONRPC.value) -> AgentInterface:
    return AgentInterface(url=url, protocol_binding=binding, protocol_version=version)


def _card(*interfaces: AgentInterface) -> AgentCard:
    """A serviceable AG2 card whose ``supported_interfaces`` are overridden.

    ``AgentCard`` is a protobuf message, so a repeated field is rebuilt by
    clearing and extending — it cannot be assigned to.
    """
    card = build_card(Agent("server", config=TestConfig(REPLY)), url=URL)
    card.ClearField("supported_interfaces")
    card.supported_interfaces.extend(interfaces)
    return card


def _client(card: AgentCard, *, card_url: str = URL, prefer: TransportName | None = None) -> Agent:
    """Client that connects using ``card``, backed by an in-process server.

    ``preset_card`` is what puts ``card`` in front of the version gate; the
    factory is only there to carry the request once the gate lets it through.
    """
    server = A2AServer(Agent("server", config=TestConfig(REPLY)))
    return Agent(
        "client",
        config=A2AConfig(
            card_url=card_url,
            preset_card=card,
            prefer=prefer,
            httpx_client_factory=make_test_client_factory(server, url=URL),
        ),
    )


@pytest.mark.asyncio
class TestVersionGate:
    @pytest.mark.parametrize("version", ["1.0", "1.0.0", "2.5", PROTOCOL_VERSION_CURRENT])
    async def test_connects_on_a_compatible_version(self, version: str) -> None:
        reply = await _client(_card(_iface(url=URL, version=version))).ask("ping")

        assert reply.body == REPLY

    async def test_connects_when_the_version_is_absent(self) -> None:
        # ``protocol_version`` is optional and the A2A SDK defaults a missing
        # one to the current version, so an empty field is not legacy.
        reply = await _client(_card(_iface(url=URL, version=""))).ask("ping")

        assert reply.body == REPLY

    @pytest.mark.parametrize("version", ["garbage", "not-a-version"])
    async def test_an_unparsable_version_is_not_treated_as_legacy(self, version: str) -> None:
        # The #2904 guard is specifically that *AG2's* gate stays out of the
        # way here. Connecting still fails — the SDK refuses to build a
        # transport for a version it cannot parse — but that is the SDK's
        # own ValueError, not anything from AG2's A2A error hierarchy.
        client = _client(_card(_iface(url=URL, version=version)))

        with pytest.raises(ValueError) as exc_info:
            await client.ask("ping")

        assert not isinstance(exc_info.value, A2AError)

    @pytest.mark.parametrize("version", ["0.3", "0.9", "0.3.0"])
    async def test_refuses_a_legacy_version(self, version: str) -> None:
        client = _client(_card(_iface(url=URL, version=version)))

        with pytest.raises(A2AIncompatibleProtocolVersionError) as exc_info:
            await client.ask("ping")

        assert exc_info.value.protocol_version == version
        assert exc_info.value.transport == "jsonrpc"
        assert exc_info.value.url == URL


@pytest.mark.asyncio
class TestInterfaceSelection:
    async def test_connect_url_picks_its_own_interface_not_the_first_listed(self) -> None:
        # A legacy interface is listed first, but the connect URL names the
        # current one — resolution must follow the URL, or the version gate
        # would inspect the wrong interface and reject a valid server.
        card = _card(
            _iface(url=LEGACY_URL, version="0.3"),
            _iface(url=URL, version=PROTOCOL_VERSION_CURRENT),
        )

        reply = await _client(card).ask("ping")

        assert reply.body == REPLY

    async def test_connecting_to_the_legacy_interface_is_refused(self) -> None:
        card = _card(
            _iface(url=LEGACY_URL, version="0.3"),
            _iface(url=URL, version=PROTOCOL_VERSION_CURRENT),
        )

        client = _client(card, card_url=LEGACY_URL)

        with pytest.raises(A2AIncompatibleProtocolVersionError) as exc_info:
            await client.ask("ping")

        assert exc_info.value.url == LEGACY_URL

    async def test_prefer_overrides_the_url_match(self) -> None:
        # ``prefer`` must win over the URL match. The gRPC interface is the
        # legacy one, so a gRPC-tagged version error is proof that selection
        # went to gRPC rather than quietly staying on JSON-RPC.
        card = _card(
            _iface(url=URL, version=PROTOCOL_VERSION_CURRENT),
            _iface(url="grpc.example:50051", version="0.3", binding=TransportProtocol.GRPC.value),
        )

        client = _client(card, prefer="grpc")

        with pytest.raises(A2AIncompatibleProtocolVersionError) as exc_info:
            await client.ask("ping")

        assert exc_info.value.transport == "grpc"
        assert exc_info.value.protocol_version == "0.3"
