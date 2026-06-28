# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for issue #2904 — beta A2A must reject AgentCards whose
selected interface advertises an A2A protocol version < 1.0, while still
accepting interfaces that omit the optional ``protocol_version`` field."""

import pytest
from a2a.client.client_factory import TransportProtocol
from a2a.types import AgentCapabilities, AgentCard, AgentInterface
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT

from ag2 import Agent
from ag2.a2a import A2AConfig
from ag2.a2a.errors import A2AIncompatibleProtocolVersionError
from ag2.a2a.transports._http import select_interface, validate_protocol_version


def _iface(*, url: str, version: str, binding: str = TransportProtocol.JSONRPC.value) -> AgentInterface:
    return AgentInterface(url=url, protocol_binding=binding, protocol_version=version)


def _card(*interfaces: AgentInterface) -> AgentCard:
    return AgentCard(
        name="t",
        description="",
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[],
        supported_interfaces=list(interfaces),
    )


class TestValidateProtocolVersion:
    @pytest.mark.parametrize(
        "version",
        ["1.0", "1.0.0", "2.5", PROTOCOL_VERSION_CURRENT],
    )
    def test_accepts_compatible(self, version: str) -> None:
        iface = _iface(url="http://example", version=version)
        validate_protocol_version(iface, url="http://example", transport="jsonrpc")

    @pytest.mark.parametrize(
        "version",
        ["", "garbage", "not-a-version"],
    )
    def test_accepts_empty_or_unparsable(self, version: str) -> None:
        # The field is optional; the A2A SDK defaults a missing/unknown version
        # to the current one, so we must not reject these as incompatible.
        iface = _iface(url="http://example", version=version)
        validate_protocol_version(iface, url="http://example", transport="jsonrpc")

    @pytest.mark.parametrize("version", ["0.3", "0.9", "0.3.0"])
    def test_rejects_legacy(self, version: str) -> None:
        iface = _iface(url="http://example", version=version)
        with pytest.raises(A2AIncompatibleProtocolVersionError) as exc:
            validate_protocol_version(iface, url="http://example", transport="jsonrpc")
        assert exc.value.protocol_version == version
        assert exc.value.transport == "jsonrpc"
        assert exc.value.url == "http://example"


class TestSelectInterface:
    def test_prefer_selects_matching_binding(self) -> None:
        card = _card(
            _iface(url="http://jsonrpc", version="1.0"),
            _iface(url="http://grpc", version="1.0", binding=TransportProtocol.GRPC.value),
        )
        iface, transport = select_interface(card, url="http://jsonrpc", prefer="grpc")
        assert transport == "grpc"
        assert iface.url == "http://grpc"

    def test_url_match_picks_exact_interface_among_same_binding(self) -> None:
        # Two JSON-RPC interfaces: a legacy one and a current one. URL resolution
        # must pick the interface matching the connect URL, not the first by
        # binding — otherwise validation would inspect the wrong interface.
        legacy = _iface(url="http://legacy", version="0.3")
        current = _iface(url="http://current", version="1.0")
        card = _card(legacy, current)

        iface, transport = select_interface(card, url="http://current", prefer=None)
        assert iface.url == "http://current"
        # The selected (current) interface validates cleanly even though a
        # legacy interface is listed first.
        validate_protocol_version(iface, url="http://current", transport=transport)

    def test_url_match_to_legacy_interface_is_rejected(self) -> None:
        legacy = _iface(url="http://legacy", version="0.3")
        current = _iface(url="http://current", version="1.0")
        card = _card(legacy, current)

        iface, transport = select_interface(card, url="http://legacy", prefer=None)
        assert iface.url == "http://legacy"
        with pytest.raises(A2AIncompatibleProtocolVersionError):
            validate_protocol_version(iface, url="http://legacy", transport=transport)


@pytest.mark.asyncio
async def test_ask_raises_on_legacy_card() -> None:
    card = _card(_iface(url="http://legacy", version="0.3"))
    agent = Agent(
        "client-agent",
        config=A2AConfig(card_url="http://legacy", preset_card=card),
    )
    with pytest.raises(A2AIncompatibleProtocolVersionError) as exc:
        await agent.ask("hello")
    assert exc.value.protocol_version == "0.3"
