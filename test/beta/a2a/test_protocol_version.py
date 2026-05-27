# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for issue #2904 — beta A2A must reject AgentCards whose
selected interface advertises an A2A protocol version < 1.0."""

import pytest
from a2a.client.client_factory import TransportProtocol
from a2a.types import AgentCapabilities, AgentCard, AgentInterface

from autogen.beta.a2a.errors import A2AIncompatibleProtocolVersionError
from autogen.beta.a2a.transports._http import (
    _parse_protocol_version,
    validate_selected_protocol_version,
)


def _card(*, protocol_version: str, binding: str = TransportProtocol.JSONRPC.value) -> AgentCard:
    return AgentCard(
        name="t",
        description="",
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[],
        supported_interfaces=[
            AgentInterface(
                url="http://example",
                protocol_binding=binding,
                protocol_version=protocol_version,
            ),
        ],
    )


class TestParseProtocolVersion:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1.0", (1, 0)),
            ("1.0.0", (1, 0, 0)),
            ("0.3", (0, 3)),
            ("2", (2,)),
            ("", None),
            ("abc", None),
            ("1.x", None),
        ],
    )
    def test_parse(self, raw: str, expected: tuple[int, ...] | None) -> None:
        assert _parse_protocol_version(raw) == expected


class TestValidateSelectedProtocolVersion:
    def test_accepts_1_0(self) -> None:
        card = _card(protocol_version="1.0")
        validate_selected_protocol_version(card, url="http://example", transport="jsonrpc")

    def test_accepts_1_0_0(self) -> None:
        card = _card(protocol_version="1.0.0")
        validate_selected_protocol_version(card, url="http://example", transport="jsonrpc")

    def test_accepts_future_2_x(self) -> None:
        card = _card(protocol_version="2.5")
        validate_selected_protocol_version(card, url="http://example", transport="jsonrpc")

    def test_rejects_0_3(self) -> None:
        card = _card(protocol_version="0.3")
        with pytest.raises(A2AIncompatibleProtocolVersionError) as exc:
            validate_selected_protocol_version(card, url="http://example", transport="jsonrpc")
        assert exc.value.protocol_version == "0.3"
        assert exc.value.transport == "jsonrpc"
        assert exc.value.url == "http://example"

    def test_rejects_unset(self) -> None:
        card = _card(protocol_version="")
        with pytest.raises(A2AIncompatibleProtocolVersionError) as exc:
            validate_selected_protocol_version(card, url="http://example", transport="jsonrpc")
        assert exc.value.protocol_version == "<unset>"

    def test_rejects_unparseable(self) -> None:
        card = _card(protocol_version="not-a-version")
        with pytest.raises(A2AIncompatibleProtocolVersionError):
            validate_selected_protocol_version(card, url="http://example", transport="jsonrpc")

    def test_only_inspects_selected_transport(self) -> None:
        # JSON-RPC interface is on the current protocol; gRPC interface is on a
        # 0.x draft. When we select the JSON-RPC transport the gRPC version is
        # not our problem — connecting via JSON-RPC must succeed.
        card = AgentCard(
            name="t",
            description="",
            version="1.0.0",
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            capabilities=AgentCapabilities(),
            skills=[],
            supported_interfaces=[
                AgentInterface(
                    url="http://jsonrpc",
                    protocol_binding=TransportProtocol.JSONRPC.value,
                    protocol_version="1.0",
                ),
                AgentInterface(
                    url="http://grpc",
                    protocol_binding=TransportProtocol.GRPC.value,
                    protocol_version="0.9",
                ),
            ],
        )
        validate_selected_protocol_version(card, url="http://jsonrpc", transport="jsonrpc")
        with pytest.raises(A2AIncompatibleProtocolVersionError):
            validate_selected_protocol_version(card, url="http://grpc", transport="grpc")
