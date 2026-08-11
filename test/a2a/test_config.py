# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from a2a.client.client_factory import TransportProtocol
from a2a.types import AgentCapabilities, AgentCard, AgentInterface
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT

from ag2.a2a import A2AConfig
from ag2.a2a.errors import A2AInvalidCardError


def _card_with_interfaces(*urls: str) -> AgentCard:
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
                url=u,
                protocol_binding=TransportProtocol.JSONRPC.value,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            )
            for u in urls
        ],
    )


class TestFromCardUrlResolution:
    def test_picks_first_non_empty_interface_url(self) -> None:
        card = _card_with_interfaces("", "http://second.example")
        config = A2AConfig.from_card(card)
        assert config.card_url == "http://second.example"

    def test_picks_first_when_all_have_urls(self) -> None:
        card = _card_with_interfaces("http://a.example", "http://b.example")
        config = A2AConfig.from_card(card)
        assert config.card_url == "http://a.example"

    def test_explicit_url_overrides_card(self) -> None:
        card = _card_with_interfaces("http://from-card.example")
        config = A2AConfig.from_card(card, card_url="http://override.example")
        assert config.card_url == "http://override.example"

    def test_no_interfaces_and_no_override_raises(self) -> None:
        card = _card_with_interfaces()
        with pytest.raises(A2AInvalidCardError):
            A2AConfig.from_card(card)

    def test_all_empty_urls_and_no_override_raises(self) -> None:
        card = _card_with_interfaces("", "")
        with pytest.raises(A2AInvalidCardError):
            A2AConfig.from_card(card)
