# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
from a2a.types import AgentCapabilities, AgentCard, AgentInterface
from a2a.utils.constants import TransportProtocol

from autogen.beta import Agent
from autogen.beta.a2a import A2AConfig
from autogen.beta.a2a.client import A2AClient
from autogen.beta.a2a.errors import A2AResponseSchemaNotSupportedError
from autogen.beta.response.proto import ResponseProto


def _card(url: str = "http://example") -> AgentCard:
    return AgentCard(
        name="remote",
        description="d",
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
        supported_interfaces=[AgentInterface(url=url, protocol_binding=TransportProtocol.JSONRPC.value)],
    )


class TestCopy:
    def test_copy_returns_equal_instance(self) -> None:
        config = A2AConfig("http://localhost:8000", max_reconnects=5)

        copied = config.copy()

        assert copied == config
        assert copied is not config

    def test_copy_applies_overrides(self) -> None:
        config = A2AConfig("http://localhost:8000", max_reconnects=3)

        copied = config.copy(max_reconnects=10, reconnect_backoff=1.5)

        assert copied.max_reconnects == 10
        assert copied.reconnect_backoff == 1.5
        assert config.max_reconnects == 3
        assert config.reconnect_backoff == 0.5


class TestCreate:
    def test_create_returns_a2a_client(self) -> None:
        config = A2AConfig("http://localhost:8000")

        client = config.create()

        assert isinstance(client, A2AClient)

    def test_create_is_synchronous(self) -> None:
        config = A2AConfig("http://localhost:8000")

        assert not inspect.iscoroutine(config.create())


class TestFromCard:
    def test_uses_card_url(self) -> None:
        card = _card("http://prebuilt:9000")

        config = A2AConfig.from_card(card)

        assert config.url == "http://prebuilt:9000"
        assert config.agent_card is card

    def test_url_override_wins(self) -> None:
        card = _card("http://prebuilt:9000")

        config = A2AConfig.from_card(card, url="http://override:1234")

        assert config.url == "http://override:1234"
        assert config.agent_card is card

    def test_extra_overrides_propagate(self) -> None:
        card = _card("http://prebuilt:9000")

        config = A2AConfig.from_card(card, max_reconnects=7, reconnect_backoff=2.0)

        assert config.max_reconnects == 7
        assert config.reconnect_backoff == 2.0


class _PassthroughSchema(ResponseProto[str]):
    name = "s"
    description = None
    json_schema: dict[str, object] | None = None
    system_prompt = None

    async def validate(self, response, context, provider=None):  # type: ignore[no-untyped-def, override]
        return response


@pytest.mark.asyncio
async def test_a2a_client_rejects_response_schema() -> None:
    agent = Agent("client", config=A2AConfig("http://x"))

    with pytest.raises(A2AResponseSchemaNotSupportedError):
        await agent.ask("hi", response_schema=_PassthroughSchema())
