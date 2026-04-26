# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
from a2a.types import AgentCapabilities, AgentCard

from autogen.beta import MemoryStream
from autogen.beta.a2a import A2AClientToolsNotSupportedError, A2AConfig
from autogen.beta.a2a.a2a_client import A2AClient
from autogen.beta.context import ConversationContext
from autogen.beta.events import ModelRequest, TextInput
from autogen.beta.response.proto import ResponseProto
from autogen.beta.tools.schemas import ToolSchema


def _card(url: str = "http://example") -> AgentCard:
    return AgentCard(
        name="remote",
        description="d",
        url=url,
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
    )


class TestCopy:
    def test_copy_returns_equal_instance(self) -> None:
        config = A2AConfig("http://localhost:8000", max_reconnects=5)

        copied = config.copy()

        assert copied == config
        assert copied is not config

    def test_copy_applies_overrides(self) -> None:
        config = A2AConfig("http://localhost:8000", max_reconnects=3)

        copied = config.copy(max_reconnects=10, polling_interval=1.5)

        assert copied.max_reconnects == 10
        assert copied.polling_interval == 1.5
        assert config.max_reconnects == 3
        assert config.polling_interval == 0.5


class TestCreate:
    def test_create_returns_a2a_client(self) -> None:
        config = A2AConfig("http://localhost:8000")

        client = config.create()

        assert isinstance(client, A2AClient)

    def test_create_is_synchronous(self) -> None:
        # Creation must not be a coroutine — the LLMClient protocol guarantees
        # a sync `create()`. Network I/O is deferred until the first __call__.
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

        config = A2AConfig.from_card(card, max_reconnects=7, polling_interval=2.0)

        assert config.max_reconnects == 7
        assert config.polling_interval == 2.0


class _PassthroughSchema(ResponseProto[str]):
    name = "s"
    description = None
    json_schema: dict[str, object] | None = None
    system_prompt = None

    async def validate(self, response, context, provider=None):  # type: ignore[no-untyped-def, override]
        return response


@pytest.mark.asyncio
class TestClientToolsValidation:
    async def test_raises_when_tools_passed(self) -> None:
        client = A2AConfig("http://x").create()
        ctx = ConversationContext(stream=MemoryStream())

        with pytest.raises(A2AClientToolsNotSupportedError):
            await client(
                [ModelRequest([TextInput("hi")])],
                ctx,
                tools=[ToolSchema(type="custom")],
                response_schema=None,
                serializer=None,  # type: ignore[arg-type]
            )

    async def test_raises_when_response_schema_passed(self) -> None:
        client = A2AConfig("http://x").create()
        ctx = ConversationContext(stream=MemoryStream())

        with pytest.raises(A2AClientToolsNotSupportedError):
            await client(
                [ModelRequest([TextInput("hi")])],
                ctx,
                tools=[],
                response_schema=_PassthroughSchema(),
                serializer=None,  # type: ignore[arg-type]
            )
