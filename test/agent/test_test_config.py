# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from unittest.mock import MagicMock

import pytest

from ag2 import Agent, Context, MemoryStream, tool
from ag2.events import ModelRequest, ToolCallEvent
from ag2.exceptions import ConfigNotProvidedError, ToolNotFoundError
from ag2.testing import ModelCall, TestConfig, TrackingConfig
from ag2.tools.final import FunctionToolSchema


@pytest.fixture()
def test_config() -> TestConfig:
    return TestConfig(
        ToolCallEvent(name="my_tool"),
        "result",
    )


@pytest.mark.asyncio()
async def test_tool_raise_exc(test_config: TestConfig) -> None:
    def my_tool() -> str:
        raise ValueError

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    with pytest.raises(ValueError):
        await agent.ask("Hi!")


@pytest.mark.asyncio()
async def test_tool_not_found(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    agent = Agent("", config=test_config)

    with pytest.raises(ToolNotFoundError, match="Tool `my_tool` not found"):
        await agent.ask("Hi!")


@pytest.mark.asyncio()
async def test_ask_with_explicit_config_option(test_config: TestConfig) -> None:
    agent = Agent("")

    res = await agent.ask(
        "Hi!",
        config=TestConfig("result"),
    )

    assert res.body == "result"


@pytest.mark.asyncio()
async def test_ask_without_any_config() -> None:
    agent = Agent("")

    with pytest.raises(ConfigNotProvidedError):
        await agent.ask("Hi!")


@pytest.mark.asyncio()
@pytest.mark.parametrize("iterable_type", ["list", "tuple", "generator"])
async def test_tracking_config_preserves_tools_iterable(iterable_type: str) -> None:
    wrapped = TrackingConfig(TestConfig("result"))
    config = TrackingConfig(wrapped)
    agent = Agent("a", config=config)
    context = Context(stream=MemoryStream())
    schemas = await tool(lambda: "answer", name="answer").schemas(context)
    tools: Iterable[FunctionToolSchema]
    if iterable_type == "list":
        tools = schemas
    elif iterable_type == "tuple":
        tools = tuple(schemas)
    else:
        tools = (schema for schema in schemas)

    await config.create()(
        [ModelRequest.ensure_request(["go"])],
        context=context,
        tools=tools,
        response_schema=None,
        serializer=agent.serializer,
    )

    expected = [ModelCall(prompt=(), tools=tuple(schemas), dependencies={}, variables={})]
    assert config.calls == expected
    assert wrapped.calls == expected
