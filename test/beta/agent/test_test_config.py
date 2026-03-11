# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent
from autogen.beta.events import ToolCall
from autogen.beta.exceptions import ConfigNotProvidedError, ToolNotFoundError
from autogen.beta.testing import TestConfig


@pytest.fixture()
def test_config() -> TestConfig:
    return TestConfig(
        ToolCall(name="my_tool"),
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

    assert res.content == "result"


@pytest.mark.asyncio()
async def test_ask_without_any_config() -> None:
    agent = Agent("")

    with pytest.raises(ConfigNotProvidedError):
        await agent.ask("Hi!")
