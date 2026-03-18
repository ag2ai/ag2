# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, Context, Variable
from autogen.beta.events import ToolCallEvent
from autogen.beta.testing import TestConfig


@pytest.fixture()
def test_config() -> TestConfig:
    return TestConfig(
        ToolCallEvent(name="my_tool"),
        "result",
    )


@pytest.mark.asyncio()
async def test_set_variables(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(ctx: Context) -> str:
        mock(ctx.variables["dep"])
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    await agent.ask("Hi!", variables={"dep": "1"})

    mock.assert_called_once_with("1")


@pytest.mark.asyncio()
async def test_agent_variables(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(ctx: Context) -> str:
        mock(ctx.variables["dep"])
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        variables={"dep": "1"},
    )

    await agent.ask("Hi!")

    mock.assert_called_once_with("1")


@pytest.mark.asyncio()
async def test_mixed_variables(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(ctx: Context) -> str:
        mock(ctx.variables)
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        variables={"dep": "1"},
    )

    await agent.ask("Hi!", variables={"dep2": "2"})

    mock.assert_called_once_with({"dep": "1", "dep2": "2"})


@pytest.mark.asyncio()
async def test_variable_alias(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(dep: Annotated[str, Variable()]) -> str:
        mock(dep)
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        variables={"dep": "1"},
    )

    await agent.ask("Hi!")
    mock.assert_called_once_with("1")


@pytest.mark.asyncio()
async def test_variable_by_name(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(d: Annotated[str, Variable("dep")]) -> str:
        mock(d)
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        variables={"dep": "1"},
    )

    await agent.ask("Hi!")
    mock.assert_called_once_with("1")


@pytest.mark.asyncio()
async def test_variable_with_default(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(dep: Annotated[str, Variable(default="1")]) -> str:
        mock(dep)
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    await agent.ask("Hi!")
    mock.assert_called_once_with("1")


@pytest.mark.asyncio()
async def test_variable_with_default_factory(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(dep: Annotated[str, Variable(default_factory=dict)]) -> str:
        mock(dep)
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    await agent.ask("Hi!")
    mock.assert_called_once_with({})


@pytest.mark.asyncio()
async def test_set_variable_by_tool(mock: MagicMock) -> None:
    def my_tool(ctx: Context) -> str:
        assert not ctx.variables
        ctx.variables["dep"] = "1"
        return ""

    def another_tool(ctx: Context) -> str:
        mock(ctx.variables["dep"])
        return ""

    agent = Agent(
        "",
        config=TestConfig(
            ToolCallEvent(name="my_tool"),
            ToolCallEvent(name="another_tool"),
            "result",
        ),
        tools=[my_tool, another_tool],
    )

    await agent.ask("Hi!")

    mock.assert_called_once_with("1")


@pytest.mark.asyncio()
async def test_variable_with_default_factory_called_once(mock: MagicMock) -> None:
    def factory() -> list[int]:
        mock.factory()
        return [1]

    def my_tool(
        dep: Annotated[list[int], Variable(default_factory=factory)],
    ) -> str:
        mock.first(dep.copy())
        dep.append(2)
        return ""

    def another_tool(
        dep: Annotated[list[int], Variable(default_factory=factory)],
    ) -> str:
        mock.second(dep.copy())
        return ""

    agent = Agent(
        "",
        config=TestConfig(
            ToolCallEvent(name="my_tool"),
            ToolCallEvent(name="another_tool"),
            "result",
        ),
        tools=[my_tool, another_tool],
    )

    await agent.ask("Hi!")

    mock.factory.assert_called_once()
    mock.first.assert_called_once_with([1])
    mock.second.assert_called_once_with([1, 2])
