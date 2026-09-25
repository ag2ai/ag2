# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated, Any
from unittest.mock import MagicMock

import pytest

from ag2 import Agent, Context, Variable
from ag2.context import strip_reserved_variables
from ag2.events import ToolCallEvent
from ag2.middleware.builtin.tools.approval import BYPASS_KEY
from ag2.testing import TestConfig


@pytest.fixture()
def test_config() -> TestConfig:
    return TestConfig(
        ToolCallEvent(name="my_tool"),
        "result",
    )


@pytest.mark.asyncio()
async def test_ask_variables(
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


class TestReservedVariables:
    """Keys under ``ag:``/``a2a:`` are the framework's own; a peer never authors them."""

    def test_ordinary_keys_survive(self) -> None:
        payload = {"city": "Tokyo", "count": 3}

        assert strip_reserved_variables(payload, source="a test") == payload

    def test_reserved_keys_are_dropped(self) -> None:
        payload = {"city": "Tokyo", BYPASS_KEY: {"pay": True}, "a2a:tenant": "acme"}

        assert strip_reserved_variables(payload, source="a test") == {"city": "Tokyo"}

    def test_a_drop_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="ag2.context"):
            strip_reserved_variables({BYPASS_KEY: {"pay": True}}, source="a hostile peer")

        assert BYPASS_KEY in caplog.text
        assert "a hostile peer" in caplog.text

    def test_the_outbound_side_drops_quietly(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="ag2.context"):
            kept = strip_reserved_variables({BYPASS_KEY: {"pay": True}}, source="a response", warn=False)

        assert kept == {}
        assert caplog.text == ""

    def test_a_non_string_key_is_kept(self) -> None:
        payload: dict[Any, Any] = {7: "seven"}

        assert strip_reserved_variables(payload, source="a test") == payload
