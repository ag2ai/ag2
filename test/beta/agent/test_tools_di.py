# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from autogen.beta import Agent, Context, Depends, Inject, Toolkit
from autogen.beta.events import ToolCallEvent
from autogen.beta.testing import TestConfig


@pytest.fixture()
def test_config() -> TestConfig:
    return TestConfig(
        ToolCallEvent(name="my_tool"),
        "result",
    )


@pytest.mark.asyncio()
async def test_call_tool_with_injected_object(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(ctx: Context) -> str:
        mock(ctx.dependencies["dep"])
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    dependency = object()

    await agent.ask("Hi!", dependencies={"dep": dependency})

    mock.assert_called_once_with(dependency)


@pytest.mark.asyncio()
async def test_call_tool_with_agent_dependency(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(ctx: Context) -> str:
        mock(ctx.dependencies["dep"])
        return ""

    dependency = object()

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        dependencies={"dep": dependency},
    )

    await agent.ask("Hi!")

    mock.assert_called_once_with(dependency)


@pytest.mark.asyncio()
async def test_call_tool_with_mixed_dependencies(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(ctx: Context) -> str:
        mock(ctx.dependencies)
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        dependencies={"dep": "1"},
    )

    await agent.ask("Hi!", dependencies={"dep2": "2"})

    mock.assert_called_once_with({"dep": "1", "dep2": "2"})


@pytest.mark.asyncio()
async def test_inject_alias(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(
        dep: Annotated[str, Inject()],
    ) -> str:
        mock(dep)
        return ""

    dependency = object()

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        dependencies={"dep": dependency},
    )

    await agent.ask("Hi!")
    mock.assert_called_once_with(dependency)


@pytest.mark.asyncio()
async def test_inject_by_custom_name(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(
        d: Annotated[str, Inject("dep")],
    ) -> str:
        mock(d)
        return ""

    dependency = object()

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        dependencies={"dep": dependency},
    )

    await agent.ask("Hi!")
    mock.assert_called_once_with(dependency)


@pytest.mark.asyncio()
async def test_inject_with_default(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(
        dep: Annotated[str, Inject(default=1)],
    ) -> str:
        mock(dep)
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    await agent.ask("Hi!")
    mock.assert_called_once_with(1)


@pytest.mark.asyncio()
async def test_miss_injection(test_config: TestConfig) -> None:
    def my_tool(
        dep: Annotated[str, Inject()],
    ) -> str:
        return dep

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    with pytest.raises(ValidationError):
        await agent.ask("Hi!")


@pytest.mark.asyncio()
async def test_depends_override(mock: MagicMock, test_config: TestConfig) -> None:
    def dep1():
        raise ValueError

    def dep2():
        return "1"

    def my_tool(
        dep: Annotated[str, Depends(dep1)],
    ) -> str:
        mock(dep)
        return dep

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    agent.dependency_provider.override(dep1, dep2)

    await agent.ask("Hi")

    mock.assert_called_once_with("1")


@pytest.mark.asyncio()
async def test_depends_override_toolkit(mock: MagicMock, test_config: TestConfig) -> None:
    def dep1():
        raise ValueError

    def dep2():
        return "1"

    def my_tool(
        dep: Annotated[str, Depends(dep1)],
    ) -> str:
        mock(dep)
        return dep

    toolkit = Toolkit(my_tool)

    agent = Agent("", config=test_config)
    agent.add_tool(toolkit)

    agent.dependency_provider.override(dep1, dep2)

    await agent.ask("Hi")

    mock.assert_called_once_with("1")
