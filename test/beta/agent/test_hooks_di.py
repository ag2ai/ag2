# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, Context, Depends, Inject, MemoryStream
from autogen.beta.events import ModelMessage, ModelResponse
from autogen.beta.testing import TestConfig


@pytest.fixture()
def test_config() -> TestConfig:
    return TestConfig(
        ModelResponse(
            message=ModelMessage(content="result"),
        ),
    )


@pytest.mark.asyncio()
async def test_sync_hook_subscriber(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def result_sub(
        event: ModelResponse,
        c: Context,
        dep: Annotated[str, Inject()],
    ) -> None:
        mock(c.dependencies["dep"] == dep == "1")
        mock.response(event.content)

    agent = Agent("", config=test_config)

    stream = MemoryStream()
    stream.where(ModelResponse).subscribe(result_sub)

    await agent.ask("Hi!", dependencies={"dep": "1"}, stream=stream)

    mock.assert_called_once_with(True)
    mock.response.assert_called_once_with("result")


@pytest.mark.asyncio()
async def test_async_hook_subscriber(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    async def result_sub(
        c: Context,
        dep: Annotated[str, Inject()],
    ) -> None:
        mock(c.dependencies["dep"] == dep == "1")

    agent = Agent("", config=test_config)

    stream = MemoryStream()
    stream.where(ModelResponse).subscribe(result_sub)

    await agent.ask("Hi!", dependencies={"dep": "1"}, stream=stream)

    mock.assert_called_once_with(True)


@pytest.mark.asyncio()
async def test_hook_with_depends(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def d(dep: Annotated[str, Inject()]) -> str:
        return dep

    def result_sub(
        c: Context,
        dep: Annotated[str, Depends(d)],
    ) -> None:
        mock(c.dependencies["dep"] == dep == "1")

    agent = Agent("", config=test_config)

    stream = MemoryStream()
    stream.where(ModelResponse).subscribe(result_sub, sync_to_thread=False)

    await agent.ask("Hi!", dependencies={"dep": "1"}, stream=stream)

    mock.assert_called_once_with(True)


@pytest.mark.asyncio()
async def test_hook_with_agent_dependency(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def result_sub(
        c: Context,
        dep: Annotated[str, Inject()],
    ) -> None:
        mock(c.dependencies["dep"] == dep == "1")

    agent = Agent("", config=test_config, dependencies={"dep": "1"})

    stream = MemoryStream()
    stream.where(ModelResponse).subscribe(result_sub)

    await agent.ask("Hi!", stream=stream)

    mock.assert_called_once_with(True)


@pytest.mark.asyncio()
async def test_hook_depends_override(mock: MagicMock, test_config: TestConfig) -> None:
    def dep1():
        raise ValueError

    def dep2():
        return "1"

    def result_sub(
        dep: Annotated[str, Depends(dep1)],
    ) -> None:
        mock(dep)

    agent = Agent("", config=test_config)
    agent.dependency_provider.override(dep1, dep2)

    stream = MemoryStream()
    stream.where(ModelResponse).subscribe(result_sub, sync_to_thread=False)

    await agent.ask("Hi!", stream=stream)

    mock.assert_called_once_with("1")
