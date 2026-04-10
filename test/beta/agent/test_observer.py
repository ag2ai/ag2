# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, Context, Depends, Inject, Variable, observer
from autogen.beta.events import ModelRequest, ModelResponse, ToolCallEvent
from autogen.beta.testing import TestConfig


@pytest.fixture()
def test_config() -> TestConfig:
    return TestConfig("response")


@pytest.mark.asyncio()
async def test_observer_fires_on_matching_event(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    agent = Agent(
        "",
        config=test_config,
        observers=[observer(ModelResponse, mock)],
    )

    await agent.ask("Hi!")

    mock.assert_called_once()
    event = mock.call_args[0][0]
    assert isinstance(event, ModelResponse)


@pytest.mark.asyncio()
async def test_observer_does_not_fire_on_non_matching_event(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    agent = Agent(
        "",
        config=test_config,
        observers=[observer(ToolCallEvent, mock)],
    )

    await agent.ask("Hi!")

    mock.assert_not_called()


@pytest.mark.asyncio()
async def test_multiple_observers(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    agent = Agent(
        "",
        config=test_config,
        observers=[
            observer(ModelRequest, mock.request),
            observer(ModelResponse, mock.response),
        ],
    )

    await agent.ask("Hi!")

    mock.request.assert_called_once()
    mock.response.assert_called_once()


@pytest.mark.asyncio()
async def test_async_observer(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    @observer(ModelResponse)
    async def track_response(event: ModelResponse) -> None:
        mock(event)

    agent = Agent(
        "",
        config=test_config,
        observers=[track_response],
    )

    await agent.ask("Hi!")

    mock.assert_called_once()


@pytest.mark.asyncio()
async def test_decorator_style(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    agent = Agent(
        "",
        config=test_config,
    )

    @agent.observer(ModelResponse)
    def log_response(event: ModelResponse) -> None:
        mock(event)

    await agent.ask("Hi!")

    mock.assert_called_once()


@pytest.mark.asyncio()
async def test_per_call_observers(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    agent = Agent(
        "",
        config=test_config,
    )

    await agent.ask("Hi!", observers=[observer(ModelResponse, mock)])

    mock.assert_called_once()


@pytest.mark.asyncio()
async def test_constructor_and_ask_observers_both_fire(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    agent = Agent(
        "",
        config=test_config,
        observers=[observer(ModelResponse, mock.constructor)],
    )

    await agent.ask("Hi!", observers=[observer(ModelResponse, mock.ask)])

    mock.constructor.assert_called_once()
    mock.ask.assert_called_once()


@pytest.mark.asyncio()
async def test_observer_with_context_injection(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    @observer(ModelResponse)
    async def track_with_context(event: ModelResponse, ctx: Context) -> None:
        mock(event, ctx.stream.id)

    agent = Agent(
        "",
        config=test_config,
        observers=[track_with_context],
    )

    await agent.ask("Hi!")

    mock.assert_called_once()
    args = mock.call_args[0]
    assert isinstance(args[0], ModelResponse)
    assert args[1] is not None


@pytest.mark.asyncio()
async def test_observer_with_inject(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    @observer(ModelResponse)
    def track(event: ModelResponse, dep: Annotated[str, Inject()]) -> None:
        mock(dep)

    agent = Agent(
        "",
        config=test_config,
        observers=[track],
    )

    await agent.ask("Hi!", dependencies={"dep": "injected_value"})

    mock.assert_called_once_with("injected_value")


@pytest.mark.asyncio()
async def test_observer_with_depends(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def get_dep(dep: Annotated[str, Inject()]) -> str:
        return dep

    @observer(ModelResponse)
    def track(event: ModelResponse, dep: Annotated[str, Depends(get_dep)]) -> None:
        mock(dep)

    agent = Agent(
        "",
        config=test_config,
        observers=[track],
    )

    await agent.ask("Hi!", dependencies={"dep": "depends_value"})

    mock.assert_called_once_with("depends_value")


@pytest.mark.asyncio()
async def test_observer_with_variable(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    @observer(ModelResponse)
    def track(event: ModelResponse, val: Annotated[str, Variable("myvar")]) -> None:
        mock(val)

    agent = Agent(
        "",
        config=test_config,
        observers=[track],
    )

    await agent.ask("Hi!", variables={"myvar": "variable_value"})

    mock.assert_called_once_with("variable_value")
