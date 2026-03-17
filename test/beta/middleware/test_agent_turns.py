# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, Context
from autogen.beta.events import BaseEvent, ModelMessage, ModelRequest, ModelResponse
from autogen.beta.middleware import AgentTurn, BaseMiddleware, Middleware
from autogen.beta.testing import TestConfig, TrackingConfig


class MockMiddleware(BaseMiddleware):
    def __init__(
        self,
        event: BaseEvent,
        ctx: Context,
        mock: MagicMock,
        position: int = 0,
    ) -> None:
        super().__init__(event, ctx)
        mock.create(event)

        self.mock = mock
        self.position = position

    async def on_turn(
        self,
        call_next: AgentTurn,
        event: BaseEvent,
        ctx: Context,
    ) -> ModelResponse:
        self.mock.enter(self.position)
        response = await call_next(event, ctx)
        self.mock.exit(self.position)
        return response


@pytest.mark.asyncio()
async def test_middleware_creation(mock: MagicMock) -> None:
    agent = Agent(
        "",
        config=TestConfig("result"),
        middleware=[Middleware(MockMiddleware, mock=mock)],
    )

    await agent.ask("Hi!")

    mock.create.assert_called_once_with(ModelRequest(content="Hi!"))


@pytest.mark.asyncio()
async def test_middleware_agent_turn_chaining(mock: MagicMock) -> None:
    agent = Agent(
        "",
        config=TestConfig("result"),
        middleware=[Middleware(MockMiddleware, mock=mock, position=i) for i in range(1, 4)],
    )

    await agent.ask("Hi!")

    assert [c[0][0] for c in mock.enter.call_args_list] == [1, 2, 3]
    assert [c[0][0] for c in mock.exit.call_args_list] == [3, 2, 1]


@pytest.mark.asyncio()
async def test_middleware_incoming_message_mutation() -> None:
    tracking_config = TrackingConfig(TestConfig("2"))

    class MutatingMiddleware(BaseMiddleware):
        async def on_turn(
            self,
            call_next: AgentTurn,
            event: BaseEvent,
            ctx: Context,
        ) -> ModelResponse:
            if isinstance(event, ModelRequest):
                event = ModelRequest(content=event.content * 2)
            result = await call_next(event, ctx)
            return ModelResponse(message=ModelMessage(content=result.content * 2))

    agent = Agent(
        "",
        config=tracking_config,
        middleware=[MutatingMiddleware, MutatingMiddleware, MutatingMiddleware],
    )

    result = await agent.ask("1")

    tracking_config.mock.assert_called_once_with(ModelRequest(content="1" * (2**3)))
    assert result.content == "2" * (2**3)
