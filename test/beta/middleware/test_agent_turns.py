# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, Context
from autogen.beta.events import BaseEvent, ModelMessage, ModelRequest, ModelResponse, TextInput
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
class TestAgentTurnMiddleware:
    @pytest.mark.asyncio()
    async def test_creation(self, mock: MagicMock) -> None:
        agent = Agent(
            "",
            config=TestConfig("result"),
            middleware=[Middleware(MockMiddleware, mock=mock)],
        )

        await agent.ask("Hi!")

        mock.create.assert_called_once_with(ModelRequest([TextInput("Hi!")]))

    async def test_chaining(self, mock: MagicMock) -> None:
        agent = Agent(
            "",
            config=TestConfig("result"),
            middleware=[Middleware(MockMiddleware, mock=mock, position=i) for i in range(1, 4)],
        )

        await agent.ask("Hi!")

        assert [c[0][0] for c in mock.enter.call_args_list] == [1, 2, 3]
        assert [c[0][0] for c in mock.exit.call_args_list] == [3, 2, 1]

    async def test_incoming_message_mutation(self) -> None:
        tracking_config = TrackingConfig(TestConfig("2"))

        class MutatingMiddleware(BaseMiddleware):
            async def on_turn(
                self,
                call_next: AgentTurn,
                event: BaseEvent,
                ctx: Context,
            ) -> ModelResponse:
                if isinstance(event, ModelRequest) and isinstance(event.parts[0], TextInput):
                    event = ModelRequest([TextInput(event.parts[0].content * 2)])
                result = await call_next(event, ctx)
                return ModelResponse(ModelMessage(result.content * 2))

        agent = Agent(
            "",
            config=tracking_config,
            middleware=[MutatingMiddleware, MutatingMiddleware, MutatingMiddleware],
        )

        result = await agent.ask("1")

        tracking_config.mock.assert_called_once_with(ModelRequest([TextInput("1" * (2**3))]))
        assert result.body == "2" * (2**3)


@pytest.mark.asyncio()
class TestInsertMiddleware:
    async def test_is_called(self, mock: MagicMock) -> None:
        agent = Agent("", config=TestConfig("result"))
        agent.insert_middleware(Middleware(MockMiddleware, mock=mock))

        await agent.ask("Hi!")

        mock.enter.assert_called_once_with(0)

    async def test_inserted_is_outer(self, mock: MagicMock) -> None:
        agent = Agent(
            "",
            config=TestConfig("result"),
            middleware=[Middleware(MockMiddleware, mock=mock, position=2)],
        )
        agent.insert_middleware(Middleware(MockMiddleware, mock=mock, position=1))

        await agent.ask("Hi!")

        assert [c[0][0] for c in mock.enter.call_args_list] == [1, 2]
        assert [c[0][0] for c in mock.exit.call_args_list] == [2, 1]


@pytest.mark.asyncio()
class TestAddMiddleware:
    async def test_is_called(self, mock: MagicMock) -> None:
        agent = Agent("", config=TestConfig("result"))
        agent.add_middleware(Middleware(MockMiddleware, mock=mock))

        await agent.ask("Hi!")

        mock.enter.assert_called_once_with(0)

    async def test_added_is_inner(self, mock: MagicMock) -> None:
        agent = Agent(
            "",
            config=TestConfig("result"),
            middleware=[Middleware(MockMiddleware, mock=mock, position=1)],
        )
        agent.add_middleware(Middleware(MockMiddleware, mock=mock, position=2))

        await agent.ask("Hi!")

        assert [c[0][0] for c in mock.enter.call_args_list] == [1, 2]
        assert [c[0][0] for c in mock.exit.call_args_list] == [2, 1]
