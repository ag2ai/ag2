# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, Context
from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, TextInput
from autogen.beta.middleware import AgentTurn, BaseMiddleware, LLMCall, Middleware
from autogen.beta.testing import TestConfig, TrackingConfig


class MockMiddleware(BaseMiddleware):
    def __init__(
        self,
        event: BaseEvent,
        ctx: Context,
        mock: MagicMock,
    ) -> None:
        super().__init__(event, ctx)
        self.mock = mock

    async def on_llm_call(
        self,
        call_next: AgentTurn,
        events: Sequence[BaseEvent],
        ctx: Context,
    ) -> ModelResponse:
        self.mock.enter(events[0])
        response = await call_next(events, ctx)
        self.mock.exit()
        return response


class OrderingMiddleware(BaseMiddleware):
    def __init__(
        self,
        event: BaseEvent,
        ctx: Context,
        mock: MagicMock,
        position: int,
    ) -> None:
        super().__init__(event, ctx)
        self.mock = mock
        self.position = position

    async def on_llm_call(
        self,
        call_next: AgentTurn,
        events: Sequence[BaseEvent],
        ctx: Context,
    ) -> ModelResponse:
        self.mock.enter(self.position)
        response = await call_next(events, ctx)
        self.mock.exit(self.position)
        return response


class TestLLMCallMiddleware:
    @pytest.mark.asyncio()
    async def test_creation(self, mock: MagicMock) -> None:
        agent = Agent(
            "",
            config=TestConfig("result"),
            middleware=[Middleware(MockMiddleware, mock=mock)],
        )

        await agent.ask("Hi!")

        mock.enter.assert_called_once_with(ModelRequest([TextInput("Hi!")]))
        mock.exit.assert_called_once()

    @pytest.mark.asyncio()
    async def test_call_sequence(self, mock: MagicMock) -> None:
        agent = Agent(
            "",
            config=TestConfig("result"),
            middleware=[Middleware(OrderingMiddleware, mock=mock, position=i) for i in range(1, 4)],
        )

        await agent.ask("Hi!")

        assert [c.args[0] for c in mock.enter.call_args_list] == [1, 2, 3]
        assert [c.args[0] for c in mock.exit.call_args_list] == [3, 2, 1]

    @pytest.mark.asyncio()
    async def test_incoming_message_mutation(self) -> None:
        tracking_config = TrackingConfig(TestConfig("2"))

        class MutatingMiddleware(BaseMiddleware):
            async def on_llm_call(
                self,
                call_next: LLMCall,
                events: Sequence[BaseEvent],
                ctx: Context,
            ) -> ModelResponse:
                last = events[-1]
                if isinstance(last, ModelRequest) and isinstance(last.inputs[0], TextInput):
                    doubled = TextInput(last.inputs[0].content * 2)
                    events[-1] = ModelRequest([doubled])
                return await call_next(events, ctx)

        agent = Agent(
            "",
            config=tracking_config,
            middleware=[MutatingMiddleware, MutatingMiddleware, MutatingMiddleware],
        )

        await agent.ask("1")

        tracking_config.mock.assert_called_once_with(ModelRequest([TextInput("1" * (2**3))]))
