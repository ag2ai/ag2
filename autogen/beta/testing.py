# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

from typing_extensions import Self

from autogen.beta import Context
from autogen.beta.config import LLMClient, ModelConfig
from autogen.beta.events import BaseEvent, ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent, ToolErrorEvent


class TestClient(LLMClient):
    __test__ = False

    def __init__(self, *events: ModelResponse) -> None:
        self.events = iter(events)

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        **kwargs: Any,
    ) -> ModelResponse:
        for m in messages:
            if isinstance(m, ToolErrorEvent):
                raise m.error

        next_msg = next(self.events)

        if isinstance(next_msg, str):
            next_msg = ModelResponse(ModelMessage(next_msg))
        elif isinstance(next_msg, ToolCallEvent):
            next_msg = ModelResponse(tool_calls=ToolCallsEvent([next_msg]))

        return next_msg


class TrackingClient(LLMClient):
    def __init__(self, client: LLMClient, mock: MagicMock) -> None:
        self.client = client
        self.mock = mock

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        **kwargs: Any,
    ) -> ModelResponse:
        self.mock(messages[-1])
        return await self.client(messages, context=context, **kwargs)


class TrackingConfig(ModelConfig):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.mock = MagicMock()

    def copy(self) -> Self:
        return self

    def create(self) -> TrackingClient:
        return TrackingClient(self.config.create(), self.mock)


class TestConfig(ModelConfig):
    __test__ = False

    def __init__(self, *events: ModelResponse | ToolCallEvent | str) -> None:
        self.events = events

    def copy(self) -> Self:
        return self

    def create(self) -> TestClient:
        return TestClient(*self.events)
