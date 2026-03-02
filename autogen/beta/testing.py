# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from autogen.beta import Context
from autogen.beta.config import LLMClient
from autogen.beta.events import BaseEvent, ModelResponse, ToolError


class TestConfig(LLMClient):
    __test__ = False

    def __init__(self, *events: ModelResponse) -> None:
        self.events = events

    def create(self) -> "TestConfig":
        return TestClient(*self.events)


class TestClient(LLMClient):
    __test__ = False

    def __init__(self, *events: ModelResponse) -> None:
        self.events = iter(events)

    async def __call__(self, *messages: BaseEvent, ctx: Context, **kwargs: Any) -> None:
        for m in messages:
            if isinstance(m, ToolError):
                raise m.error

        next_msg = next(self.events)
        await ctx.send(next_msg)
