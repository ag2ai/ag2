# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Annotated

from ag2 import Agent, Inject, TextInput
from ag2.events import BaseEvent, Input, ToolCallEvent
from ag2.middleware import statusbar_middleware
from ag2.testing import TestConfig, TrackingConfig


@statusbar_middleware
def supplied_parts(parts: Annotated[Sequence[Input], Inject()]) -> Sequence[Input]:
    return parts


def echo() -> str:
    return "tool output"


async def status_model_inputs(*parts: Input) -> tuple[BaseEvent, ...]:
    config = TrackingConfig(TestConfig(ToolCallEvent("echo", id="call-1"), "done"))
    agent = Agent("status", config=config, middleware=[supplied_parts], tools=[echo])
    await agent.ask(TextInput("start"), dependencies={"parts": parts})
    return config.calls[-1].messages
