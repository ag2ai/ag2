# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

from ag2 import Agent, Inject, TextInput
from ag2.events import BaseEvent, Input, ToolCallEvent
from ag2.middleware import statusbar_middleware
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.final import FunctionDefinition, FunctionToolSchema


@dataclass
class ToolStub:
    name: str
    schema: FunctionToolSchema


def make_parameterless_tool() -> ToolStub:
    return ToolStub(
        name="ask_human",
        schema=FunctionToolSchema(
            function=FunctionDefinition(
                name="ask_human",
                description="Ask the human for input.",
                parameters={"type": "null"},
            )
        ),
    )


def make_tool() -> ToolStub:
    return ToolStub(
        name="search_docs",
        schema=FunctionToolSchema(
            function=FunctionDefinition(
                name="search_docs",
                description="Search documentation by query.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                    "required": ["query"],
                },
            )
        ),
    )


def supplied_parts(parts: Annotated[Sequence[Input], Inject()]) -> Sequence[Input]:
    return parts


def echo() -> str:
    return "tool output"


async def status_model_inputs(*parts: Input, persist: bool = False) -> tuple[BaseEvent, ...]:
    """Model input of the call after one tool round, with status supplied as ``parts``."""
    config = TrackingConfig(TestConfig(ToolCallEvent("echo", id="call-1"), "done"))
    agent = Agent(
        "status", config=config, middleware=[statusbar_middleware(supplied_parts, persist=persist)], tools=[echo]
    )
    await agent.ask(TextInput("start"), dependencies={"parts": parts})
    return config.calls[-1].messages
