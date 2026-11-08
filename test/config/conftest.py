# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from ag2 import Agent, Context, MemoryStream, ToolResult, events, testing, tool
from ag2.events import BaseEvent
from ag2.tools.builtin.tool_search import ToolSearchTool
from ag2.tools.schemas import ToolSchema


@pytest.fixture
def context() -> Context:
    return Context(stream=MagicMock())


def _lookup(order_id: str) -> str:
    return order_id


@pytest_asyncio.fixture
async def tool_search_schema(context: Context) -> ToolSchema:
    [schema, *_] = await ToolSearchTool(_lookup).schemas(context)
    return schema


@tool
def finish() -> ToolResult:
    return ToolResult("sent", final=True)


@tool
def lookup() -> str:
    return "lookup result"


@pytest_asyncio.fixture
async def final_turn_history() -> list[BaseEvent]:
    """History after a ``final=True`` turn with a sibling call, then a plain turn.

    The model calls ``finish`` (``final=True``) and ``lookup`` in one batch, so
    the first turn ends on the forced reply ``"sent"``; the second turn answers
    ``"second reply"``. Each mapper must replay both tool results exactly once.
    """
    calls = [events.ToolCallEvent("finish", id="final-call"), events.ToolCallEvent("lookup", id="lookup-call")]
    stream = MemoryStream()
    agent = Agent("assistant", config=testing.TestConfig(calls, "second reply"), tools=[finish, lookup])

    reply = await agent.ask("first turn", stream=stream)
    await reply.ask("second turn")

    return list(await stream.history.get_events())
