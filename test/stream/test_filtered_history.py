# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Agent, Context, MemoryStream
from ag2.context import Stream
from ag2.events import BaseEvent, ModelRequest, TextInput, ToolCallEvent
from ag2.testing import TestConfig, TrackingConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [1, 2])
async def test_filtered_stream_shares_conversation_history(depth: int) -> None:
    parent = MemoryStream()
    stream: Stream = parent
    for _ in range(depth):
        stream = stream.where(ToolCallEvent)
    context = Context(stream)
    tool_call = ToolCallEvent(name="lookup", arguments="{}")
    request = ModelRequest([TextInput("question")])

    await context.send(tool_call)
    await context.send(request)

    assert list(await stream.history.get_events()) == [tool_call, request]
    assert list(await parent.history.get_events()) == [tool_call, request]


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [1, 2])
async def test_agent_receives_current_question_through_filtered_stream(depth: int) -> None:
    parent = MemoryStream()
    earlier = ModelRequest([TextInput("earlier question")])
    await Context(parent).send(earlier)
    stream: Stream = parent
    for _ in range(depth):
        stream = stream.where(BaseEvent)
    config = TrackingConfig(TestConfig("answer"))
    agent = Agent("test", config=config)

    reply = await agent.ask("current question", stream=stream)

    current = ModelRequest([TextInput("current question")])
    assert reply.body == "answer"
    config.mock.assert_called_once_with(current)
    requests = [event for event in await parent.history.get_events() if isinstance(event, ModelRequest)]
    assert requests == [earlier, current]
