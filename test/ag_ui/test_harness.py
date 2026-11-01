# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The helpers every AG-UI test reads a run through."""

import pytest
from ag_ui.core import (
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageChunkEvent,
    UserMessage,
)

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.events import ModelRequest, ModelResponse, TextInput
from ag2.testing import TestConfig
from test.ag_ui.harness import dispatch_events, each, kinds_of, recording_history, run_input, sole

pytestmark = pytest.mark.asyncio


async def test_a_run_is_returned_as_typed_events_in_order() -> None:
    agent = Agent("test_agent", config=TestConfig("hello"))

    events = await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="hi")))

    assert kinds_of(events) == [RunStartedEvent, TextMessageChunkEvent, RunFinishedEvent]
    assert sole(events, TextMessageChunkEvent).delta == "hello"


async def test_each_is_empty_when_the_run_emitted_none_of_that_kind() -> None:
    agent = Agent("test_agent", config=TestConfig("hello"))

    events = await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="hi")))

    assert each(events, TextMessageChunkEvent) != []
    assert each([], TextMessageChunkEvent) == []


async def test_sole_refuses_a_run_with_two_of_a_kind() -> None:
    agent = Agent("test_agent", config=TestConfig("one", "two"))
    stream = AGUIStream(agent)
    first = await dispatch_events(stream, run_input(UserMessage(id="m1", content="a"), thread_id="t"))
    second = await dispatch_events(stream, run_input(UserMessage(id="m2", content="b"), thread_id="t"))

    with pytest.raises(ValueError):
        sole([*first, *second], RunStartedEvent)


async def test_recorded_history_holds_one_entry_per_model_call() -> None:
    agent = Agent("test_agent", config=TestConfig(ModelResponse(), "done"))
    middleware, calls = recording_history()

    await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="hi")), middleware=[middleware])

    [call] = calls
    assert [event for event in call.events if isinstance(event, ModelRequest)] == [ModelRequest([TextInput("hi")])]
