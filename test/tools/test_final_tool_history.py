# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Agent, DataInput, MemoryStream, TextInput, ToolResult, events, testing, tool
from ag2.history import MemoryStorage, close_unanswered_tool_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("persist_all", [False, True])
@pytest.mark.parametrize("position", ["only", "first", "last"])
@pytest.mark.parametrize("structured", [False, True])
async def test_final_tool_history_retains_completed_batch(persist_all: bool, position: str, structured: bool) -> None:
    result = ToolResult(DataInput({"status": "sent"}) if structured else TextInput("sent"), final=True)
    result.metadata["delivery"] = "complete"

    @tool
    def finish() -> ToolResult:
        return result

    @tool
    def lookup() -> str:
        return "lookup result"

    calls = [events.ToolCallEvent("finish", id="final-call")]
    expected_results = [events.ToolResultEvent.from_call(calls[0], result)]
    if position != "only":
        sibling = events.ToolCallEvent("lookup", id="lookup-call")
        index = 0 if position == "last" else 1
        calls.insert(index, sibling)
        expected_results.insert(index, events.ToolResultEvent.from_call(sibling, "lookup result"))

    storage = MemoryStorage()
    stream = MemoryStream(storage, persist_all=persist_all)
    config = testing.TrackingConfig(testing.TestConfig(calls, "second reply"))
    agent = Agent("assistant", config=config, tools=[finish, lookup])

    reply = await agent.ask("first turn", stream=stream)

    content = '{"status":"sent"}' if structured else "sent"
    assert reply.body == content
    assert config.mock.call_count == 1
    assert await stream.history.where(events.ModelResponse | events.ToolResultsEvent).all() == [
        events.ModelResponse(tool_calls=events.ToolCallsEvent(calls)),
        events.ToolResultsEvent(expected_results),
        events.ModelResponse(events.ModelMessage(content, metadata=result.metadata), response_force=True),
    ]

    await reply.ask("second turn")

    assert config.mock.call_count == 2
    config.mock.assert_called_with(events.ModelRequest([TextInput("second turn")]))
    # A fresh stream sharing the storage can replay the complete batch too.
    restored = MemoryStream(storage, id=stream.id)
    assert await restored.history.where(events.ToolResultsEvent).all() == [events.ToolResultsEvent(expected_results)]
    before_repair = list(await restored.history.get_events())
    assert await close_unanswered_tool_calls(restored.history, result="interrupted") == 0
    assert list(await restored.history.get_events()) == before_repair


@pytest.mark.asyncio
@pytest.mark.parametrize("final_first", [False, True])
async def test_final_tool_history_retains_sibling_errors(final_first: bool) -> None:
    @tool
    def finish() -> ToolResult:
        return ToolResult("sent", final=True)

    @tool
    def fail() -> str:
        raise ValueError("lookup failed")

    calls = [events.ToolCallEvent("finish"), events.ToolCallEvent("fail")]
    if not final_first:
        calls.reverse()
    stream = MemoryStream()
    config = testing.TrackingConfig(testing.TestConfig(calls, "second reply", raise_tool_errors=False))
    agent = Agent("assistant", config=config, tools=[finish, fail])

    reply = await agent.ask("first turn", stream=stream)
    await reply.ask("second turn")

    expected = [
        events.ToolResultEvent.from_call(call, ToolResult("sent", final=True))
        if call.name == "finish"
        else events.ToolErrorEvent.from_call(call, ValueError("lookup failed"))
        for call in calls
    ]
    assert config.mock.call_count == 2
    assert await stream.history.where(events.ToolResultsEvent).all() == [events.ToolResultsEvent(expected)]


@pytest.mark.asyncio
async def test_final_tool_history_preserves_first_final_reply() -> None:
    @tool
    def finish(message: str) -> ToolResult:
        return ToolResult(message, final=True)

    calls = [
        events.ToolCallEvent("finish", arguments='{"message":"first"}'),
        events.ToolCallEvent("finish", arguments='{"message":"second"}'),
    ]
    config = testing.TrackingConfig(testing.TestConfig(calls))
    stream = MemoryStream()
    agent = Agent("assistant", config=config, tools=[finish])

    reply = await agent.ask("first turn", stream=stream)

    assert reply.body == "first"
    assert config.mock.call_count == 1
    assert await stream.history.where(events.ToolResultsEvent).all() == [
        events.ToolResultsEvent([
            events.ToolResultEvent.from_call(calls[0], ToolResult("first", final=True)),
            events.ToolResultEvent.from_call(calls[1], ToolResult("second", final=True)),
        ])
    ]
