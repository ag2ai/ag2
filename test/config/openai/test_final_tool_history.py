# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls

from ag2 import Agent, MemoryStream, ToolResult, events, testing, tool
from ag2.config.openai.mappers import convert_messages, events_to_responses_input


@pytest.mark.asyncio
async def test_final_tool_history_replays_openai_results_before_reply() -> None:
    @tool
    def finish() -> ToolResult:
        return ToolResult("sent", final=True)

    @tool
    def lookup() -> str:
        return "lookup result"

    calls = [events.ToolCallEvent("finish", id="final-call"), events.ToolCallEvent("lookup", id="lookup-call")]
    config = testing.TrackingConfig(testing.TestConfig(calls, "second reply"))
    stream = MemoryStream()
    agent = Agent("assistant", config=config, tools=[finish, lookup])

    reply = await agent.ask("first turn", stream=stream)
    await reply.ask("second turn")
    history = list(await stream.history.get_events())

    assert config.mock.call_count == 2
    assert convert_messages([], history, SerializerCls) == [
        {"role": "system", "content": ""},
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": None, "tool_calls": [call.to_api() for call in calls]},
        {"role": "tool", "tool_call_id": "final-call", "content": "sent"},
        {"role": "tool", "tool_call_id": "lookup-call", "content": "lookup result"},
        {"role": "assistant", "content": "sent"},
        {"role": "user", "content": "second turn"},
        {"role": "assistant", "content": "second reply"},
    ]
    assert events_to_responses_input(history, SerializerCls) == [
        {"role": "user", "content": [{"type": "input_text", "text": "first turn"}]},
        {"type": "function_call", "call_id": "final-call", "name": "finish", "arguments": "{}"},
        {"type": "function_call", "call_id": "lookup-call", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "final-call", "output": "sent"},
        {"type": "function_call_output", "call_id": "lookup-call", "output": "lookup result"},
        {"role": "assistant", "content": "sent"},
        {"role": "user", "content": [{"type": "input_text", "text": "second turn"}]},
        {"role": "assistant", "content": "second reply"},
    ]
