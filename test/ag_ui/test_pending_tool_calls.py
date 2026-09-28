# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A run that stops on client tool calls names them on its success outcome."""

import pytest
from ag_ui.core import UserMessage

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.events import ToolCallEvent
from ag2.testing import TestConfig
from test.ag_ui.harness import dispatch_run, every, outcome_of, run_input, weather_tool

pytestmark = pytest.mark.asyncio


async def test_the_calls_left_for_the_client_are_named_in_call_order() -> None:
    agent = Agent(
        "test_agent",
        config=TestConfig([
            ToolCallEvent(name="get_weather", arguments='{"location":"Paris"}'),
            ToolCallEvent(name="get_weather", arguments='{"location":"London"}'),
        ]),
    )

    events = await dispatch_run(
        AGUIStream(agent), run_input(UserMessage(id="m1", content="Paris and London?"), tools=[weather_tool()])
    )

    paris, london = every(events, "TOOL_CALL_CHUNK")
    assert outcome_of(events) == {"type": "success", "pendingToolCallIds": [paris["toolCallId"], london["toolCallId"]]}


async def test_a_server_call_answered_in_the_run_is_not_pending() -> None:
    agent = Agent(
        "test_agent",
        config=TestConfig([
            ToolCallEvent(name="get_time"),
            ToolCallEvent(name="get_weather", arguments='{"location":"London"}'),
        ]),
    )

    @agent.tool
    def get_time() -> str:
        return "noon"

    events = await dispatch_run(
        AGUIStream(agent), run_input(UserMessage(id="m1", content="time and weather?"), tools=[weather_tool()])
    )

    [client_call] = every(events, "TOOL_CALL_CHUNK")
    assert outcome_of(events) == {"type": "success", "pendingToolCallIds": [client_call["toolCallId"]]}


async def test_a_run_with_no_client_calls_names_none() -> None:
    """An empty list would say nothing, so none is sent at all."""
    agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="get_time"), "it is noon"))

    @agent.tool
    def get_time() -> str:
        return "noon"

    events = await dispatch_run(AGUIStream(agent), run_input(UserMessage(id="m1", content="time?")))

    assert outcome_of(events) == {"type": "success"}
