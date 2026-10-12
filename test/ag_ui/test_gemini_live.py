# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A Gemini 3 frontend tool survives an AG-UI history round trip."""

import os
from pathlib import Path

import pytest
from ag_ui.core import AssistantMessage, FunctionCall, ToolCall, ToolMessage, UserMessage
from dotenv import load_dotenv

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.config import GeminiConfig
from test.ag_ui.harness import dispatch_run, every, outcome_of, run_input, weather_tool

pytestmark = [pytest.mark.asyncio, pytest.mark.gemini]


async def test_frontend_tool_signature_round_trips() -> None:
    if os.getenv("AG2_LIVE_GEMINI") != "1":
        pytest.skip("set AG2_LIVE_GEMINI=1 to run the live Gemini check")
    load_dotenv(Path(__file__).parents[2] / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY is unavailable")

    agent = Agent("weather", config=GeminiConfig(model="gemini-3-flash-preview", api_key=api_key))
    stream = AGUIStream(agent)
    user = UserMessage(id="m1", content="What's the weather in Paris? Use get_weather.")
    first = await dispatch_run(stream, run_input(user, tools=[weather_tool()], thread_id="t1"))
    [call] = every(first, "TOOL_CALL_CHUNK")
    [signature] = every(first, "REASONING_ENCRYPTED_VALUE")
    assert signature["entityId"] == call["toolCallId"]

    assistant = AssistantMessage(
        id="a1",
        tool_calls=[
            ToolCall(
                id=call["toolCallId"],
                function=FunctionCall(name=call["toolCallName"], arguments=call.get("delta") or "{}"),
                encrypted_value=signature["encryptedValue"],
            )
        ],
    )
    result = ToolMessage(id="tm1", tool_call_id=call["toolCallId"], content="Sunny, 21C")
    second = await dispatch_run(stream, run_input(user, assistant, result, tools=[weather_tool()], thread_id="t1"))

    assert outcome_of(second)["type"] == "success"
    assert not every(second, "RUN_ERROR")
