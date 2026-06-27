# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.events import ToolCallEvent, ToolResultEvent, ToolResultsEvent

from ._helpers import make_pair


def get_weather(city: str) -> str:
    return f"It is sunny in {city}"


@pytest.mark.asyncio
class TestE2EClientTools:
    async def test_round_trip_with_local_tool(self) -> None:
        pair = make_pair(
            ToolCallEvent(name="get_weather", arguments='{"city": "Paris"}'),
            after_tool="Weather report ready",
            client_tools=[get_weather],
            streaming=False,
        )

        reply = await pair.client.ask("how is paris?")

        assert reply.response.content == "Weather report ready"

    async def test_server_sees_tool_results_in_history_replay(self) -> None:
        tool_call = ToolCallEvent(name="get_weather", arguments='{"city": "Paris"}')
        pair = make_pair(
            tool_call,
            after_tool="all good",
            client_tools=[get_weather],
            streaming=False,
        )

        await pair.client.ask("paris?")

        pair.tracking.mock.assert_called_with(
            ToolResultsEvent([ToolResultEvent.from_call(tool_call, "It is sunny in Paris")]),
        )

    async def test_streaming_tool_round_trip(self) -> None:
        pair = make_pair(
            ToolCallEvent(name="get_weather", arguments='{"city": "Paris"}'),
            after_tool="Done",
            client_tools=[get_weather],
            streaming=True,
        )

        reply = await pair.client.ask("paris?")

        assert reply.response.content == "Done"
