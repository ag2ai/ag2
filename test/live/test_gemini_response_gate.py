# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("google.genai")

from google.genai import types as gtypes

from ag2.events import ModelRequest, TextInput
from test.live._gemini_helpers import live_agent, tool_call, turn_complete


def lookup() -> str:
    """Look the answer up."""
    return "42"


@pytest.mark.asyncio
class TestResponseGate:
    async def test_tool_result_is_sent_at_once_and_requests_no_response(self) -> None:
        agent, session = live_agent(lookup)

        async with agent.run():
            await session.emit(tool_call("call-1"))

            assert session.tool_responses() == [
                gtypes.FunctionResponse(id="call-1", name="lookup", response={"result": "42"}),
            ]

            await session.emit(turn_complete())

            assert session.response_requests() == 0

    async def test_tool_call_starts_a_response(self) -> None:
        agent, session = live_agent()

        async with agent.run() as context:
            await session.emit(tool_call("call-1", name="unknown"))
            await context.send(ModelRequest([TextInput("hello")]))

            assert session.response_requests() == 0
