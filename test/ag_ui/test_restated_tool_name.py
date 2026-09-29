# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Gemini is handed the name of a tool result an AG-UI client restated.

An AG-UI `ToolMessage` carries only the call id, and Gemini refuses a function
response with no name.
"""

import pytest

pytest.importorskip("google.genai")

from ag_ui.core import AssistantMessage, FunctionCall, ToolCall, ToolMessage, UserMessage  # noqa: E402
from fast_depends.use import SerializerCls  # noqa: E402

from ag2.ag_ui.stream import AGStreamInput, map_agui_messages_to_events  # noqa: E402
from ag2.config.gemini.mappers import convert_messages  # noqa: E402
from test.ag_ui.harness import run_input  # noqa: E402


def _command(*messages: object) -> AGStreamInput:
    return AGStreamInput(incoming=run_input(*messages), variables={})


def _restated_round_trip() -> AGStreamInput:
    return _command(
        UserMessage(id="u1", content="weather in Paris?"),
        AssistantMessage(
            id="a1",
            tool_calls=[ToolCall(id="c1", function=FunctionCall(name="get_weather", arguments='{"city": "Paris"}'))],
        ),
        ToolMessage(id="t1", tool_call_id="c1", content="sunny"),
    )


def test_gemini_receives_the_function_response_name() -> None:
    _, messages, _ = map_agui_messages_to_events(_restated_round_trip())

    contents = convert_messages(messages, SerializerCls)
    [response] = [p.function_response for c in contents for p in c.parts or () if p.function_response]
    assert response.name == "get_weather"
