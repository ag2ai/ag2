# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls
from mistralai.client.models import AssistantMessage, FunctionCall, ToolCall, ToolMessage, UserMessage

from ag2 import TextInput
from ag2.config.mistral.mappers import convert_messages
from test.config._helpers import status_model_inputs


@pytest.mark.asyncio
@pytest.mark.parametrize("persist", [False, True])
async def test_status_follows_the_tool_result_in_mistral(persist: bool) -> None:
    messages = await status_model_inputs(TextInput("saved"), persist=persist)

    status = UserMessage(content="saved")
    assert convert_messages([], list(messages), SerializerCls) == [
        UserMessage(content="start"),
        *([status] if persist else []),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(function=FunctionCall(name="echo", arguments="{}"), id="call-1", type="function", index=0)
            ],
        ),
        ToolMessage(content="tool output", tool_call_id="call-1"),
        status,
    ]
