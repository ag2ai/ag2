# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls

from ag2 import TextInput
from ag2.config.dashscope.mappers import convert_messages
from test.config._helpers import status_model_inputs


@pytest.mark.asyncio
@pytest.mark.parametrize("persist", [False, True])
async def test_status_follows_the_tool_result_in_dashscope(persist: bool) -> None:
    messages = await status_model_inputs(TextInput("saved"), persist=persist)

    status = {"role": "user", "content": "saved"}
    assert convert_messages([], list(messages), SerializerCls) == [
        {"role": "user", "content": "start"},
        *([status] if persist else []),
        {
            "content": None,
            "role": "assistant",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"arguments": "{}", "name": "echo"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "tool output"},
        status,
    ]
