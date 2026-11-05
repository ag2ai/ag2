# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls

from ag2 import TextInput
from ag2.config.bedrock.mappers import convert_messages
from test.config._helpers import status_model_inputs


@pytest.mark.asyncio
@pytest.mark.parametrize("persist", [False, True])
async def test_status_joins_the_tool_result_turn_in_bedrock(persist: bool) -> None:
    messages = await status_model_inputs(TextInput("saved"), persist=persist)

    first_turn = [{"text": "start"}, {"text": "saved"}] if persist else [{"text": "start"}]
    assert convert_messages(list(messages), SerializerCls) == [
        {"role": "user", "content": first_turn},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "call-1", "name": "echo", "input": {}}}]},
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "call-1", "content": [{"text": "tool output"}]}},
                {"text": "saved"},
            ],
        },
    ]
