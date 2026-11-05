# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls
from xai_sdk.chat import tool_result, user

from ag2 import TextInput
from ag2.config.xai.mappers import convert_messages
from test.config._helpers import status_model_inputs


@pytest.mark.asyncio
@pytest.mark.parametrize("persist", [False, True])
async def test_status_follows_the_tool_result_in_xai(persist: bool) -> None:
    messages = await status_model_inputs(TextInput("saved"), persist=persist)

    # Without the native assistant turn item xai replays no assistant message;
    # only the status placement relative to the tool result is asserted.
    assert convert_messages([], list(messages), SerializerCls) == [
        user("start"),
        *([user("saved")] if persist else []),
        tool_result("tool output", tool_call_id="call-1"),
        user("saved"),
    ]
