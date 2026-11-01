# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls

from ag2 import TextInput
from ag2.config.zai.mappers import convert_messages
from test.config._helpers import status_model_inputs


@pytest.mark.asyncio
@pytest.mark.parametrize("persist", [False, True])
async def test_status_follows_the_tool_result_in_zai(persist: bool) -> None:
    messages = await status_model_inputs(TextInput("saved"), persist=persist)

    converted = convert_messages([], list(messages), SerializerCls)

    status = {"role": "user", "content": "saved"}
    head = [{"role": "user", "content": "start"}, *([status] if persist else [])]
    assert converted[: len(head)] == head
    assert converted[-2:] == [IsPartialDict({"role": "tool", "content": "tool output"}), status]
    assert converted.count(status) == (2 if persist else 1)
