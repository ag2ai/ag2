# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls

from ag2 import DataInput, ImageInput, TextInput
from ag2.config.gemini.mappers import convert_messages
from test.config._helpers import status_model_inputs


@pytest.mark.asyncio
@pytest.mark.parametrize("persist", [False, True])
async def test_status_uses_gemini_input_parts_and_preserves_tool_links(persist: bool) -> None:
    messages = await status_model_inputs(
        TextInput("status"), DataInput({"step": 1}), ImageInput(data=b"png", media_type="image/png"), persist=persist
    )

    expected = [
        {"role": "user", "parts": [{"text": "start"}]},
        {"role": "model", "parts": [{"function_call": {"id": "call-1", "name": "echo", "args": {}}}]},
        {
            "role": "user",
            "parts": [{"function_response": {"id": "call-1", "name": "echo", "response": {"result": "tool output"}}}],
        },
        {
            "role": "user",
            "parts": [
                {"text": "status"},
                {"text": '{"step":1}'},
                {"inline_data": {"data": b"png", "mime_type": "image/png"}},
            ],
        },
    ]
    if persist:
        expected.insert(1, expected[-1])
    assert [content.model_dump(exclude_none=True) for content in convert_messages(messages, SerializerCls)] == expected
