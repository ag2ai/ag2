# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls

from ag2 import DataInput, ImageInput, TextInput, VideoInput
from ag2.config.anthropic.mappers import convert_messages
from ag2.exceptions import UnsupportedInputError
from test.statusbar_support import status_model_inputs


@pytest.mark.asyncio
@pytest.mark.parametrize("persist", [False, True])
async def test_status_uses_anthropic_input_parts_and_preserves_tool_links(persist: bool) -> None:
    messages = await status_model_inputs(
        TextInput("status"), DataInput({"step": 1}), ImageInput(data=b"png", media_type="image/png"), persist=persist
    )

    expected = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "call-1", "name": "echo", "input": {}}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "tool output"}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "status"},
                {"type": "text", "text": '{"step":1}'},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "cG5n"}},
            ],
        },
    ]
    if persist:
        expected.insert(1, expected[-1])
    assert convert_messages(messages, SerializerCls) == expected


@pytest.mark.asyncio
async def test_dynamic_status_retains_normal_anthropic_unsupported_media_error() -> None:
    messages = await status_model_inputs(VideoInput(data=b"video", media_type="video/mp4"))

    with pytest.raises(UnsupportedInputError):
        convert_messages(messages, SerializerCls)
