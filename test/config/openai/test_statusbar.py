# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls

from ag2 import DataInput, ImageInput, TextInput, VideoInput
from ag2.config.openai.mappers import convert_messages, events_to_responses_input
from ag2.exceptions import UnsupportedInputError
from test.statusbar_support import status_model_inputs


@pytest.mark.asyncio
@pytest.mark.parametrize("persist", [False, True])
async def test_status_uses_completions_input_parts_and_preserves_tool_links(persist: bool) -> None:
    messages = await status_model_inputs(
        TextInput("status"), DataInput({"step": 1}), ImageInput("https://host/image.png"), persist=persist
    )

    expected = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "tool output"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "status"},
                {"type": "text", "text": '{"step":1}'},
                {"type": "image_url", "image_url": {"url": "https://host/image.png"}},
            ],
        },
    ]
    if persist:
        expected.insert(2, expected[-1])
    assert convert_messages([], messages, SerializerCls) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("persist", [False, True])
async def test_status_uses_responses_input_parts_and_preserves_tool_links(persist: bool) -> None:
    messages = await status_model_inputs(
        TextInput("status"), DataInput({"step": 1}), ImageInput("https://host/image.png"), persist=persist
    )

    expected = [
        {"role": "user", "content": [{"type": "input_text", "text": "start"}]},
        {"type": "function_call", "call_id": "call-1", "name": "echo", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call-1", "output": "tool output"},
        {"role": "user", "content": [{"type": "input_text", "text": "status"}]},
        {"role": "user", "content": [{"type": "input_text", "text": '{"step":1}'}]},
        {"role": "user", "content": [{"type": "input_image", "image_url": "https://host/image.png"}]},
    ]
    if persist:
        expected[1:1] = expected[-3:]
    assert events_to_responses_input(messages, SerializerCls) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
async def test_dynamic_status_retains_normal_unsupported_media_error(responses: bool) -> None:
    messages = await status_model_inputs(VideoInput(data=b"video", media_type="video/mp4"))

    with pytest.raises(UnsupportedInputError):
        if responses:
            events_to_responses_input(messages, SerializerCls)
        else:
            convert_messages([], messages, SerializerCls)
