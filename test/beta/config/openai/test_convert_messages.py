# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config.openai.mappers import convert_messages, events_to_responses_input
from autogen.beta.events import ImageInput

IMAGE_URL = "https://example.com/image.png"


def test_completions_image_input() -> None:
    result = convert_messages([], [ImageInput(url=IMAGE_URL)])

    msg = result[1]  # index 0 is the system prompt
    assert msg["role"] == "user"
    assert msg["content"] == [
        {"type": "image_url", "image_url": {"url": IMAGE_URL}},
    ]


def test_responses_image_input() -> None:
    result = events_to_responses_input([ImageInput(url=IMAGE_URL)])

    assert result == [
        {
            "role": "user",
            "content": [{"type": "input_image", "image_url": IMAGE_URL}],
        }
    ]
