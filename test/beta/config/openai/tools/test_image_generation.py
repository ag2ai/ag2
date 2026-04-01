# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.config.openai.mappers import tool_to_api, tool_to_responses_api
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.tools.builtin.image_generation import ImageGenerationToolSchema


def test_tool_to_api_image_generation_raises() -> None:
    schema = ImageGenerationToolSchema()

    with pytest.raises(UnsupportedToolError) as exc_info:
        tool_to_api(schema)

    assert "image_generation" in str(exc_info.value)
    assert "openai-completions" in str(exc_info.value)


def test_tool_to_responses_api_image_generation_defaults() -> None:
    schema = ImageGenerationToolSchema()
    result = tool_to_responses_api(schema)

    assert result == {"type": "image_generation"}


def test_tool_to_responses_api_image_generation_quality() -> None:
    schema = ImageGenerationToolSchema(quality="high")
    result = tool_to_responses_api(schema)

    assert result == {"type": "image_generation", "quality": "high"}


def test_tool_to_responses_api_image_generation_size() -> None:
    schema = ImageGenerationToolSchema(size="1536x1024")
    result = tool_to_responses_api(schema)

    assert result == {"type": "image_generation", "size": "1536x1024"}


def test_tool_to_responses_api_image_generation_background() -> None:
    schema = ImageGenerationToolSchema(background="transparent")
    result = tool_to_responses_api(schema)

    assert result == {"type": "image_generation", "background": "transparent"}


def test_tool_to_responses_api_image_generation_output_format() -> None:
    schema = ImageGenerationToolSchema(output_format="webp")
    result = tool_to_responses_api(schema)

    assert result == {"type": "image_generation", "output_format": "webp"}


def test_tool_to_responses_api_image_generation_output_compression() -> None:
    schema = ImageGenerationToolSchema(output_format="jpeg", output_compression=75)
    result = tool_to_responses_api(schema)

    assert result == {"type": "image_generation", "output_format": "jpeg", "output_compression": 75}


def test_tool_to_responses_api_image_generation_partial_images() -> None:
    schema = ImageGenerationToolSchema(partial_images=2)
    result = tool_to_responses_api(schema)

    assert result == {"type": "image_generation", "partial_images": 2}


def test_tool_to_responses_api_image_generation_all_params() -> None:
    schema = ImageGenerationToolSchema(
        quality="medium",
        size="1024x1024",
        background="opaque",
        output_format="png",
        partial_images=1,
    )
    result = tool_to_responses_api(schema)

    assert result == {
        "type": "image_generation",
        "quality": "medium",
        "size": "1024x1024",
        "background": "opaque",
        "output_format": "png",
        "partial_images": 1,
    }
