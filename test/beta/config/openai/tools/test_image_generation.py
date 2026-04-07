# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.config.openai.mappers import tool_to_responses_api
from autogen.beta.context import Context
from autogen.beta.tools.builtin.image_generation import ImageGenerationTool


@pytest.mark.asyncio
async def test_responses_api_defaults(context: Context) -> None:
    tool = ImageGenerationTool()

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {"type": "image_generation"}


@pytest.mark.asyncio
async def test_responses_api_quality(context: Context) -> None:
    tool = ImageGenerationTool(quality="high")

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {"type": "image_generation", "quality": "high"}


@pytest.mark.asyncio
async def test_responses_api_size(context: Context) -> None:
    tool = ImageGenerationTool(size="1536x1024")

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {"type": "image_generation", "size": "1536x1024"}


@pytest.mark.asyncio
async def test_responses_api_background(context: Context) -> None:
    tool = ImageGenerationTool(background="transparent")

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {"type": "image_generation", "background": "transparent"}


@pytest.mark.asyncio
async def test_responses_api_output_format(context: Context) -> None:
    tool = ImageGenerationTool(output_format="webp")

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {"type": "image_generation", "output_format": "webp"}


@pytest.mark.asyncio
async def test_responses_api_output_compression(context: Context) -> None:
    tool = ImageGenerationTool(output_format="jpeg", output_compression=75)

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {
        "type": "image_generation",
        "output_format": "jpeg",
        "output_compression": 75,
    }


@pytest.mark.asyncio
async def test_responses_api_partial_images(context: Context) -> None:
    tool = ImageGenerationTool(partial_images=2)

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {"type": "image_generation", "partial_images": 2}


@pytest.mark.asyncio
async def test_responses_api_all_params(context: Context) -> None:
    tool = ImageGenerationTool(
        quality="medium",
        size="1024x1024",
        background="opaque",
        output_format="png",
        partial_images=1,
    )

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {
        "type": "image_generation",
        "quality": "medium",
        "size": "1024x1024",
        "background": "opaque",
        "output_format": "png",
        "partial_images": 1,
    }
