# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent
from autogen.beta.events import ModelResponse
from autogen.beta.testing import TestConfig
from autogen.beta.tools.builtin.image_generation import (
    ImageGenerationTool,
    ImageGenerationToolSchema,
)


def test_schema_with_all_params() -> None:
    schema = ImageGenerationToolSchema(
        quality="high",
        size="1024x1024",
        background="transparent",
        output_format="png",
        output_compression=80,
        partial_images=2,
    )

    assert schema.quality == "high"
    assert schema.size == "1024x1024"
    assert schema.background == "transparent"
    assert schema.output_format == "png"
    assert schema.output_compression == 80
    assert schema.partial_images == 2


def test_schema_type_is_image_generation() -> None:
    schema = ImageGenerationToolSchema(quality="low")
    assert schema.type == "image_generation"


@pytest.mark.asyncio()
async def test_tool_schemas_returns_single_schema() -> None:
    tool = ImageGenerationTool(quality="medium", size="1536x1024")
    mock_ctx = MagicMock()
    schemas = await tool.schemas(mock_ctx)

    assert len(schemas) == 1
    schema = schemas[0]
    assert isinstance(schema, ImageGenerationToolSchema)
    assert schema.quality == "medium"
    assert schema.size == "1536x1024"


@pytest.mark.asyncio()
async def test_reply_images_empty_by_default() -> None:
    agent = Agent("test", config=TestConfig(ModelResponse()))
    reply = await agent.ask("hello")
    assert reply.images == []


@pytest.mark.asyncio()
async def test_reply_images_returns_decoded_bytes() -> None:
    raw = b"PNG_IMAGE_DATA"
    encoded = base64.b64encode(raw).decode()

    response = ModelResponse(images=[base64.b64decode(encoded)])
    agent = Agent("test", config=TestConfig(response))

    reply = await agent.ask("generate an image")
    assert reply.images == [raw]


@pytest.mark.asyncio()
async def test_reply_images_multiple() -> None:
    images_raw = [b"IMAGE_1", b"IMAGE_2"]
    images_bytes = [base64.b64decode(base64.b64encode(b)) for b in images_raw]

    response = ModelResponse(images=images_bytes)
    agent = Agent("test", config=TestConfig(response))

    reply = await agent.ask("generate images")
    assert len(reply.images) == 2
    assert reply.images[0] == b"IMAGE_1"
    assert reply.images[1] == b"IMAGE_2"
