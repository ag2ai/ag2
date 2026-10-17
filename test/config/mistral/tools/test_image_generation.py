# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from mistralai.client.models import ImageGenerationTool as MistralImageGenerationTool

from ag2 import Context
from ag2.config.mistral.mappers import server_tool_name, server_tool_result, tool_to_api
from ag2.events import BinaryType, TextInput, UrlInput
from ag2.tools.builtin.image_generation import ImageGenerationTool


@pytest.mark.asyncio
class TestToolMapping:
    async def test_maps_to_mistral_tool(self, context: Context) -> None:
        tool = ImageGenerationTool()

        [schema] = await tool.schemas(context)

        assert isinstance(tool_to_api(schema), MistralImageGenerationTool)

    async def test_openai_only_options_are_dropped(self, context: Context) -> None:
        """Mistral's tool takes no configuration; size/quality have nowhere to go."""
        tool = ImageGenerationTool(size="1024x1024", quality="high", output_format="png")

        [schema] = await tool.schemas(context)

        assert tool_to_api(schema) == MistralImageGenerationTool()


def test_server_name_is_mapped_to_the_ag2_tool_name() -> None:
    """Mistral calls it ``generate_image``; unmapped, the agent logs a not-found."""
    assert server_tool_name("generate_image") == "image_generation"


def test_unknown_server_names_pass_through() -> None:
    assert server_tool_name("something_else") == "something_else"


def test_image_result_becomes_a_url_input() -> None:
    result = server_tool_result('{"url": "https://example.com/generated.jpg"}')

    assert result.parts == [UrlInput("https://example.com/generated.jpg", kind=BinaryType.IMAGE)]


def test_non_json_result_falls_back_to_text() -> None:
    assert server_tool_result("something went wrong").parts == [TextInput("something went wrong")]


def test_json_without_url_falls_back_to_text() -> None:
    assert server_tool_result('{"status": "pending"}').parts == [TextInput('{"status": "pending"}')]
