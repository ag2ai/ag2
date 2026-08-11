# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from google.genai import types

from ag2 import Context, MemoryStream
from ag2.config.gemini.gemini_client import GeminiClient
from ag2.events import BinaryResult

_PNG = b"\x89PNG\r\n\x1a\nfake-image-bytes"


def _candidate(parts: list) -> SimpleNamespace:
    return SimpleNamespace(content=SimpleNamespace(parts=parts), finish_reason=None, grounding_metadata=None)


def _response(candidates: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(candidates=candidates, usage_metadata=None)


@pytest.fixture
def client() -> GeminiClient:
    with patch("ag2.config.gemini.gemini_client.genai.Client"):
        return GeminiClient(model="gemini-3.1-flash-image", api_key="test-key")


@pytest.fixture
def context() -> Context:
    return Context(stream=MemoryStream())


@pytest.mark.asyncio
class TestImageOutput:
    async def test_inline_image_becomes_binary_file(self, client: GeminiClient, context: Context) -> None:
        image_part = types.Part(inline_data=types.Blob(data=_PNG, mime_type="image/png"))

        response = await client._process_response(_response([_candidate([image_part])]), context)

        assert response.files == [BinaryResult(_PNG, metadata={"media_type": "image/png"})]

    async def test_text_and_image_parts_split_into_body_and_files(self, client: GeminiClient, context: Context) -> None:
        text_part = types.Part(text="Here is your image:")
        image_part = types.Part(inline_data=types.Blob(data=_PNG, mime_type="image/png"))

        response = await client._process_response(_response([_candidate([text_part, image_part])]), context)

        assert response.content == "Here is your image:"
        assert response.files == [BinaryResult(_PNG, metadata={"media_type": "image/png"})]

    async def test_no_image_yields_empty_files(self, client: GeminiClient, context: Context) -> None:
        response = await client._process_response(_response([_candidate([types.Part(text="just text")])]), context)

        assert response.files == []

    async def test_streamed_inline_image_becomes_binary_file(self, client: GeminiClient, context: Context) -> None:
        text_part = types.Part(text="rendering")
        image_part = types.Part(inline_data=types.Blob(data=_PNG, mime_type="image/png"))

        async def chunks():
            yield _response([_candidate([text_part])])
            yield _response([_candidate([image_part])])

        response = await client._process_stream(chunks(), context)

        assert response.content == "rendering"
        assert response.files == [BinaryResult(_PNG, metadata={"media_type": "image/png"})]
