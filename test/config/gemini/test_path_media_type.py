# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import httpx
import pytest
from dirty_equals import IsOneOf, IsPartialDict
from fast_depends.pydantic import PydanticSerializer

from ag2 import Context, MemoryStream
from ag2.config import GeminiConfig
from ag2.events import ImageInput, ModelRequest


@pytest.mark.asyncio
async def test_explicit_path_media_type_reaches_sdk_request(tmp_path: Path) -> None:
    path = tmp_path / "download.jpg"
    path.write_bytes(b"image")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = GeminiConfig(model="test", api_key="test", http_client=http_client).create()
        response = await client(
            [ModelRequest([ImageInput(path=path, media_type="image/png")])],
            Context(stream=MemoryStream()),
            tools=(),
            response_schema=None,
            serializer=PydanticSerializer(),
        )

    [request] = requests
    assert json.loads(request.content) == IsPartialDict({
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": IsOneOf(
                            {"data": "aW1hZ2U=", "mimeType": "image/png"},
                            {"data": "aW1hZ2U=", "mime_type": "image/png"},
                        )
                    }
                ],
            }
        ]
    })
    assert response.content == "ok"
