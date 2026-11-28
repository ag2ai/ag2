# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import httpx
import httpx2
import pytest
from fast_depends.pydantic import PydanticSerializer

from ag2 import Context, MemoryStream
from ag2.config import GeminiConfig
from ag2.events import ModelRequest, TextInput
from ag2.files import FilesAPI

GENERATE_RESPONSE = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
}


async def _ask(config: GeminiConfig) -> None:
    await config.create()(
        messages=[ModelRequest([TextInput("capital of France?")])],
        context=Context(stream=MemoryStream()),
        tools=[],
        response_schema=None,
        serializer=PydanticSerializer(),
    )


@pytest.mark.asyncio
class TestApiVersion:
    async def test_default_is_v1(self) -> None:
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return httpx.Response(200, json=GENERATE_RESPONSE)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await _ask(GeminiConfig(model="gemini-3.6-flash", api_key="test", http_client=http_client))

        assert paths == ["/v1/models/gemini-3.6-flash:generateContent"]

    async def test_set_version_reaches_generate_content(self) -> None:
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return httpx.Response(200, json=GENERATE_RESPONSE)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            config = GeminiConfig(
                model="gemini-3.1-pro-preview",
                api_key="test",
                api_version="v1beta",
                http_client=http_client,
            )
            await _ask(config)

        assert paths == ["/v1beta/models/gemini-3.1-pro-preview:generateContent"]

    async def test_set_version_reaches_files_api(self) -> None:
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return httpx.Response(200, json={"files": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            api = FilesAPI(GeminiConfig(model="test", api_key="test", api_version="v1beta", http_client=http_client))
            await api.list()

        assert paths == ["/v1beta/files"]


@pytest.mark.asyncio
async def test_httpx2_client_carries_the_request() -> None:
    paths: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        return httpx2.Response(200, json=GENERATE_RESPONSE)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http_client:
        await _ask(GeminiConfig(model="gemini-3.6-flash", api_key="test", http_client=http_client))

    assert paths == ["/v1/models/gemini-3.6-flash:generateContent"]
