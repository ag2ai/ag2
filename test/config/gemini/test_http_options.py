# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Callable
from typing import Any

import httpx
import httpx2
import pytest
from fast_depends.pydantic import PydanticSerializer
from google.oauth2.credentials import Credentials

from ag2 import Context, MemoryStream
from ag2.config import GeminiConfig, VertexAIConfig
from ag2.config.gemini import _compat
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
    async def test_default_follows_the_installed_sdk(self) -> None:
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return httpx.Response(200, json=GENERATE_RESPONSE)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await _ask(GeminiConfig(model="gemini-3.6-flash", api_key="test", http_client=http_client))

        sdk_default = "v1" if _compat.SDK_3X else "v1beta"
        assert paths == [f"/{sdk_default}/models/gemini-3.6-flash:generateContent"]

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
@pytest.mark.skipif(not _compat.SDK_3X, reason="google-genai 2.x only accepts an httpx client")
async def test_httpx2_client_carries_the_request() -> None:
    paths: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        return httpx2.Response(200, json=GENERATE_RESPONSE)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http_client:
        await _ask(GeminiConfig(model="gemini-3.6-flash", api_key="test", http_client=http_client))

    assert paths == ["/v1/models/gemini-3.6-flash:generateContent"]


@pytest.mark.skipif(_compat.SDK_3X, reason="google-genai 3.x accepts an httpx2 client")
def test_httpx2_client_is_rejected_by_sdk_2x() -> None:
    config = GeminiConfig(model="gemini-3.6-flash", api_key="test", http_client=httpx2.AsyncClient())

    with pytest.raises(TypeError, match=r"needs google-genai>=3\.0"):
        config.create()


CREATE_CALLS: list[Callable[[], Any]] = [
    lambda: GeminiConfig(model="gemini-3.6-flash", api_key="test").create(),
    lambda: GeminiConfig(model="gemini-3.6-flash", api_key="test").create_files_client(),
    lambda: VertexAIConfig(
        model="gemini-3.6-flash",
        project="test-project",
        location="us-central1",
        credentials=Credentials(token="fake-token"),
    ).create(),
]


@pytest.mark.skipif(_compat.SDK_3X, reason="only google-genai 2.x is deprecated")
@pytest.mark.parametrize("create", CREATE_CALLS, ids=["gemini", "gemini-files", "vertexai"])
def test_sdk_2x_is_deprecated(create: Callable[[], Any]) -> None:
    with pytest.warns(DeprecationWarning, match=r"google-genai 2\.x is deprecated and will be removed in 1\.4\.0"):
        create()


@pytest.mark.skipif(not _compat.SDK_3X, reason="google-genai 2.x is deprecated")
@pytest.mark.parametrize("create", CREATE_CALLS, ids=["gemini", "gemini-files", "vertexai"])
def test_sdk_3x_creates_without_warning(create: Callable[[], Any]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        create()
