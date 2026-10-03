# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx2
import pytest
import pytest_asyncio

pytest.importorskip("openai")

from openai import AsyncOpenAI, OpenAIError

from ag2.context import ConversationContext
from ag2.live import (
    OpenAIRealTimeConfig,
    OpenAITTSConfig,
    OpenAITranscriber,
    OpenAITranslationTranscriber,
    openai,
)
from ag2.live.stt import VoiceInput
from ag2.stream import MemoryStream


class SDKClients:
    """Construct real SDK clients, replacing only their HTTP transport."""

    def __init__(self) -> None:
        self.clients: list[AsyncOpenAI] = []
        self.http_clients: list[httpx2.AsyncClient] = []
        self.requests: list[httpx2.Request] = []

    def __call__(self, **kwargs: Any) -> AsyncOpenAI:
        http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(self.respond))
        self.http_clients.append(http_client)
        client = AsyncOpenAI(http_client=http_client, **kwargs)
        self.clients.append(client)
        return client

    def respond(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.url.path == "/v1/audio/transcriptions":
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text='data: {"type":"transcript.text.delta","delta":"Hello."}\n\ndata: [DONE]\n\n',
            )
        if request.url.path == "/v1/audio/translations":
            return httpx2.Response(200, headers={"content-type": "text/plain"}, text="Hello.")
        if request.url.path == "/v1/audio/speech":
            return httpx2.Response(200, content=b"test-pcm")
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")


@pytest_asyncio.fixture
async def sdk_clients(monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[SDKClients]:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.test/v1")
    clients = SDKClients()
    # The SDK constructor is a public dependency; no AG2 private methods are replaced.
    monkeypatch.setattr(openai, "AsyncOpenAI", clients)
    yield clients
    for client in clients.http_clients:
        await client.aclose()


@pytest.fixture(
    params=[OpenAIRealTimeConfig, OpenAITranscriber, OpenAITranslationTranscriber, OpenAITTSConfig],
    ids=["realtime", "transcription", "translation", "speech"],
)
def config_class(request: pytest.FixtureRequest) -> Any:
    return request.param


async def check_credentials(config: Any, client: AsyncOpenAI, sdk_clients: SDKClients, key: str) -> None:
    assert client.api_key == key
    assert client.auth_headers["Authorization"] == f"Bearer {key}"
    if isinstance(config, OpenAIRealTimeConfig):
        # Realtime uses WebSockets; inspect its public SDK client without opening a connection.
        assert config.client is client
        return
    if isinstance(config, OpenAITTSConfig):
        assert await config.synthesize("Hello.") == b"test-pcm"
    else:
        assert config.client is client
        context = ConversationContext(stream=MemoryStream())
        voice = VoiceInput(content=b"\x00\x00" * 16, frame_rate=24000, channels=1)
        assert await config.transcribe(voice, context) == "Hello."
    assert sdk_clients.requests[-1].headers["Authorization"] == f"Bearer {key}"


@pytest.mark.asyncio
@pytest.mark.parametrize("environment_key", [None, "test-environment-key"])
async def test_explicit_key_reaches_sdk(
    config_class: Any,
    sdk_clients: SDKClients,
    monkeypatch: pytest.MonkeyPatch,
    environment_key: str | None,
) -> None:
    if environment_key is not None:
        monkeypatch.setenv("OPENAI_API_KEY", environment_key)
    config = config_class("test-model", api_key="test-explicit-key")
    [client] = sdk_clients.clients
    await check_credentials(config, client, sdk_clients, "test-explicit-key")


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{}, {"api_key": None}], ids=["omitted", "none"])
async def test_environment_default(
    config_class: Any,
    sdk_clients: SDKClients,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-environment-key")
    config = config_class(model="test-model", **kwargs)
    [client] = sdk_clients.clients
    await check_credentials(config, client, sdk_clients, "test-environment-key")


@pytest.mark.parametrize("kwargs", [{}, {"api_key": None}], ids=["omitted", "none"])
def test_missing_key_keeps_sdk_error(config_class: Any, sdk_clients: SDKClients, kwargs: dict[str, Any]) -> None:
    with pytest.raises(OpenAIError, match="api_key"):
        config_class("test-model", **kwargs)
    assert not sdk_clients.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{}, {"api_key": None}], ids=["omitted", "none"])
async def test_injected_client_is_reused(config_class: Any, sdk_clients: SDKClients, kwargs: dict[str, Any]) -> None:
    client = sdk_clients(api_key="test-injected-key")
    for _ in range(2):
        config = config_class("test-model", client=client, **kwargs)
        await check_credentials(config, client, sdk_clients, "test-injected-key")
    assert sdk_clients.clients == [client]
    assert not client.is_closed()


@pytest.mark.parametrize("key", ["test-conflicting-key", "test-injected-key", ""], ids=["different", "same", "empty"])
def test_client_and_key_conflict(config_class: Any, sdk_clients: SDKClients, key: str) -> None:
    client = sdk_clients(api_key="test-injected-key")
    with pytest.raises(ValueError, match="client.*api_key") as exc:
        config_class("test-model", client=client, api_key=key)
    assert "test-injected-key" not in str(exc.value)
    assert "test-conflicting-key" not in str(exc.value)
    assert sdk_clients.clients == [client]
    assert not sdk_clients.requests
    assert not client.is_closed()


def test_api_key_is_optional_and_keyword_only(config_class: Any, sdk_clients: SDKClients) -> None:
    parameter = inspect.signature(config_class).parameters["api_key"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
    assert parameter.annotation == str | None
    with pytest.raises(TypeError):
        config_class("test-model", "test-positional-key")
    assert not sdk_clients.clients


@pytest.mark.asyncio
async def test_speech_options_are_preserved(sdk_clients: SDKClients) -> None:
    config = OpenAITTSConfig("tts-1", api_key="test-explicit-key", voice="nova", speed=1.25)
    assert await config.synthesize("Hello.") == b"test-pcm"
    [request] = sdk_clients.requests
    assert json.loads(request.content) == {
        "model": "tts-1",
        "voice": "nova",
        "input": "Hello.",
        "speed": 1.25,
        "response_format": "pcm",
    }
