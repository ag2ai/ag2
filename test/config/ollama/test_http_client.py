# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest
from dirty_equals import IsPartialDict

from ag2 import Context, MemoryStream
from ag2.config.ollama import OllamaConfig
from ag2.events import ModelMessage, ModelMessageChunk, ModelReasoning, Usage
from ag2.tools import tool
from test.config.ollama._helpers import FakeOllama, ask, make_chunk


@pytest.mark.asyncio
async def test_host_is_the_url_base_of_the_callers_client(ollama: FakeOllama) -> None:
    await ask(ollama.config(host="http://ollama.test:1234"))

    assert str(ollama.request.url) == "http://ollama.test:1234/api/chat"


@pytest.mark.asyncio
async def test_host_is_normalised_by_the_sdk(ollama: FakeOllama) -> None:
    await ask(ollama.config(host="ollama.test"))

    assert str(ollama.request.url) == "http://ollama.test:11434/api/chat"


@pytest.mark.asyncio
async def test_default_host_is_localhost(ollama: FakeOllama) -> None:
    await ask(ollama.config())

    assert str(ollama.request.url) == "http://localhost:11434/api/chat"


@pytest.mark.asyncio
async def test_callers_client_is_not_modified_or_closed(ollama: FakeOllama) -> None:
    client = ollama.client(headers={"x-trace": "t1"}, timeout=7)

    await ask(ollama.config(client, host="http://ollama.test:1234", api_key="secret"))

    assert client.base_url == httpx.URL("")
    assert client.headers["x-trace"] == "t1"
    assert "authorization" not in client.headers
    assert client.timeout == httpx.Timeout(7)
    assert not client.is_closed
    assert ollama.request.headers["x-trace"] == "t1"


@pytest.mark.asyncio
async def test_api_key_is_sent_as_bearer_token(ollama: FakeOllama) -> None:
    await ask(ollama.config(api_key="secret"))

    assert ollama.request.headers["authorization"] == "Bearer secret"


def test_copy_carries_and_overrides_new_fields() -> None:
    client = httpx.AsyncClient()
    config = OllamaConfig(model="m1", api_key="k", http_client=client)

    assert config.copy().http_client is client
    assert config.copy().api_key == "k"
    assert config.copy(api_key="other").api_key == "other"


@pytest.mark.asyncio
async def test_non_streaming_reply(ollama: FakeOllama) -> None:
    ollama.chunks = [make_chunk(content="Paris", thinking="hmm", done=True)]

    result, events = await ask(ollama.config())

    assert result.message == ModelMessage("Paris")
    assert result.usage == Usage(prompt_tokens=4, completion_tokens=6, total_tokens=10)
    assert result.model == "m1"
    assert result.finish_reason == "stop"
    assert events == [ModelReasoning("hmm"), ModelMessage("Paris")]


@pytest.mark.asyncio
async def test_streaming_reasoning_text_and_final_usage(ollama: FakeOllama) -> None:
    ollama.chunks = [
        make_chunk(thinking="let me "),
        make_chunk(thinking="think"),
        make_chunk(content="Par"),
        make_chunk(content="is"),
        make_chunk(done=True),
    ]

    result, events = await ask(ollama.config(streaming=True))

    assert events == [
        ModelReasoning("let me "),
        ModelReasoning("think"),
        ModelMessageChunk("Par"),
        ModelMessageChunk("is"),
        ModelMessage("Paris"),
    ]
    assert result.message == ModelMessage("Paris")
    assert result.usage == Usage(prompt_tokens=4, completion_tokens=6, total_tokens=10)
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_request_carries_options_and_tools(ollama: FakeOllama) -> None:
    @tool
    def get_weather(city: str) -> str:
        """Weather for a city."""
        return city

    config = ollama.config(streaming=True, temperature=0.2, top_p=0.9, max_tokens=50, seed=7)

    await ask(config, tools=list(await get_weather.schemas(Context(stream=MemoryStream()))))

    assert ollama.body == IsPartialDict({
        "model": "m1",
        "stream": True,
        "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 50, "seed": 7},
        "tools": [
            IsPartialDict({
                "type": "function",
                "function": IsPartialDict({"name": "get_weather", "description": "Weather for a city."}),
            })
        ],
    })
