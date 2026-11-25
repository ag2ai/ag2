# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Chat Completions reasoning is read from either provider field name.

OpenAI-compatible servers disagree on where reasoning goes: DeepSeek, SGLang,
xAI and DashScope send ``reasoning_content``; vLLM, OpenRouter and Ollama send
``reasoning``. A completion and a stream must both read either field and emit
one ``ModelReasoning`` per message or delta.
"""

from typing import Any

import httpx2
import pytest

from ag2 import Agent, MemoryStream
from ag2.config import OpenAICompletionsConfig
from ag2.events import ModelReasoning

from ._helpers import completion_chunk, completion_sse, recording

streaming_modes = pytest.mark.parametrize("streaming", [False, True], ids=["completion", "stream"])


def _completion(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _replying(response: httpx2.Response) -> httpx2.AsyncClient:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return response

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


def _config(*reasoning_deltas: dict[str, Any], streaming: bool) -> OpenAICompletionsConfig:
    """A server answering "ok" after the given reasoning fields, as one completion or as stream deltas."""
    if streaming:
        chunks = [completion_chunk(delta) for delta in reasoning_deltas]
        chunks.append(completion_chunk({"content": "ok"}, finish_reason="stop"))
        response = httpx2.Response(200, content=completion_sse(chunks), headers={"content-type": "text/event-stream"})
    else:
        message: dict[str, Any] = {"role": "assistant", "content": "ok"}
        for fields in reasoning_deltas:
            message.update(fields)
        response = httpx2.Response(200, json=_completion(message))

    return OpenAICompletionsConfig(
        model="test-model", api_key="test", streaming=streaming, http_client=_replying(response)
    )


async def _ask(config: OpenAICompletionsConfig) -> tuple[str | None, list[ModelReasoning]]:
    stream = MemoryStream()
    events = recording(stream)

    reply = await Agent("reasoner", config=config).ask("go", stream=stream)

    return reply.body, [e for e in events if isinstance(e, ModelReasoning)]


@pytest.mark.asyncio
@streaming_modes
@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
async def test_reasoning_is_read_from_either_field(field: str, streaming: bool) -> None:
    config = _config({field: "thought"}, streaming=streaming)

    assert await _ask(config) == ("ok", [ModelReasoning("thought")])


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
async def test_stream_emits_reasoning_per_delta(field: str) -> None:
    config = _config({field: "step 1 "}, {field: "step 2"}, streaming=True)

    assert await _ask(config) == ("ok", [ModelReasoning("step 1 "), ModelReasoning("step 2")])


@pytest.mark.asyncio
@streaming_modes
async def test_reasoning_content_wins_when_both_fields_are_set(streaming: bool) -> None:
    config = _config({"reasoning_content": "preferred", "reasoning": "fallback"}, streaming=streaming)

    assert await _ask(config) == ("ok", [ModelReasoning("preferred")])


@pytest.mark.asyncio
@streaming_modes
@pytest.mark.parametrize("empty", [None, ""])
async def test_empty_reasoning_content_falls_back_to_reasoning(empty: str | None, streaming: bool) -> None:
    config = _config({"reasoning_content": empty, "reasoning": "fallback"}, streaming=streaming)

    assert await _ask(config) == ("ok", [ModelReasoning("fallback")])
