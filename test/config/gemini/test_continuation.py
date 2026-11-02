# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for responses Gemini cuts at the per-request token limit (``FinishReason.CONTINUATION``)."""

import json

import httpx
import pytest
from google.genai import types

from ag2 import Agent
from ag2.config.gemini import GeminiConfig, VertexAIConfig
from ag2.events import ModelResponse, Usage

TOKEN = b"resume-here"


def _hop(text: str, *, finish: str, token: bytes | None, prompt: int, candidates: int) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                finish_reason=types.FinishReason(finish),
                continuation_token=token,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt,
            candidates_token_count=candidates,
            total_token_count=prompt + candidates,
        ),
    )


FIRST_HOP = _hop("The answer ", finish="CONTINUATION", token=TOKEN, prompt=10, candidates=4)
LAST_HOP = _hop("is 42.", finish="STOP", token=None, prompt=12, candidates=3)


class _Server:
    """Answers the first request with ``CONTINUATION`` and a resumed one with ``STOP``."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        hop = LAST_HOP if body.get("continuationToken") else FIRST_HOP
        payload = hop.model_dump_json(by_alias=True, exclude_none=True)
        if request.url.path.endswith(":streamGenerateContent"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=f"data: {payload}\n\n")
        return httpx.Response(200, content=payload, headers={"content-type": "application/json"})


async def _call(config: GeminiConfig, server: _Server) -> ModelResponse:
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as http_client:
        reply = await Agent("test_agent", config=config.copy(http_client=http_client)).ask("Hello")
    return reply.response


def test_copy_overrides_automatic_continuation() -> None:
    assert GeminiConfig(model="m").copy(automatic_continuation=False).automatic_continuation is False
    assert VertexAIConfig(model="m").copy(automatic_continuation=False).automatic_continuation is False


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("automatic_continuation", [None, True])
async def test_automatic_continuation_joins_the_hops(streaming: bool, automatic_continuation: bool | None) -> None:
    server = _Server()

    response = await _call(
        GeminiConfig(
            model="gemini-3-flash", api_key="test", streaming=streaming, automatic_continuation=automatic_continuation
        ),
        server,
    )

    assert [r.get("continuationToken") for r in server.requests] == [None, "cmVzdW1lLWhlcmU="]
    assert response.message is not None
    assert response.message.content == "The answer is 42."
    assert response.finish_reason == "stop"
    # Every hop is a request billed on its own, so usage is the sum of the hops.
    assert response.usage == Usage(prompt_tokens=22, completion_tokens=7, total_tokens=29)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_disabled_continuation_reports_the_truncation(streaming: bool) -> None:
    server = _Server()

    response = await _call(
        GeminiConfig(model="gemini-3-flash", api_key="test", streaming=streaming, automatic_continuation=False),
        server,
    )

    assert len(server.requests) == 1
    assert response.message is not None
    assert response.message.content == "The answer "
    assert response.finish_reason == "continuation"
    assert response.usage == Usage(prompt_tokens=10, completion_tokens=4, total_tokens=14)
