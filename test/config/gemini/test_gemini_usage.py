# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Gemini client usage normalization."""

from unittest.mock import MagicMock

import httpx
import pytest
from google.genai import types

from ag2 import Agent
from ag2.config.gemini import GeminiConfig
from ag2.config.gemini.mappers import normalize_usage
from ag2.events import Usage


def _make_metadata(prompt=100, candidates=20, total=120, cached=None, thoughts=None, tool_use=None):
    m = MagicMock()
    m.prompt_token_count = prompt
    m.candidates_token_count = candidates
    m.total_token_count = total
    m.cached_content_token_count = cached
    m.thoughts_token_count = thoughts
    m.tool_use_prompt_token_count = tool_use
    return m


class TestNormalizeUsage:
    def test_normalizes_to_standard_keys(self):
        result = normalize_usage(_make_metadata())
        assert result == Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120)

    def test_includes_cache_read_tokens(self):
        result = normalize_usage(_make_metadata(cached=500))
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=500,
        )

    def test_no_cache_key_when_none(self):
        result = normalize_usage(_make_metadata(cached=None))
        assert result.cache_read_input_tokens is None

    def test_no_cache_key_when_zero(self):
        result = normalize_usage(_make_metadata(cached=0))
        assert result.cache_read_input_tokens is None

    def test_handles_none_token_counts_on_streaming_chunks(self):
        """Vertex streaming emits UsageMetadata with None fields before the
        final chunk. Those fields must not crash ``float(...)``."""
        result = normalize_usage(_make_metadata(prompt=None, candidates=None, total=None))
        assert result == Usage()
        assert bool(result) is False

    def test_handles_partial_token_counts(self):
        result = normalize_usage(_make_metadata(prompt=50, candidates=None, total=None))
        assert result == Usage(prompt_tokens=50)

    def test_counts_tool_use_prompt_tokens_as_prompt(self):
        # Grounding and code-execution output enters the prompt, is billed as
        # prompt, and sits inside the total beside the prompt count.
        result = normalize_usage(_make_metadata(tool_use=40, total=160))
        assert result == Usage(prompt_tokens=140, completion_tokens=20, total_tokens=160)

    def test_tool_use_prompt_tokens_without_a_prompt_count(self):
        result = normalize_usage(_make_metadata(prompt=None, tool_use=40))
        assert result.prompt_tokens == 40

    def test_includes_thinking_tokens(self):
        # Gemini counts thoughts beside the candidates, and both inside the total.
        result = normalize_usage(_make_metadata(thoughts=296, total=416))
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=316,
            total_tokens=416,
            thinking_tokens=296,
        )

    def test_no_thinking_key_when_none(self):
        result = normalize_usage(_make_metadata(thoughts=None))
        assert result.thinking_tokens is None

    def test_no_thinking_key_when_zero(self):
        result = normalize_usage(_make_metadata(thoughts=0))
        assert result.thinking_tokens is None

    def test_handles_metadata_without_thoughts_field(self):
        """Older google-genai versions may not expose ``thoughts_token_count``."""
        m = MagicMock(
            spec=["prompt_token_count", "candidates_token_count", "total_token_count", "cached_content_token_count"]
        )
        m.prompt_token_count = 100
        m.candidates_token_count = 20
        m.total_token_count = 120
        m.cached_content_token_count = None

        result = normalize_usage(m)
        assert result == Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120)


@pytest.mark.parametrize(
    ("candidates", "thoughts", "completion"),
    [(5, 40, 45), (None, 40, 40), (0, 40, 40), (5, None, 5), (5, 0, 5), (0, 0, 0), (None, 0, None), (None, None, None)],
)
def test_real_sdk_completion_includes_thinking(
    candidates: int | None, thoughts: int | None, completion: int | None
) -> None:
    metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=10,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
        total_token_count=55,
    )

    assert normalize_usage(metadata) == Usage(
        prompt_tokens=10,
        completion_tokens=completion,
        total_tokens=55,
        thinking_tokens=thoughts or None,
    )


def _usage_response(request: httpx.Request) -> httpx.Response:
    response = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text="done")]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=5,
            thoughts_token_count=40,
            total_token_count=55,
        ),
    )
    if request.url.path.endswith(":streamGenerateContent"):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {response.model_dump_json(by_alias=True)}\n\n",
        )
    return httpx.Response(200, json=response.model_dump(by_alias=True, exclude_none=True))


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_agent_usage_counts_thinking_as_completion(streaming: bool) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(_usage_response)) as http_client:
        agent = Agent(
            "test_agent",
            config=GeminiConfig(model="gemini-3-flash", api_key="test", http_client=http_client, streaming=streaming),
        )
        reply = await agent.ask("Hello")

    assert (await reply.usage()).total == Usage(
        prompt_tokens=10, completion_tokens=45, total_tokens=55, thinking_tokens=40
    )


def test_real_sdk_thinking_preserves_cache_and_tool_prompt_counts() -> None:
    metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=10,
        candidates_token_count=5,
        thoughts_token_count=40,
        total_token_count=62,
        tool_use_prompt_token_count=7,
        cached_content_token_count=8,
    )

    assert normalize_usage(metadata) == Usage(
        prompt_tokens=17,
        completion_tokens=45,
        total_tokens=62,
        thinking_tokens=40,
        cache_read_input_tokens=8,
    )
