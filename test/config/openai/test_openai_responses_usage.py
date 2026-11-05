# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenAI Responses client usage normalization."""

import pytest
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage

from ag2.config.openai.mappers import normalize_responses_usage
from ag2.events import Usage

from ._helpers import ask_client, capturing_config, message, response


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("long_lived", [0, 60])
async def test_retention_breakdown_from_provider_json(streaming: bool, long_lived: int) -> None:
    usage = {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "input_tokens_details": {
            "cached_tokens": 0,
            "cache_write_tokens": 80,
            "cache_write_tokens_details": {"ttl_30m_tokens": 20, "ttl_12h_tokens": long_lived},
        },
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    config, bodies = capturing_config(response(message("ok"), usage=usage), stream=streaming)
    result = await ask_client(config.copy(prompt_cache_options={"ttl": "12h"}).create())

    assert bodies[0]["prompt_cache_options"] == {"ttl": "12h"}
    assert result.usage == Usage(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=80,
        thinking_tokens=0,
        cache_creation_input_tokens_by_ttl={"30m": 20, "12h": long_lived},
    )


class TestNormalizeUsage:
    def test_normalizes_input_output_keys(self):
        usage = ResponseUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )
        result = normalize_responses_usage(usage)
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            thinking_tokens=0,
        )

    def test_lifts_cached_tokens(self):
        usage = ResponseUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_tokens_details=InputTokensDetails(cached_tokens=80, cache_write_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )
        result = normalize_responses_usage(usage)
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=80,
            cache_creation_input_tokens=0,
            thinking_tokens=0,
        )

    def test_lifts_reasoning_tokens(self):
        usage = ResponseUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=10),
        )
        result = normalize_responses_usage(usage)
        # Reasoning tokens stay separate from the reported cache-write count.
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            thinking_tokens=10,
        )
