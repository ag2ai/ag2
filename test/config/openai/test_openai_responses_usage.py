# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenAI Responses client usage normalization."""

from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage

from ag2.config.openai.mappers import normalize_responses_usage
from ag2.events import Usage


class TestNormalizeUsage:
    def test_normalizes_input_output_keys(self):
        usage = ResponseUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )
        result = normalize_responses_usage(usage)
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=0,
            thinking_tokens=0,
        )

    def test_lifts_cached_tokens(self):
        usage = ResponseUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_tokens_details=InputTokensDetails(cached_tokens=80),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )
        result = normalize_responses_usage(usage)
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=80,
            thinking_tokens=0,
        )

    def test_lifts_reasoning_tokens(self):
        usage = ResponseUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=10),
        )
        result = normalize_responses_usage(usage)
        # reasoning tokens map to thinking_tokens, NOT cache_creation_input_tokens
        # (OpenAI has no cache-write metric, so cache_creation stays None)
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=0,
            thinking_tokens=10,
        )
        assert result.cache_creation_input_tokens is None
