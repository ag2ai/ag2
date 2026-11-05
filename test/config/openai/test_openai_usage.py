# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenAI client usage normalization (cache token lifting)."""

from openai.types.completion_usage import CompletionTokensDetails, CompletionUsage, PromptTokensDetails
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage

from ag2.config.openai.mappers import normalize_responses_usage, normalize_usage
from ag2.events import Usage


class TestNormalizeUsage:
    def test_lifts_cached_tokens(self):
        usage = CompletionUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=80),
        )
        result = normalize_usage(usage)
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=80,
        )

    def test_no_details_no_cache_key(self):
        usage = CompletionUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60)
        result = normalize_usage(usage)
        assert result.cache_read_input_tokens is None

    def test_details_with_zero_cached_tokens(self):
        usage = CompletionUsage(
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
        )
        result = normalize_usage(usage)
        assert result.cache_read_input_tokens == 0

    def test_none_details(self):
        usage = CompletionUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60, prompt_tokens_details=None)
        result = normalize_usage(usage)
        assert result.cache_read_input_tokens is None

    def test_lifts_reasoning_tokens(self):
        usage = CompletionUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=15),
        )
        result = normalize_usage(usage)
        # reasoning tokens map to thinking_tokens, NOT cache_creation_input_tokens
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            thinking_tokens=15,
        )
        assert result.cache_creation_input_tokens is None

    def test_chat_completions_lifts_cache_write_tokens(self):
        usage = CompletionUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=0, cache_write_tokens=80),
        )
        result = normalize_usage(usage)
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=80,
        )

    def test_responses_lifts_cache_write_tokens(self):
        usage = ResponseUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=80),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )
        result = normalize_responses_usage(usage)
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=80,
            thinking_tokens=0,
        )

    def test_chat_completions_details_none_returns_none(self):
        usage = CompletionUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60, prompt_tokens_details=None)
        assert normalize_usage(usage).cache_creation_input_tokens is None

    def test_cached_tokens_still_maps_to_cache_read_input_tokens(self):
        chat_usage = CompletionUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=80, cache_write_tokens=40),
        )
        assert normalize_usage(chat_usage).cache_read_input_tokens == 80

        resp_usage = ResponseUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_tokens_details=InputTokensDetails(cached_tokens=80, cache_write_tokens=40),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )
        assert normalize_responses_usage(resp_usage).cache_read_input_tokens == 80

    def test_details_with_zero_cache_write_tokens(self):
        chat_usage = CompletionUsage(
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=0, cache_write_tokens=0),
        )
        result_chat = normalize_usage(chat_usage)
        assert result_chat.cache_creation_input_tokens == 0

        resp_usage = ResponseUsage(
            input_tokens=50,
            output_tokens=10,
            total_tokens=60,
            input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )
        result_resp = normalize_responses_usage(resp_usage)
        assert result_resp.cache_creation_input_tokens == 0
