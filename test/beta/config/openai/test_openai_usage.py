# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenAI client usage normalization (cache token lifting)."""

from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

from autogen.beta.config.openai.mappers import normalize_usage
from autogen.beta.events import Usage


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
