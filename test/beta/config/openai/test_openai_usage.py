# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenAI client usage normalization (cache token lifting)."""

from autogen.beta.config.openai.mappers import normalize_usage


class TestNormalizeUsage:
    def test_lifts_cached_tokens(self):
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 80, "audio_tokens": None},
        }
        result = normalize_usage(usage)
        assert result["cache_read_input_tokens"] == 80
        assert result["prompt_tokens"] == 100

    def test_no_details_no_cache_key(self):
        usage = {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        result = normalize_usage(usage)
        assert "cache_read_input_tokens" not in result

    def test_details_with_zero_cached_tokens(self):
        usage = {
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "total_tokens": 60,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
        result = normalize_usage(usage)
        assert "cache_read_input_tokens" not in result

    def test_none_details(self):
        usage = {"prompt_tokens": 50, "prompt_tokens_details": None}
        result = normalize_usage(usage)
        assert "cache_read_input_tokens" not in result
