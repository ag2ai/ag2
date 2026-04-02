# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenAI Responses client usage normalization."""

from autogen.beta.config.openai.mappers import normalize_responses_usage
from autogen.beta.events import Usage


class TestNormalizeUsage:
    def test_normalizes_input_output_keys(self):
        usage = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
        result = normalize_responses_usage(usage)
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )

    def test_lifts_cached_tokens(self):
        usage = {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 80},
        }
        result = normalize_responses_usage(usage)
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=80,
        )

    def test_no_details_no_cache_key(self):
        usage = {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}
        result = normalize_responses_usage(usage)
        assert result.cache_read_input_tokens is None

    def test_does_not_overwrite_existing_prompt_tokens(self):
        usage = {"input_tokens": 100, "prompt_tokens": 999}
        result = normalize_responses_usage(usage)
        assert result.prompt_tokens == 999
