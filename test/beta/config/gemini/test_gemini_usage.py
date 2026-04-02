# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Gemini client usage normalization."""

from unittest.mock import MagicMock

from autogen.beta.config.gemini.mappers import normalize_usage
from autogen.beta.events import Usage


def _make_metadata(prompt=100, candidates=20, total=120, cached=None):
    m = MagicMock()
    m.prompt_token_count = prompt
    m.candidates_token_count = candidates
    m.total_token_count = total
    m.cached_content_token_count = cached
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
