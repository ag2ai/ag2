# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Converse usage normalization (inputTokens → prompt_tokens, cache keys)."""

from ag2.config.bedrock.mappers import normalize_usage
from ag2.events import Usage


def test_full_usage_with_cache_tokens() -> None:
    usage = normalize_usage({
        "inputTokens": 100,
        "outputTokens": 25,
        "totalTokens": 125,
        "cacheReadInputTokens": 40,
        "cacheWriteInputTokens": 10,
    })

    assert usage == Usage(
        prompt_tokens=100,
        completion_tokens=25,
        total_tokens=125,
        cache_read_input_tokens=40,
        cache_creation_input_tokens=10,
    )


def test_total_computed_when_missing() -> None:
    usage = normalize_usage({"inputTokens": 3, "outputTokens": 7})

    assert usage == Usage(prompt_tokens=3, completion_tokens=7, total_tokens=10)
    assert usage.cache_read_input_tokens is None
    assert usage.cache_creation_input_tokens is None


def test_empty_usage_is_falsy() -> None:
    usage = normalize_usage({})

    assert usage == Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    assert not usage
