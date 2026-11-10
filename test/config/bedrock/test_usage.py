# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Converse usage normalization.

``prompt_tokens`` is the whole input: ``inputTokens`` plus both cache counts (ADR 0036).
"""

from ag2.config.bedrock.mappers import normalize_usage
from ag2.events import Usage


def test_a_total_that_leaves_out_the_cache_is_recomputed() -> None:
    """The cache counts join the input, so a reported total below input plus output is not believed."""
    usage = normalize_usage({
        "inputTokens": 100,
        "outputTokens": 25,
        "totalTokens": 125,
        "cacheReadInputTokens": 40,
        "cacheWriteInputTokens": 10,
    })

    assert usage == Usage(
        prompt_tokens=150,
        completion_tokens=25,
        total_tokens=175,
        cache_read_input_tokens=40,
        cache_creation_input_tokens=10,
    )


def test_a_total_that_already_counts_the_cache_is_kept() -> None:
    usage = normalize_usage({
        "inputTokens": 4,
        "outputTokens": 236,
        "totalTokens": 1468,
        "cacheWriteInputTokens": 1228,
    })

    assert usage == Usage(
        prompt_tokens=1232,
        completion_tokens=236,
        total_tokens=1468,
        cache_creation_input_tokens=1228,
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
