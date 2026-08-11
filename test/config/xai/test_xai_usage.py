# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from xai_sdk.proto import usage_pb2

from ag2.config.xai.mappers import normalize_usage
from ag2.events import Usage


def test_none_returns_empty_usage() -> None:
    result = normalize_usage(None)

    assert result == Usage()
    assert not result


def test_basic_token_counts() -> None:
    raw = usage_pb2.SamplingUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    result = normalize_usage(raw)

    assert result == Usage(prompt_tokens=10.0, completion_tokens=5.0, total_tokens=15.0)


def test_cached_prompt_tokens_map_to_cache_read() -> None:
    raw = usage_pb2.SamplingUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cached_prompt_text_tokens=4)

    result = normalize_usage(raw)

    assert result.cache_read_input_tokens == 4.0


def test_reasoning_tokens_map_to_thinking_tokens() -> None:
    raw = usage_pb2.SamplingUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, reasoning_tokens=3)

    result = normalize_usage(raw)

    assert result.thinking_tokens == 3.0


def test_zero_or_missing_fields_become_none() -> None:
    raw = usage_pb2.SamplingUsage()

    result = normalize_usage(raw)

    # zero counts collapse to None per the normalize_usage contract
    assert result == Usage()
    assert not result
