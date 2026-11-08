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
    raw = usage_pb2.SamplingUsage(prompt_tokens=10, completion_tokens=5, total_tokens=18, reasoning_tokens=3)

    result = normalize_usage(raw)

    assert result.thinking_tokens == 3.0


def test_completion_includes_reasoning_tokens() -> None:
    # xAI reports reasoning beside the completion: total = prompt + completion + reasoning.
    raw = usage_pb2.SamplingUsage(prompt_tokens=204, completion_tokens=32, total_tokens=536, reasoning_tokens=300)

    result = normalize_usage(raw)

    assert result == Usage(prompt_tokens=204.0, completion_tokens=332.0, total_tokens=536.0, thinking_tokens=300.0)
    assert result.prompt_tokens + result.completion_tokens == result.total_tokens


def test_reasoning_without_completion_tokens() -> None:
    raw = usage_pb2.SamplingUsage(prompt_tokens=10, total_tokens=50, reasoning_tokens=40)

    assert normalize_usage(raw).completion_tokens == 40.0


def test_zero_or_missing_fields_become_none() -> None:
    raw = usage_pb2.SamplingUsage()

    result = normalize_usage(raw)

    # zero counts collapse to None per the normalize_usage contract
    assert result == Usage()
    assert not result
