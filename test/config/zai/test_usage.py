# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.config.zai.mappers import normalize_usage
from ag2.events import Usage
from test.config.zai._helpers import make_usage


def test_object_usage() -> None:
    result = normalize_usage(make_usage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    assert result == Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def test_total_tokens_defaults_to_sum() -> None:
    result = normalize_usage(make_usage(prompt_tokens=10, completion_tokens=5))

    assert result == Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def test_missing_values() -> None:
    assert normalize_usage(None) == Usage()


def test_optional_cache_and_reasoning_fields() -> None:
    result = normalize_usage(
        make_usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cached_tokens=3,
            reasoning_tokens=2,
        )
    )

    assert result == Usage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cache_read_input_tokens=3,
        thinking_tokens=2,
    )
