# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from ag2.config.mistral.mappers import normalize_usage
from ag2.events import Usage
from test.config.mistral._helpers import make_usage


def test_none_usage() -> None:
    assert normalize_usage(None) == Usage()


def test_full_usage() -> None:
    assert normalize_usage(make_usage(10, 5, 15)) == Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def test_total_is_derived_when_absent() -> None:
    assert normalize_usage(make_usage(10, 5, None)).total_tokens == 15


def test_cached_tokens_from_dict_details() -> None:
    usage = normalize_usage(make_usage(10, 5, 15, cached_tokens=6))

    assert usage.cache_read_input_tokens == 6


def test_cached_tokens_from_object_details() -> None:
    """The field is a pydantic extra, so it may arrive as an object too."""
    raw = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        prompt_tokens_details=SimpleNamespace(cached_tokens=2),
    )

    assert normalize_usage(raw).cache_read_input_tokens == 2


def test_missing_details_leaves_cache_unset() -> None:
    assert normalize_usage(make_usage(1, 1, 2)).cache_read_input_tokens is None


def test_empty_usage_stays_empty() -> None:
    assert normalize_usage(make_usage()) == Usage()
