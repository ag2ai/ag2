# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("openai")

from openai.types.realtime.realtime_response_usage import RealtimeResponseUsage
from openai.types.realtime.realtime_response_usage_input_token_details import (
    RealtimeResponseUsageInputTokenDetails,
)

from autogen.beta.events import Usage
from autogen.beta.live.openai import normalize_realtime_usage


def test_maps_standard_fields() -> None:
    usage = RealtimeResponseUsage(
        input_tokens=100,
        output_tokens=40,
        total_tokens=140,
        input_token_details=RealtimeResponseUsageInputTokenDetails(cached_tokens=30),
    )

    assert normalize_realtime_usage(usage) == Usage(
        prompt_tokens=100,
        completion_tokens=40,
        total_tokens=140,
        cache_read_input_tokens=30,
    )


def test_without_input_details() -> None:
    usage = RealtimeResponseUsage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        input_token_details=None,
    )

    assert normalize_realtime_usage(usage) == Usage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )


def test_none_usage_is_empty() -> None:
    assert normalize_realtime_usage(None) == Usage()
