# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("google.genai")

from google.genai import types as gtypes

from ag2.events import Usage
from ag2.live.gemini import normalize_realtime_usage


def test_maps_live_fields() -> None:
    # Live API names output tokens ``response_token_count`` (not the batch
    # API's ``candidates_token_count``).
    metadata = gtypes.UsageMetadata(
        prompt_token_count=80,
        response_token_count=20,
        total_token_count=100,
        cached_content_token_count=10,
        thoughts_token_count=5,
    )

    assert normalize_realtime_usage(metadata) == Usage(
        prompt_tokens=80,
        completion_tokens=20,
        total_tokens=100,
        cache_read_input_tokens=10,
        thinking_tokens=5,
    )


def test_none_metadata_is_empty() -> None:
    assert normalize_realtime_usage(None) == Usage()
