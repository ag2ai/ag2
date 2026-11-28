# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A stream that ends in something other than `response.completed` is still reported."""

from typing import Any

import pytest
from openai import APIError

from ag2 import MemoryStream

from ._helpers import USAGE, ask, response, streaming_config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "extra"),
    [
        ("incomplete", {"incomplete_details": {"reason": "max_output_tokens"}}),
        ("failed", {"error": {"code": "server_error", "message": "boom"}}),
    ],
)
async def test_non_completed_terminal_event_keeps_response_metadata(status: str, extra: dict[str, Any]) -> None:
    events = [{"type": f"response.{status}", "sequence_number": 1, "response": response(status=status, **extra)}]

    result = await ask(streaming_config(events), stream=MemoryStream())

    assert result.finish_reason == status
    assert result.response_id == "resp_1"
    assert result.model == "gpt-5"
    assert result.usage.prompt_tokens == USAGE["input_tokens"]
    assert result.usage.completion_tokens == USAGE["output_tokens"]


@pytest.mark.asyncio
async def test_error_event_raises() -> None:
    events = [{"type": "error", "sequence_number": 1, "code": "server_error", "message": "boom", "param": None}]

    with pytest.raises(APIError, match="boom") as exc_info:
        await ask(streaming_config(events), stream=MemoryStream())

    assert exc_info.value.code == "server_error"
