# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Normalized usage, read off provider JSON the SDK parsed.

The reasoning count arrives nested under ``output_tokens_details``, and whether a
cache count was *measured* is a question about the payload's shape — neither is
observable through a mock whose ``model_dump`` decides that shape. So these drive a
real client through an ``httpx2`` mock transport and assert on the ``Usage`` that
comes back, the same seam ``test_request_body.py`` uses for the outgoing direction.
"""

import json
from typing import Any

import httpx2
import pytest
from fast_depends.pydantic import PydanticSerializer

from ag2 import Context, MemoryStream
from ag2.config.anthropic import AnthropicConfig
from ag2.events import ModelRequest, ModelResponse, TextInput, Usage, UsageEvent
from ag2.usage import UsageReport


def _message(usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }


def _stream_events(start_usage: dict[str, Any], delta_usage: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return (
        {"type": "message_start", "message": {**_message(start_usage), "content": [], "stop_reason": None}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": delta_usage,
        },
        {"type": "message_stop"},
    )


def _serving(payload: dict[str, Any] | tuple[dict[str, Any], ...]) -> httpx2.AsyncClient:
    """A client whose transport answers with the given message, or SSE events."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if isinstance(payload, dict):
            return httpx2.Response(200, json=payload)
        body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in payload)
        return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


async def _usage_of(
    payload: dict[str, Any] | tuple[dict[str, Any], ...],
    *,
    streaming: bool = False,
) -> Usage:
    config = AnthropicConfig(
        model="claude-haiku-4-5",
        api_key="test",
        streaming=streaming,
        http_client=_serving(payload),
    )
    response = await config.create()(
        messages=[ModelRequest([TextInput("hi")])],
        context=Context(stream=MemoryStream()),
        tools=[],
        response_schema=None,
        serializer=PydanticSerializer(),
    )
    assert isinstance(response, ModelResponse)
    return response.usage


pytestmark = pytest.mark.asyncio


class TestThinkingTokens:
    async def test_a_non_streaming_call_reports_the_providers_figure(self) -> None:
        """The whole-object comparison pins the decomposition too: 214 thinking tokens
        are part of the 300 output tokens, so neither the completion nor the total grows."""
        usage = await _usage_of(
            _message({
                "input_tokens": 12,
                "output_tokens": 300,
                "output_tokens_details": {"thinking_tokens": 214},
            })
        )

        assert usage == Usage(prompt_tokens=12, completion_tokens=300, total_tokens=312, thinking_tokens=214)

    async def test_a_streaming_call_reports_it_from_the_final_message(self) -> None:
        usage = await _usage_of(
            _stream_events(
                {"input_tokens": 12, "output_tokens": 1},
                {"output_tokens": 300, "output_tokens_details": {"thinking_tokens": 214}},
            ),
            streaming=True,
        )

        assert usage == Usage(prompt_tokens=12, completion_tokens=300, total_tokens=312, thinking_tokens=214)

    async def test_an_omitted_details_object_leaves_it_absent(self) -> None:
        """The live API always sends the details object (see ``TestTheShapeTheApiSends``),
        so this pins the contract for a payload that omits it rather than today's traffic."""
        usage = await _usage_of(_message({"input_tokens": 12, "output_tokens": 8}))

        assert usage == Usage(prompt_tokens=12, completion_tokens=8, total_tokens=20)

    async def test_a_call_that_did_not_reason_reports_the_measured_zero(self) -> None:
        usage = await _usage_of(
            _message({
                "input_tokens": 12,
                "output_tokens": 8,
                "output_tokens_details": {"thinking_tokens": 0},
            })
        )

        assert usage == Usage(prompt_tokens=12, completion_tokens=8, total_tokens=20, thinking_tokens=0)


class TestMeasuredZero:
    async def test_zero_cache_creation_tokens_survive_as_zero(self) -> None:
        usage = await _usage_of(_message({"input_tokens": 12, "output_tokens": 8, "cache_creation_input_tokens": 0}))

        assert usage == Usage(prompt_tokens=12, completion_tokens=8, total_tokens=20, cache_creation_input_tokens=0)

    async def test_zero_cache_read_tokens_survive_as_zero(self) -> None:
        usage = await _usage_of(_message({"input_tokens": 12, "output_tokens": 8, "cache_read_input_tokens": 0}))

        assert usage == Usage(prompt_tokens=12, completion_tokens=8, total_tokens=20, cache_read_input_tokens=0)

    async def test_a_measured_zero_reaches_the_usage_report_as_a_zero(self) -> None:
        """A zero that reached ``Usage`` still has to reach the report: ``UsageReport``
        skips falsy usage, and a zero cache count is falsy on its own."""
        usage = await _usage_of(
            _message({
                "input_tokens": 12,
                "output_tokens": 8,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            })
        )
        report = UsageReport.from_events([UsageEvent(usage=usage, kind="model_call")])

        assert report.total == Usage(
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

    async def test_an_absent_cache_count_stays_absent(self) -> None:
        usage = await _usage_of(_message({"input_tokens": 12, "output_tokens": 8}))

        assert usage == Usage(prompt_tokens=12, completion_tokens=8, total_tokens=20)


async def test_request_counts_and_routing_labels_do_not_enter_usage() -> None:
    """``server_tool_use`` counts tool *requests*; ``service_tier`` and ``inference_geo``
    are labels. ``Usage`` carries token counts only, so the payload carrying them changes
    nothing."""
    payload = {
        "input_tokens": 12,
        "output_tokens": 8,
        "server_tool_use": {"web_search_requests": 3},
        "service_tier": "standard",
        "inference_geo": "us",
    }

    assert await _usage_of(_message(payload)) == Usage(prompt_tokens=12, completion_tokens=8, total_tokens=20)


class TestTheShapeTheApiSends:
    """Pinned to a body the live API returned, not one composed from the SDK's types.

    Probed 2026-09-19 against ``claude-opus-5``. Two things it settles, both of which
    the SDK's optional types leave open: Anthropic sends ``output_tokens_details`` even
    when nothing reasoned, carrying a measured ``0`` rather than omitting the object;
    and it sends both cache counts as a measured ``0`` on a call that used no caching.
    The second is why the presence-versus-truthiness fix is not an edge case — before
    it, every Anthropic call reported its cache counts as never measured.
    """

    @staticmethod
    def _live_usage(*, thinking_tokens: int, input_tokens: int, output_tokens: int) -> dict[str, Any]:
        return {
            "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "inference_geo": "global",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "output_tokens_details": {"thinking_tokens": thinking_tokens},
            "server_tool_use": None,
            "service_tier": "standard",
        }

    async def test_a_reasoning_call_reports_every_count_it_measured(self) -> None:
        usage = await _usage_of(_message(self._live_usage(thinking_tokens=45, input_tokens=36, output_tokens=269)))

        assert usage == Usage(
            prompt_tokens=36,
            completion_tokens=269,
            total_tokens=305,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            thinking_tokens=45,
        )

    async def test_thinking_disabled_still_carries_a_measured_zero(self) -> None:
        usage = await _usage_of(_message(self._live_usage(thinking_tokens=0, input_tokens=16, output_tokens=4)))

        assert usage == Usage(
            prompt_tokens=16,
            completion_tokens=4,
            total_tokens=20,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            thinking_tokens=0,
        )
