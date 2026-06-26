# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for TempoTraceSource (``ag2.eval.sources.tempo``) with mocked HTTP.

Uses ``httpx.MockTransport`` so the search/fetch contract is exercised without
a running Tempo. A live integration check lives in the agent_judge tryout.
"""

import httpx
import pytest

from ag2.eval.sources.tempo import TempoTraceSource
from ag2.events import ToolCallEvent

_TRACE_DOC = {
    "batches": [
        {
            "scopeSpans": [
                {
                    "spans": [
                        {
                            "name": "invoke_agent weather",
                            "spanId": "AAAA",
                            "startTimeUnixNano": "0",
                            "endTimeUnixNano": "300000000",
                            "attributes": [{"key": "ag2.span.type", "value": {"stringValue": "agent"}}],
                            "status": {},
                        },
                        {
                            "name": "execute_tool get_weather",
                            "spanId": "BBBB",
                            "parentSpanId": "AAAA",
                            "startTimeUnixNano": "100000000",
                            "endTimeUnixNano": "120000000",
                            "attributes": [
                                {"key": "ag2.span.type", "value": {"stringValue": "tool"}},
                                {"key": "gen_ai.tool.name", "value": {"stringValue": "get_weather"}},
                                {"key": "gen_ai.tool.call.id", "value": {"stringValue": "c1"}},
                                {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": '{"city": "Paris"}'}},
                                {"key": "gen_ai.tool.call.result", "value": {"stringValue": "Sunny"}},
                            ],
                            "status": {},
                        },
                    ]
                }
            ]
        }
    ]
}


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/search":
        return httpx.Response(200, json={"traces": [{"traceID": "abc123", "rootServiceName": "weather"}]})
    if request.url.path.startswith("/api/traces/"):
        assert request.url.path == "/api/traces/abc123"
        return httpx.Response(200, json=_TRACE_DOC)
    return httpx.Response(404)


@pytest.mark.asyncio()
async def test_tempo_lists_and_loads_traces() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        source = TempoTraceSource("http://tempo:3200", client=client)

        refs = [ref async for ref in source.list()]
        assert [r.trace_id for r in refs] == ["abc123"]
        assert refs[0].metadata["root_service"] == "weather"

        trace = await source.load(refs[0])
        assert [c.name for c in trace.events_of(ToolCallEvent)] == ["get_weather"]
        assert trace.events_of(ToolCallEvent)[0].arguments == '{"city": "Paris"}'
        assert trace.duration_ms == 300
