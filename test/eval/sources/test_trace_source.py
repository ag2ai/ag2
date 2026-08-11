# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for TraceSource backends (``ag2.eval.sources.trace_source``)."""

import pytest

from ag2._telemetry_consts import ATTR_SPAN_TYPE, SPAN_TYPE_AGENT, SPAN_TYPE_TOOL
from ag2.eval.sources._spans import SpanData
from ag2.eval.sources.trace_source import DirectoryTraceSource, InMemoryTraceSource, TraceRef, save_trace
from ag2.eval.trace import Trace
from ag2.events import ToolCallEvent, ToolResultEvent

_MS = 1_000_000


@pytest.mark.asyncio()
async def test_in_memory_source_lists_and_loads() -> None:
    t1 = Trace(events=[], exception=None, duration_ms=5)
    t2 = Trace(events=[], exception=None, duration_ms=7)
    source = InMemoryTraceSource([(TraceRef("a", task_id="task-a"), t1), (TraceRef("b"), t2)])

    refs = [ref async for ref in source.list()]

    assert [r.trace_id for r in refs] == ["a", "b"]
    assert refs[0].task_id == "task-a"
    assert refs[1].task_id is None
    assert (await source.load(refs[0])).duration_ms == 5
    assert (await source.load(refs[1])).duration_ms == 7


@pytest.mark.asyncio()
async def test_directory_source_round_trip(tmp_path) -> None:
    spans = [
        SpanData("invoke_agent x", "root", None, 0, 300 * _MS, {ATTR_SPAN_TYPE: SPAN_TYPE_AGENT}),
        SpanData(
            "execute_tool get_weather",
            "tool-1",
            "root",
            10 * _MS,
            30 * _MS,
            {
                ATTR_SPAN_TYPE: SPAN_TYPE_TOOL,
                "gen_ai.tool.name": "get_weather",
                "gen_ai.tool.call.id": "c1",
                "gen_ai.tool.call.arguments": '{"city": "NYC"}',
                "gen_ai.tool.call.result": "sunny",
            },
        ),
    ]
    save_trace(tmp_path, "trace-1", spans, task_id="task-1", metadata={"k": "v"})

    source = DirectoryTraceSource(tmp_path)
    refs = [ref async for ref in source.list()]

    assert len(refs) == 1
    assert refs[0] == TraceRef(trace_id="trace-1", task_id="task-1", metadata={"k": "v"})

    trace = await source.load(refs[0])
    assert [c.name for c in trace.events_of(ToolCallEvent)] == ["get_weather"]
    assert trace.events_of(ToolCallEvent)[0].arguments == '{"city": "NYC"}'
    assert len(trace.events_of(ToolResultEvent)) == 1
    assert trace.duration_ms == 300
