# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OTLP-JSON -> SpanData parser (``ag2.eval.sources._otlp_json``).

The fixture mirrors the real shape returned by Tempo's ``GET /api/traces/{id}``:
``batches`` of ``scopeSpans``, attributes as ``{"key","value":{"<type>Value"}}``,
nanosecond times as strings, int attrs as strings, numeric/string status codes.
"""

from ag2.eval.sources._otlp_json import otlp_json_to_spans
from ag2.eval.sources._spans import spans_to_trace
from ag2.events import ModelResponse, ToolCallEvent, ToolErrorEvent

_OTLP = {
    "batches": [
        {
            "scopeSpans": [
                {
                    "spans": [
                        {
                            "name": "invoke_agent weather",
                            "spanId": "AAAA",
                            "startTimeUnixNano": "0",
                            "endTimeUnixNano": "500000000",
                            "attributes": [{"key": "ag2.span.type", "value": {"stringValue": "agent"}}],
                            "status": {},
                        }
                    ]
                }
            ]
        },
        {
            "scopeSpans": [
                {
                    "spans": [
                        {
                            "name": "chat mock",
                            "spanId": "BBBB",
                            "parentSpanId": "AAAA",
                            "startTimeUnixNano": "100000000",
                            "endTimeUnixNano": "150000000",
                            "attributes": [
                                {"key": "ag2.span.type", "value": {"stringValue": "llm"}},
                                {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "12"}},
                                {"key": "gen_ai.usage.output_tokens", "value": {"intValue": "7"}},
                                {
                                    "key": "gen_ai.output.messages",
                                    "value": {"stringValue": '[{"content": "Sunny in Paris.", "role": "assistant"}]'},
                                },
                            ],
                            "status": {"code": "STATUS_CODE_OK"},
                        },
                        {
                            "name": "execute_tool book",
                            "spanId": "CCCC",
                            "parentSpanId": "AAAA",
                            "startTimeUnixNano": "200000000",
                            "endTimeUnixNano": "220000000",
                            "attributes": [
                                {"key": "ag2.span.type", "value": {"stringValue": "tool"}},
                                {"key": "gen_ai.tool.name", "value": {"stringValue": "book"}},
                                {"key": "gen_ai.tool.call.id", "value": {"stringValue": "c1"}},
                                {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": "{}"}},
                            ],
                            "status": {"code": 2},
                            "events": [
                                {
                                    "name": "exception",
                                    "attributes": [{"key": "exception.message", "value": {"stringValue": "boom"}}],
                                }
                            ],
                        },
                    ]
                }
            ]
        },
    ]
}


def test_parses_attributes_status_and_events_across_batches() -> None:
    spans = otlp_json_to_spans(_OTLP)

    assert len(spans) == 3  # iterated all batches
    by_type = {s.attributes.get("ag2.span.type"): s for s in spans}

    assert by_type["agent"].parent_id is None  # empty parentSpanId -> None (root)
    assert by_type["llm"].attributes["gen_ai.usage.input_tokens"] == 12  # intValue string -> int
    assert by_type["tool"].status == "ERROR"  # numeric code 2 -> ERROR
    assert by_type["tool"].events[0].name == "exception"
    assert by_type["tool"].events[0].attributes["exception.message"] == "boom"


def test_reconstructs_trace_via_adapter() -> None:
    trace = spans_to_trace(otlp_json_to_spans(_OTLP))

    assert trace.tokens.input == 12
    assert trace.tokens.output == 7
    assert trace.events_of(ModelResponse)[0].content == "Sunny in Paris."
    assert [c.name for c in trace.events_of(ToolCallEvent)] == ["book"]
    assert len(trace.events_of(ToolErrorEvent)) == 1
    assert "boom" in str(trace.events_of(ToolErrorEvent)[0].error)
    assert trace.duration_ms == 500  # from the root agent span
