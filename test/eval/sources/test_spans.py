# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the span -> Trace adapter (``ag2.eval.sources._spans``)."""

import json
import logging

from ag2._telemetry_consts import (
    ATTR_HUMAN_INPUT_PROMPT,
    ATTR_HUMAN_INPUT_RESPONSE,
    ATTR_SPAN_TYPE,
    SPAN_TYPE_AGENT,
    SPAN_TYPE_HUMAN_INPUT,
    SPAN_TYPE_LLM,
    SPAN_TYPE_TOOL,
)
from ag2.eval.scorers import no_tool_errors, tool_called
from ag2.eval.sources._spans import (
    AG2GenAIConvention,
    OpenInferenceConvention,
    SpanData,
    SpanEvent,
    spans_to_trace,
)
from ag2.events import (
    HumanInputRequest,
    HumanMessage,
    ModelResponse,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
)

_MS = 1_000_000
_OI_KIND = "openinference.span.kind"


# AG2 gen_ai dialect span builders (ATTR_SPAN_TYPE + gen_ai.*).
def _agent_span(start_ns: int = 0, end_ns: int = 500 * _MS) -> SpanData:
    return SpanData(
        name="invoke_agent test",
        span_id="root",
        parent_id=None,
        start_ns=start_ns,
        end_ns=end_ns,
        attributes={ATTR_SPAN_TYPE: SPAN_TYPE_AGENT},
    )


def _llm_span(start_ns: int, *, content: str = "hello", in_tok: int = 10, out_tok: int = 5) -> SpanData:
    output = json.dumps([{"content": content, "role": "assistant"}])
    return SpanData(
        name="chat gpt-x",
        span_id=f"llm-{start_ns}",
        parent_id="root",
        start_ns=start_ns,
        end_ns=start_ns + 50 * _MS,
        attributes={
            ATTR_SPAN_TYPE: SPAN_TYPE_LLM,
            "gen_ai.usage.input_tokens": in_tok,
            "gen_ai.usage.output_tokens": out_tok,
            "gen_ai.output.messages": output,
            "gen_ai.response.model": "gpt-x",
            "gen_ai.response.finish_reasons": ["stop"],
        },
    )


def _tool_span(start_ns: int, *, name: str, call_id: str, args: str = "{}", result: str = "ok") -> SpanData:
    return SpanData(
        name=f"execute_tool {name}",
        span_id=f"tool-{call_id}",
        parent_id="root",
        start_ns=start_ns,
        end_ns=start_ns + 20 * _MS,
        attributes={
            ATTR_SPAN_TYPE: SPAN_TYPE_TOOL,
            "gen_ai.tool.name": name,
            "gen_ai.tool.call.id": call_id,
            "gen_ai.tool.call.arguments": args,
            "gen_ai.tool.call.result": result,
        },
    )


# OpenInference dialect span builders (openinference.span.kind + llm.*/tool.*).
def _oi_agent(start_ns: int = 0, end_ns: int = 500 * _MS) -> SpanData:
    return SpanData(
        name="Agent.run",
        span_id="root",
        parent_id=None,
        start_ns=start_ns,
        end_ns=end_ns,
        attributes={_OI_KIND: "AGENT"},
    )


def _oi_llm(
    start_ns: int, *, content: str = "hi", model: str = "gpt-4o-mini", in_tok: int = 10, out_tok: int = 5
) -> SpanData:
    return SpanData(
        name="OpenAIChat.invoke",
        span_id=f"oi-llm-{start_ns}",
        parent_id="root",
        start_ns=start_ns,
        end_ns=start_ns + 50 * _MS,
        attributes={
            _OI_KIND: "LLM",
            "llm.output_messages.0.message.content": content,
            "llm.model_name": model,
            "llm.provider": "OpenAI",
            "llm.token_count.prompt": in_tok,
            "llm.token_count.completion": out_tok,
        },
    )


def _oi_tool(
    start_ns: int,
    *,
    name: str = "get_weather",
    args: str = '{"city": "Paris"}',
    result: str = "Sunny",
    status: str = "OK",
) -> SpanData:
    return SpanData(
        name=name,
        span_id=f"oi-tool-{start_ns}",
        parent_id="root",
        start_ns=start_ns,
        end_ns=start_ns + 10 * _MS,
        attributes={_OI_KIND: "TOOL", "tool.name": name, "tool.parameters": args, "output.value": result},
        status=status,
    )


class TestAG2GenAIConvention:
    """The AG2 gen_ai dialect (ATTR_SPAN_TYPE + gen_ai.*) reconstructs into typed Trace events."""

    def test_llm_span_reconstructs_response_and_tokens(self) -> None:
        trace = spans_to_trace([_agent_span(), _llm_span(10 * _MS, in_tok=12, out_tok=7)])

        responses = trace.events_of(ModelResponse)
        assert len(responses) == 1
        assert responses[0].content == "hello"
        assert responses[0].finish_reason == "stop"
        assert trace.tokens.input == 12
        assert trace.tokens.output == 7
        assert trace.tokens.total == 19

    def test_tool_span_success_reconstructs_call_and_result(self) -> None:
        trace = spans_to_trace([
            _agent_span(),
            _tool_span(10 * _MS, name="get_weather", call_id="c1", args='{"city": "NYC"}'),
        ])

        calls = trace.events_of(ToolCallEvent, name="get_weather")
        assert len(calls) == 1
        assert calls[0].id == "c1"
        assert calls[0].arguments == '{"city": "NYC"}'
        assert len(trace.events_of(ToolResultEvent)) == 1
        assert len(trace.events_of(ToolErrorEvent)) == 0

        # The reconstruction is what the real prebuilt scorers see.
        assert tool_called("get_weather")._fn(trace=trace) is True
        assert no_tool_errors()._fn(trace=trace) is True

    def test_tool_span_error_reconstructs_tool_error_event(self) -> None:
        err_span = _tool_span(10 * _MS, name="flaky", call_id="c2")
        err_span = SpanData(
            name=err_span.name,
            span_id=err_span.span_id,
            parent_id=err_span.parent_id,
            start_ns=err_span.start_ns,
            end_ns=err_span.end_ns,
            attributes={k: v for k, v in err_span.attributes.items() if k != "gen_ai.tool.call.result"},
            status="ERROR",
            events=(SpanEvent("exception", {"exception.message": "boom"}),),
        )
        trace = spans_to_trace([_agent_span(), err_span])

        errors = trace.events_of(ToolErrorEvent)
        assert len(errors) == 1
        assert "boom" in str(errors[0].error)
        assert no_tool_errors()._fn(trace=trace) is False

    def test_human_input_span_reconstructs_request_and_message(self) -> None:
        human = SpanData(
            name="await_human_input",
            span_id="h1",
            parent_id="root",
            start_ns=10 * _MS,
            end_ns=20 * _MS,
            attributes={
                ATTR_SPAN_TYPE: SPAN_TYPE_HUMAN_INPUT,
                ATTR_HUMAN_INPUT_PROMPT: "approve?",
                ATTR_HUMAN_INPUT_RESPONSE: "yes",
            },
        )
        trace = spans_to_trace([_agent_span(), human])

        assert [e.content for e in trace.events_of(HumanInputRequest)] == ["approve?"]
        assert [e.content for e in trace.events_of(HumanMessage)] == ["yes"]


class TestOpenInferenceConvention:
    """The OpenInference dialect (openinference.span.kind + llm.*/tool.*) reconstructs into typed Trace events."""

    def test_llm_reconstructs_model_response(self) -> None:
        trace = spans_to_trace([
            _oi_agent(),
            _oi_llm(100 * _MS, content="The weather is sunny.", in_tok=62, out_tok=14),
        ])
        [resp] = trace.events_of(ModelResponse)
        assert resp.content == "The weather is sunny."
        assert resp.model == "gpt-4o-mini"
        assert trace.tokens.input == 62
        assert trace.tokens.output == 14

    def test_tool_reconstructs_call_and_result(self) -> None:
        trace = spans_to_trace([_oi_agent(), _oi_tool(100 * _MS, name="get_weather", args='{"city": "Paris"}')])
        [call] = trace.events_of(ToolCallEvent)
        assert call.name == "get_weather"
        assert call.arguments == '{"city": "Paris"}'
        assert len(trace.events_of(ToolResultEvent)) == 1
        # prebuilt scorers reach the right verdict on the reconstructed OpenInference trace
        assert tool_called("get_weather")._fn(trace=trace) is True
        assert no_tool_errors()._fn(trace=trace) is True


class TestConventionDispatch:
    """How ``spans_to_trace`` selects a reader across dialects (auto-detect, mix, override)."""

    def test_openinference_auto_detected_by_default(self) -> None:
        """No explicit conventions → spans_to_trace still recognizes OpenInference (auto-detect)."""
        trace = spans_to_trace([_oi_agent(), _oi_tool(100 * _MS), _oi_llm(200 * _MS, content="done")])
        assert [type(e).__name__ for e in trace.events] == ["ToolCallEvent", "ToolResultEvent", "ModelResponse"]

    def test_mixed_dialect_trace_uses_both_readers(self) -> None:
        """One trace with an AG2 span and an OpenInference span → both reconstruct (multiple readers, one trace)."""
        trace = spans_to_trace([_agent_span(), _llm_span(100 * _MS), _oi_tool(200 * _MS, name="lookup")])
        assert len(trace.events_of(ModelResponse)) == 1  # AG2 llm span
        assert [c.name for c in trace.events_of(ToolCallEvent)] == ["lookup"]  # OpenInference tool span

    def test_unrecognized_spans_warn_and_produce_no_events(self, caplog) -> None:
        """A span in no known dialect is skipped; an all-unrecognized trace warns instead of silently grading empty."""
        noise = SpanData(
            name="GET /api", span_id="x", parent_id=None, start_ns=0, end_ns=_MS, attributes={"http.method": "GET"}
        )
        with caplog.at_level(logging.WARNING):
            trace = spans_to_trace([noise])
        assert not trace.events
        assert "0 events" in caplog.text

    def test_pinning_a_single_convention_skips_other_dialects(self) -> None:
        """conventions=[AG2GenAIConvention()] ignores OpenInference spans (the override escape hatch)."""
        ag2_only = spans_to_trace([_oi_agent(), _oi_llm(100 * _MS)], conventions=[AG2GenAIConvention()])
        assert not ag2_only.events
        oi_only = spans_to_trace([_oi_agent(), _oi_llm(100 * _MS)], conventions=[OpenInferenceConvention()])
        assert len(oi_only.events_of(ModelResponse)) == 1


class TestTraceAssembly:
    """Dialect-independent envelope: event ordering, duration, and root-span exception."""

    def test_events_are_ordered_by_span_start_time(self) -> None:
        trace = spans_to_trace([
            _tool_span(300 * _MS, name="second", call_id="c2"),
            _agent_span(),
            _llm_span(100 * _MS),
            _tool_span(200 * _MS, name="first", call_id="c1"),
        ])

        kinds = [type(e).__name__ for e in trace.events]
        # llm(100) -> tool first(200): call+result -> tool second(300): call+result
        assert kinds == [
            "ModelResponse",
            "ToolCallEvent",
            "ToolResultEvent",
            "ToolCallEvent",
            "ToolResultEvent",
        ]
        assert [c.name for c in trace.events_of(ToolCallEvent)] == ["first", "second"]

    def test_duration_from_root_agent_span(self) -> None:
        trace = spans_to_trace([_agent_span(start_ns=0, end_ns=750 * _MS), _llm_span(100 * _MS)])
        assert trace.duration_ms == 750

    def test_explicit_duration_override(self) -> None:
        trace = spans_to_trace([_agent_span(end_ns=750 * _MS)], duration_ms=1234)
        assert trace.duration_ms == 1234

    def test_errored_agent_span_reconstructs_trace_exception(self) -> None:
        """A root agent span recorded with an exception → trace.exception, so crash detection survives."""
        errored = SpanData(
            name="invoke_agent test",
            span_id="root",
            parent_id=None,
            start_ns=0,
            end_ns=500 * _MS,
            attributes={ATTR_SPAN_TYPE: SPAN_TYPE_AGENT},
            status="ERROR",
            events=(SpanEvent("exception", {"exception.message": "boom"}),),
        )
        trace = spans_to_trace([errored, _llm_span(100 * _MS)])
        assert trace.exception is not None
        assert "boom" in str(trace.exception)

    def test_ok_agent_span_leaves_trace_exception_none(self) -> None:
        """A successful agent span → trace.exception is None (handled errors don't count as a crash)."""
        trace = spans_to_trace([_agent_span(), _llm_span(100 * _MS)])
        assert trace.exception is None
