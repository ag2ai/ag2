# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the span -> Trace adapter (``autogen.beta.eval.sources._spans``)."""

import json

from autogen.beta._telemetry_consts import (
    ATTR_HUMAN_INPUT_PROMPT,
    ATTR_HUMAN_INPUT_RESPONSE,
    ATTR_SPAN_TYPE,
    SPAN_TYPE_AGENT,
    SPAN_TYPE_HUMAN_INPUT,
    SPAN_TYPE_LLM,
    SPAN_TYPE_TOOL,
)
from autogen.beta.eval.scorers import no_tool_errors, tool_called
from autogen.beta.eval.sources._spans import SpanData, SpanEvent, spans_to_trace
from autogen.beta.events import (
    HumanInputRequest,
    HumanMessage,
    ModelResponse,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
)

_MS = 1_000_000


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


def test_llm_span_reconstructs_response_and_tokens() -> None:
    trace = spans_to_trace([_agent_span(), _llm_span(10 * _MS, in_tok=12, out_tok=7)])

    responses = trace.events_of(ModelResponse)
    assert len(responses) == 1
    assert responses[0].content == "hello"
    assert responses[0].finish_reason == "stop"
    assert trace.tokens.input == 12
    assert trace.tokens.output == 7
    assert trace.tokens.total == 19


def test_tool_span_success_reconstructs_call_and_result() -> None:
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


def test_tool_span_error_reconstructs_tool_error_event() -> None:
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


def test_human_input_span_reconstructs_request_and_message() -> None:
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


def test_events_are_ordered_by_span_start_time() -> None:
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


def test_duration_from_root_agent_span() -> None:
    trace = spans_to_trace([_agent_span(start_ns=0, end_ns=750 * _MS), _llm_span(100 * _MS)])
    assert trace.duration_ms == 750


def test_explicit_duration_override() -> None:
    trace = spans_to_trace([_agent_span(end_ns=750 * _MS)], duration_ms=1234)
    assert trace.duration_ms == 1234


def test_errored_agent_span_reconstructs_trace_exception() -> None:
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


def test_ok_agent_span_leaves_trace_exception_none() -> None:
    """A successful agent span → trace.exception is None (handled errors don't count as a crash)."""
    trace = spans_to_trace([_agent_span(), _llm_span(100 * _MS)])
    assert trace.exception is None
