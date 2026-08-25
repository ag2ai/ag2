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
    ATTR_USAGE_KIND,
    ATTR_USAGE_LABEL,
    ATTR_USAGE_TOTAL,
    SPAN_TYPE_AGENT,
    SPAN_TYPE_HUMAN_INPUT,
    SPAN_TYPE_LLM,
    SPAN_TYPE_TOOL,
    SPAN_TYPE_USAGE,
)
from ag2.eval.scorers import no_tool_errors, tool_called
from ag2.eval.sources._spans import (
    DEFAULT_CONVENTIONS,
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
    Usage,
    UsageEvent,
)

_MS = 1_000_000
_OI_KIND = "openinference.span.kind"


# AG2 gen_ai dialect span builders (ATTR_SPAN_TYPE + gen_ai.*).
class _NullConvention:
    """A caller's own reader that recognizes none of these spans."""

    def to_events(self, span: SpanData) -> list | None:
        return None


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


def _worker_agent_span(start_ns: int, *, agent_name: str) -> SpanData:
    """A nested ``invoke_agent`` span — a sub-agent that is instrumented too."""
    return SpanData(
        name=f"invoke_agent {agent_name}",
        span_id=f"agent-{agent_name}",
        parent_id="root",
        start_ns=start_ns,
        end_ns=start_ns + 100 * _MS,
        attributes={ATTR_SPAN_TYPE: SPAN_TYPE_AGENT, "gen_ai.agent.name": agent_name},
    )


def _usage_span(
    start_ns: int,
    *,
    kind: str,
    label: str | None = None,
    parent_id: str = "root",
    in_tok: int = 900,
    out_tok: int = 90,
    total: int = 990,
    cache_create: int = 7,
    cache_read: int = 3,
) -> SpanData:
    attributes: dict[str, object] = {
        ATTR_SPAN_TYPE: SPAN_TYPE_USAGE,
        ATTR_USAGE_KIND: kind,
        "gen_ai.usage.input_tokens": in_tok,
        "gen_ai.usage.output_tokens": out_tok,
        ATTR_USAGE_TOTAL: total,
    }
    if label is not None:
        attributes[ATTR_USAGE_LABEL] = label
    if cache_create:
        attributes["gen_ai.usage.cache_creation_input_tokens"] = cache_create
    if cache_read:
        attributes["gen_ai.usage.cache_read_input_tokens"] = cache_read
    return SpanData(
        name=f"record_usage {kind}",
        span_id=f"usage-{start_ns}",
        parent_id=parent_id,
        start_ns=start_ns,
        end_ns=start_ns,
        attributes=attributes,
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

    def test_usage_span_reconstructs_the_accounting_event(self) -> None:
        trace = spans_to_trace([_agent_span(), _usage_span(20 * _MS, kind="subtask", label="worker")])

        [usage_event] = trace.events_of(UsageEvent)
        assert usage_event.usage == Usage(
            prompt_tokens=900,
            completion_tokens=90,
            total_tokens=990,
            cache_creation_input_tokens=7,
            cache_read_input_tokens=3,
        )
        assert usage_event.kind == "subtask"
        assert usage_event.label == "worker"

    def test_usage_span_and_the_llm_span_it_accompanies_count_once(self) -> None:
        """A main-loop call produces both spans — exactly one accounting event."""
        trace = spans_to_trace([
            _agent_span(),
            _llm_span(10 * _MS, in_tok=12, out_tok=7),
            _usage_span(11 * _MS, kind="model_call", in_tok=12, out_tok=7, total=19, cache_create=0, cache_read=0),
        ])

        assert len(trace.events_of(UsageEvent)) == 1

    def test_llm_span_does_not_synthesize_when_the_trace_records_usage(self) -> None:
        """AG2 records usage as its own span, so synthesizing as well would double-count."""
        trace = spans_to_trace([
            _agent_span(),
            _llm_span(10 * _MS, in_tok=12, out_tok=7),
            _usage_span(11 * _MS, kind="model_call", in_tok=12, out_tok=7, total=19, cache_create=0, cache_read=0),
            _llm_span(30 * _MS, in_tok=1, out_tok=1),
            _usage_span(31 * _MS, kind="model_call", in_tok=1, out_tok=1, total=2, cache_create=0, cache_read=0),
        ])

        assert [int(e.usage.total_tokens or 0) for e in trace.events_of(UsageEvent)] == [19, 2]

    def test_an_instrumented_worker_is_not_counted_twice(self) -> None:
        """A rollup and the worker's own accounting are the same tokens.

        ``UsageReport``'s no-double-count invariant rests on a sub-agent's
        per-call events staying on its private stream. A span tree flattens
        every instrumented agent into one trace, so when the worker is
        instrumented too, both readings are present and the rollup is the
        redundant one.
        """
        trace = spans_to_trace([
            _agent_span(),
            _usage_span(10 * _MS, kind="model_call", in_tok=100, out_tok=10, total=110, cache_create=0, cache_read=0),
            _worker_agent_span(20 * _MS, agent_name="worker"),
            _usage_span(
                21 * _MS,
                kind="model_call",
                parent_id="agent-worker",
                in_tok=900,
                out_tok=90,
                total=990,
                cache_create=0,
                cache_read=0,
            ),
            _usage_span(
                22 * _MS,
                kind="subtask",
                label="worker",
                in_tok=900,
                out_tok=90,
                total=990,
                cache_create=0,
                cache_read=0,
            ),
        ])

        assert [e.kind for e in trace.events_of(UsageEvent)] == ["model_call", "model_call"]
        assert trace.tokens.total == 1100

    def test_a_rollup_is_kept_when_its_worker_is_not_instrumented(self) -> None:
        """The ordinary case — the rollup is the only record of the worker's spend."""
        trace = spans_to_trace([
            _agent_span(),
            _usage_span(10 * _MS, kind="model_call", in_tok=100, out_tok=10, total=110, cache_create=0, cache_read=0),
            _usage_span(
                20 * _MS,
                kind="subtask",
                label="worker",
                in_tok=900,
                out_tok=90,
                total=990,
                cache_create=0,
                cache_read=0,
            ),
        ])

        assert [e.kind for e in trace.events_of(UsageEvent)] == ["model_call", "subtask"]
        assert trace.tokens.total == 1100

    def test_supplying_the_default_conventions_reads_the_same_as_omitting_them(self) -> None:
        """``conventions=DEFAULT_CONVENTIONS`` reads as a no-op and must behave as one.

        Choosing a reader says what a *span* means; it does not make the same
        delegation bill twice. Gating rollup dedupe on ``conventions is None``
        double-counted every instrumented sub-agent for anyone who spelled the
        defaults out — or who added a reader of their own alongside them.
        """
        spans = [
            _agent_span(),
            _usage_span(10 * _MS, kind="model_call", in_tok=100, out_tok=10, total=110, cache_create=0, cache_read=0),
            _worker_agent_span(20 * _MS, agent_name="worker"),
            _usage_span(
                21 * _MS,
                kind="model_call",
                parent_id="agent-worker",
                in_tok=900,
                out_tok=90,
                total=990,
                cache_create=0,
                cache_read=0,
            ),
            _usage_span(
                22 * _MS,
                kind="subtask",
                label="worker",
                in_tok=900,
                out_tok=90,
                total=990,
                cache_create=0,
                cache_read=0,
            ),
        ]

        assert spans_to_trace(spans, conventions=DEFAULT_CONVENTIONS).tokens == spans_to_trace(spans).tokens
        assert spans_to_trace(spans, conventions=DEFAULT_CONVENTIONS).tokens.total == 1100

    def test_supplying_the_default_conventions_keeps_back_compat_synthesis(self) -> None:
        """Whether counts live on LLM spans is a fact about the trace, not a caller preference.

        ``DEFAULT_CONVENTIONS`` holds an ``AG2GenAIConvention`` built with
        synthesis off, so passing it — the natural way to extend the readers —
        reported **zero** for every archived trace.
        """
        spans = [_agent_span(), _llm_span(10 * _MS, in_tok=12, out_tok=7)]

        assert spans_to_trace(spans, conventions=DEFAULT_CONVENTIONS).tokens.total == 19
        assert spans_to_trace(spans, conventions=(*DEFAULT_CONVENTIONS, _NullConvention())).tokens.total == 19

    def test_a_nested_agent_that_produced_no_rollup_cancels_nothing(self) -> None:
        """An instrumented agent reached by a plain ``ask`` has no rollup of its own.

        ``as_tool``/``run_task`` roll a sub-agent's spend up onto the parent;
        ``await other.ask(...)`` from inside a tool does not. Matching a rollup
        on value alone let that agent's spend cancel an *unrelated*
        uninstrumented worker's rollup of the same size, losing those tokens.
        The rollup's label and the agent span's name identify who is whose.
        """
        trace = spans_to_trace([
            _agent_span(),
            _usage_span(10 * _MS, kind="model_call", in_tok=10, out_tok=1, total=11, cache_create=0, cache_read=0),
            # instrumented, invoked by plain ask() -> usage spans, no rollup
            _worker_agent_span(20 * _MS, agent_name="sidecar"),
            _usage_span(
                21 * _MS,
                kind="model_call",
                parent_id="agent-sidecar",
                in_tok=100,
                out_tok=10,
                total=110,
                cache_create=0,
                cache_read=0,
            ),
            # uninstrumented, delegated -> a rollup of exactly the same size
            _usage_span(
                30 * _MS,
                kind="subtask",
                label="blind",
                in_tok=100,
                out_tok=10,
                total=110,
                cache_create=0,
                cache_read=0,
            ),
        ])

        assert [e.kind for e in trace.events_of(UsageEvent)] == ["model_call", "model_call", "subtask"]
        assert trace.tokens.total == 231

    def test_an_unnamed_nested_agent_still_cancels_its_rollup_by_value(self) -> None:
        """``agent_name`` is optional; the span then says ``"unknown"``.

        That names nobody, so value-only matching remains the best available
        signal and the rollup is still recognized as the redundant reading.
        """
        trace = spans_to_trace([
            _agent_span(),
            _worker_agent_span(20 * _MS, agent_name="unknown"),
            _usage_span(
                21 * _MS,
                kind="model_call",
                parent_id="agent-unknown",
                in_tok=900,
                out_tok=90,
                total=990,
                cache_create=0,
                cache_read=0,
            ),
            _usage_span(
                22 * _MS,
                kind="subtask",
                label="worker",
                in_tok=900,
                out_tok=90,
                total=990,
                cache_create=0,
                cache_read=0,
            ),
        ])

        assert [e.kind for e in trace.events_of(UsageEvent)] == ["model_call"]
        assert trace.tokens.total == 990

    def test_trace_captured_before_usage_spans_still_reports_its_tokens(self) -> None:
        """Spans exported by an older AG2 carry counts on the LLM span alone.

        Reading only the accounting event would report zero for every archived
        trace, which is a regression rather than the intended correction.
        """
        trace = spans_to_trace([_agent_span(), _llm_span(10 * _MS, in_tok=12, out_tok=7)])

        [usage_event] = trace.events_of(UsageEvent)
        assert usage_event.usage == Usage(prompt_tokens=12, completion_tokens=7)
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

    def test_llm_synthesizes_an_accounting_event(self) -> None:
        """A foreign producer emits no ``UsageEvent``, so its per-call usage is synthesized.

        Without this the trace carries no accounting record at all, and it would
        report zero tokens once ``Trace.tokens`` reads that record — a regression
        for people running third-party instrumentors who changed nothing.
        """
        trace = spans_to_trace([_oi_agent(), _oi_llm(100 * _MS, in_tok=62, out_tok=14)])

        [usage_event] = trace.events_of(UsageEvent)
        assert usage_event.usage == Usage(prompt_tokens=62, completion_tokens=14)
        assert usage_event.kind == "model_call"
        assert usage_event.model == "gpt-4o-mini"
        assert usage_event.provider == "OpenAI"

    def test_llm_without_usage_synthesizes_nothing(self) -> None:
        """No counts declared is not the same as a call that cost nothing."""
        bare = SpanData(
            name="OpenAIChat.invoke",
            span_id="oi-llm-bare",
            parent_id="root",
            start_ns=100 * _MS,
            end_ns=150 * _MS,
            attributes={_OI_KIND: "LLM", "llm.model_name": "gpt-4o-mini"},
        )

        trace = spans_to_trace([_oi_agent(), bare])

        assert trace.events_of(UsageEvent) == ()
        assert len(trace.events_of(ModelResponse)) == 1


class TestConventionDispatch:
    """How ``spans_to_trace`` selects a reader across dialects (auto-detect, mix, override)."""

    def test_openinference_auto_detected_by_default(self) -> None:
        """No explicit conventions → spans_to_trace still recognizes OpenInference (auto-detect)."""
        trace = spans_to_trace([_oi_agent(), _oi_tool(100 * _MS), _oi_llm(200 * _MS, content="done")])
        assert [type(e).__name__ for e in trace.events] == [
            "ToolCallEvent",
            "ToolResultEvent",
            "ModelResponse",
            "UsageEvent",
        ]

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
        # llm(100), with its back-compat usage synthesized alongside (this trace
        # records no usage spans) -> tool first(200): call+result -> tool
        # second(300): call+result
        assert kinds == [
            "ModelResponse",
            "UsageEvent",
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
