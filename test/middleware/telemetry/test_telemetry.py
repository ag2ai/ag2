# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from collections.abc import Sequence as SequenceType

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult, SpanExporter
from pydantic import BaseModel

from ag2 import Agent, Context
from ag2.agent import KnowledgeConfig
from ag2.compact import CompactTrigger, CompactionSummary, SummarizeCompact
from ag2.events import (
    BaseEvent,
    ImageInput,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolResultEvent,
    ToolResultsEvent,
    Usage,
    UsageEvent,
)
from ag2.knowledge import MemoryKnowledgeStore
from ag2.middleware import BaseMiddleware, Middleware
from ag2.middleware.builtin.telemetry import MAX_TOOL_RESULT_CHARS, TelemetryMiddleware, _build_input_messages
from ag2.stream import MemoryStream
from ag2.testing import TestConfig
from ag2.tools import ToolResult, tool


class _InMemorySpanExporter(SpanExporter):
    """In-memory span exporter for tests."""

    def __init__(self) -> None:
        self._spans: list[ReadableSpan] = []
        self._lock = threading.Lock()

    def export(self, spans: SequenceType[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self) -> list[ReadableSpan]:
        with self._lock:
            return list(self._spans)

    def shutdown(self) -> None:
        with self._lock:
            self._spans.clear()


@pytest.fixture()
def otel_setup():
    exporter = _InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider


@pytest.mark.asyncio()
async def test_turn_span_emitted(otel_setup):
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(ModelResponse(ModelMessage("Hello!"))),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    await agent.ask("Hi")

    spans = exporter.get_finished_spans()
    turn_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "agent"]
    assert len(turn_spans) == 1
    span = turn_spans[0]
    assert span.name == "invoke_agent assistant"
    assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert span.attributes["gen_ai.agent.name"] == "assistant"


@pytest.mark.asyncio()
async def test_llm_span_with_usage(otel_setup):
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                message=ModelMessage("Hi!"),
                usage=Usage(prompt_tokens=10, completion_tokens=5),
            ),
        ),
        middleware=[
            TelemetryMiddleware(
                tracer_provider=provider,
                agent_name="assistant",
                provider_name="openai",
                model_name="gpt-4o-mini",
            )
        ],
    )

    await agent.ask("Hello")

    spans = exporter.get_finished_spans()
    llm_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "llm"]
    assert len(llm_spans) == 1
    span = llm_spans[0]
    assert span.name == "chat gpt-4o-mini"
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4o-mini"
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
    assert span.attributes["gen_ai.usage.output_tokens"] == 5


class _UsageAfterTurn(BaseMiddleware):
    """Emits a ``UsageEvent`` after its ``call_next`` — as compaction does.

    History compaction and memory aggregation both do their work once the turn
    beneath them has finished, then surface only their usage onto the stream.
    This models that ordering without dragging a knowledge store into the test.
    """

    def __init__(self, event: BaseEvent, context: Context, *, usage: Usage, kind: str) -> None:
        super().__init__(event, context)
        self._usage = usage
        self._kind = kind

    async def on_turn(self, call_next, event: BaseEvent, context: Context):
        result = await call_next(event, context)
        await context.send(UsageEvent(self._usage, kind=self._kind))
        return result


def _usage_after_turn(usage: Usage, kind: str) -> Middleware:
    return Middleware(_UsageAfterTurn, usage=usage, kind=kind)


def _usage_spans(exporter: _InMemorySpanExporter) -> list[ReadableSpan]:
    """The accounting spans only — every test here asserts against these."""
    return [s for s in exporter.get_finished_spans() if s.attributes.get("ag2.span.type") == "usage"]


@pytest.mark.asyncio()
class TestUsageSpans:
    """``UsageEvent`` — the accounting record — is carried onto its own span.

    Spend that never produces an LLM span reaches a trace only this way: a
    sub-task rollup (the worker is not instrumented), history compaction and
    memory aggregation (they call the model outside the middleware hooks), and
    the live-session clients.
    """

    async def test_own_model_call_records_a_usage_span(self, otel_setup):
        exporter, provider = otel_setup

        agent = Agent(
            "assistant",
            config=TestConfig(
                ModelResponse(
                    message=ModelMessage("Hi!"),
                    usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                )
            ),
            middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
        )

        await agent.ask("Hello")

        [span] = _usage_spans(exporter)
        assert span.attributes["ag2.usage.kind"] == "model_call"
        assert span.attributes["gen_ai.usage.input_tokens"] == 10
        assert span.attributes["gen_ai.usage.output_tokens"] == 5
        assert span.attributes["ag2.usage.total_tokens"] == 15

    async def test_delegated_spend_is_captured_without_instrumenting_the_worker(self, otel_setup):
        """The rollup lands on the parent's stream, so the parent's telemetry sees it.

        Instrumenting the worker as well would put both its per-call events and
        the rollup in one trace and double-count them.
        """
        exporter, provider = otel_setup

        worker = Agent(
            "worker",
            config=TestConfig(
                ModelResponse(ModelMessage("researched"), usage=Usage(prompt_tokens=900, completion_tokens=90))
            ),
        )
        coordinator = Agent(
            "coordinator",
            config=TestConfig(
                ToolCallEvent(name="task_worker", arguments='{"objective": "research X"}'),
                ModelResponse(ModelMessage("done"), usage=Usage(prompt_tokens=100, completion_tokens=10)),
            ),
            tools=[worker.as_tool(description="Delegate research")],
            middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="coordinator")],
        )

        await coordinator.ask("Tell me about X")

        usage_spans = _usage_spans(exporter)
        by_kind = {s.attributes["ag2.usage.kind"]: s for s in usage_spans}
        assert set(by_kind) == {"model_call", "subtask"}
        assert by_kind["subtask"].attributes["ag2.usage.label"] == "worker"
        assert by_kind["subtask"].attributes["gen_ai.usage.input_tokens"] == 900
        assert by_kind["subtask"].attributes["gen_ai.usage.output_tokens"] == 90

    async def test_cache_counts_are_carried(self, otel_setup):
        exporter, provider = otel_setup

        agent = Agent(
            "assistant",
            config=TestConfig(
                ModelResponse(
                    message=ModelMessage("Hi!"),
                    usage=Usage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        cache_creation_input_tokens=7,
                        cache_read_input_tokens=3,
                    ),
                )
            ),
            middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
        )

        await agent.ask("Hello")

        [span] = _usage_spans(exporter)
        assert span.attributes["gen_ai.usage.cache_creation_input_tokens"] == 7
        assert span.attributes["gen_ai.usage.cache_read_input_tokens"] == 3

    async def test_each_run_records_its_own_spend_once(self, otel_setup):
        """A leaked subscription would re-record spend on every later run.

        Two runs on one shared stream spend once each, so two usage spans. A
        subscription surviving the first run would catch the second run's
        event as well and yield three.
        """
        exporter, provider = otel_setup
        stream = MemoryStream()

        agent = Agent(
            "assistant",
            config=TestConfig(ModelResponse(ModelMessage("ok"), usage=Usage(prompt_tokens=10, completion_tokens=1))),
            middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
        )

        await agent.ask("first", stream=stream)
        await agent.ask("second", stream=stream)

        usage_spans = _usage_spans(exporter)
        assert [s.attributes["gen_ai.usage.input_tokens"] for s in usage_spans] == [10, 10]

    async def test_uninstrumented_run_on_a_shared_stream_records_nothing(self, otel_setup):
        """The watcher outlives its turn, but must not outlive its *run*.

        An agent with no telemetry that reuses an instrumented agent's stream
        would otherwise have its spend recorded into the earlier run's finished
        trace — spend attributed to the wrong agent, in a trace that was already
        complete, from an ``ask`` the caller never asked to instrument.
        """
        exporter, provider = otel_setup
        stream = MemoryStream()
        usage = Usage(prompt_tokens=10, completion_tokens=1)

        instrumented = Agent(
            "instrumented",
            config=TestConfig(ModelResponse(ModelMessage("ok"), usage=usage)),
            middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="instrumented")],
        )
        plain = Agent("plain", config=TestConfig(ModelResponse(ModelMessage("ok"), usage=usage)))

        await instrumented.ask("first", stream=stream)
        await plain.ask("second", stream=stream)

        usage_spans = _usage_spans(exporter)
        assert [s.attributes["ag2.usage.kind"] for s in usage_spans] == ["model_call"]

    async def test_records_usage_emitted_by_outer_middleware(self, otel_setup):
        """History compaction reports its spend *after* the turn it followed.

        Compaction runs as agent-level middleware, so it emits after the inner
        middleware chain has unwound. Telemetry passed through ``ask`` sits
        inside it — which is exactly how the eval runner installs it — so a
        subscription torn down when the turn span closes would miss the spend
        the 0.2 accounting promises to count.
        """
        exporter, provider = otel_setup
        telemetry = TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")
        emitter = _usage_after_turn(Usage(prompt_tokens=500, completion_tokens=50), kind="compaction")

        agent = Agent(
            "assistant",
            config=TestConfig(ModelResponse(ModelMessage("hi"), usage=Usage(prompt_tokens=10, completion_tokens=1))),
            middleware=[emitter],
        )

        await agent.ask("go", middleware=[telemetry])

        usage_spans = _usage_spans(exporter)
        assert sorted(s.attributes["ag2.usage.kind"] for s in usage_spans) == ["compaction", "model_call"]

    async def test_late_usage_stays_in_the_turns_trace(self, otel_setup):
        """A span recorded after the turn closed must not start a new trace.

        Otherwise a backend that groups by trace id drops it, and the spend is
        invisible again for exactly the ingestion paths this exists to serve.
        """
        exporter, provider = otel_setup
        telemetry = TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")
        emitter = _usage_after_turn(Usage(prompt_tokens=500, completion_tokens=50), kind="compaction")

        agent = Agent(
            "assistant",
            config=TestConfig(ModelResponse(ModelMessage("hi"), usage=Usage(prompt_tokens=10, completion_tokens=1))),
            middleware=[emitter],
        )

        await agent.ask("go", middleware=[telemetry])

        trace_ids = {s.context.trace_id for s in exporter.get_finished_spans()}
        assert len(trace_ids) == 1

    async def test_two_instrumented_agents_on_one_stream_record_each_event_once(self, otel_setup):
        """Separate telemetry objects must not both record the same event.

        Two agents handed the same stream each install their own watcher. If
        those accumulate, the first agent's watcher also records the second's
        spend — into the first agent's already-closed trace, where it is both a
        duplicate and misattributed.
        """
        exporter, provider = otel_setup
        stream = MemoryStream()

        first = Agent(
            "first",
            config=TestConfig(ModelResponse(ModelMessage("a"), usage=Usage(prompt_tokens=10, completion_tokens=1))),
            middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="first")],
        )
        second = Agent(
            "second",
            config=TestConfig(ModelResponse(ModelMessage("b"), usage=Usage(prompt_tokens=20, completion_tokens=2))),
            middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="second")],
        )

        await first.ask("one", stream=stream)
        await second.ask("two", stream=stream)

        usage_spans = _usage_spans(exporter)
        assert sorted(s.attributes["gen_ai.usage.input_tokens"] for s in usage_spans) == [10, 20]

    async def test_a_fresh_telemetry_per_ask_does_not_accumulate_watchers(self, otel_setup):
        """Building the middleware per call is ordinary usage, not a leak."""
        exporter, provider = otel_setup
        stream = MemoryStream()

        agent = Agent(
            "assistant",
            config=TestConfig(ModelResponse(ModelMessage("ok"), usage=Usage(prompt_tokens=10, completion_tokens=1))),
        )

        for _ in range(3):
            await agent.ask("go", stream=stream, middleware=[TelemetryMiddleware(tracer_provider=provider)])

        usage_spans = _usage_spans(exporter)
        assert len(usage_spans) == 3

    async def test_no_telemetry_no_usage_spans(self, otel_setup):
        """Instrumentation stays opt-in — nothing subscribes when it is off."""
        exporter, _ = otel_setup

        agent = Agent(
            "assistant",
            config=TestConfig(ModelResponse(ModelMessage("Hi!"), usage=Usage(prompt_tokens=10, completion_tokens=5))),
        )

        await agent.ask("Hello")

        assert exporter.get_finished_spans() == []


@pytest.mark.asyncio()
async def test_tool_span(otel_setup):
    exporter, provider = otel_setup

    @tool
    def get_weather(city: str) -> str:
        """Get weather."""
        return f"Sunny in {city}"

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent(
                    calls=[ToolCallEvent(id="call_1", name="get_weather", arguments='{"city": "NYC"}')]
                ),
            ),
            ModelResponse(ModelMessage("It's sunny in NYC")),
        ),
        tools=[get_weather],
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant", capture_content=False)],
    )

    await agent.ask("Weather?")

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "tool"]
    assert len(tool_spans) == 1
    span = tool_spans[0]
    assert span.name == "execute_tool get_weather"
    assert span.attributes["gen_ai.tool.name"] == "get_weather"
    assert span.attributes["gen_ai.tool.call.id"] == "call_1"
    # capture_content=False explicitly, so no arguments attribute
    assert "gen_ai.tool.call.arguments" not in span.attributes


@pytest.mark.asyncio()
async def test_tool_span_with_content_capture(otel_setup):
    exporter, provider = otel_setup

    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello {name}"

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent(
                    calls=[ToolCallEvent(id="call_1", name="greet", arguments='{"name": "World"}')]
                ),
            ),
            ModelResponse(ModelMessage("Done")),
        ),
        tools=[greet],
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant", capture_content=True)],
    )

    await agent.ask("Greet")

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "tool"]
    assert len(tool_spans) == 1
    span = tool_spans[0]
    assert span.attributes["gen_ai.tool.call.arguments"] == '{"name": "World"}'
    assert "Hello World" in span.attributes["gen_ai.tool.call.result"]


def _tool_span_of(exporter):
    spans = [s for s in exporter.get_finished_spans() if s.attributes.get("ag2.span.type") == "tool"]
    assert len(spans) == 1
    return spans[0]


def _agent_calling(tool_fn, provider, *, capture_content=True, **telemetry_kwargs):
    return Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent([ToolCallEvent(id="call_1", name=tool_fn.name, arguments="{}")]),
            ),
            ModelResponse(ModelMessage("Done")),
        ),
        tools=[tool_fn],
        middleware=[
            TelemetryMiddleware(
                tracer_provider=provider,
                agent_name="assistant",
                capture_content=capture_content,
                **telemetry_kwargs,
            )
        ],
    )


@pytest.mark.asyncio()
async def test_tool_span_records_dict_result_as_json(otel_setup):
    exporter, provider = otel_setup

    @tool
    def weather() -> dict:
        """Report the weather."""
        return {"city": "Oslo", "temp_c": 12, "conditions": ["cloud", "rain"]}

    await _agent_calling(weather, provider).ask("Weather?")

    recorded = _tool_span_of(exporter).attributes["gen_ai.tool.call.result"]
    assert json.loads(recorded) == {"city": "Oslo", "temp_c": 12, "conditions": ["cloud", "rain"]}


@pytest.mark.asyncio()
async def test_tool_span_records_pydantic_result_as_json(otel_setup):
    exporter, provider = otel_setup

    class Report(BaseModel):
        city: str
        temp_c: int

    @tool
    def weather() -> Report:
        """Report the weather."""
        return Report(city="Oslo", temp_c=12)

    await _agent_calling(weather, provider).ask("Weather?")

    recorded = _tool_span_of(exporter).attributes["gen_ai.tool.call.result"]
    assert json.loads(recorded) == {"city": "Oslo", "temp_c": 12}


@pytest.mark.asyncio()
async def test_tool_span_keeps_every_part_of_a_multi_part_result(otel_setup):
    exporter, provider = otel_setup

    @tool
    def summarise() -> ToolResult:
        """Summarise, in parts."""
        return ToolResult("headline", {"rows": 2}, "footnote")

    await _agent_calling(summarise, provider).ask("Summarise")

    recorded = _tool_span_of(exporter).attributes["gen_ai.tool.call.result"]
    assert recorded.splitlines() == ["headline", '{"rows": 2}', "footnote"]


@pytest.mark.asyncio()
async def test_tool_span_truncates_and_flags_an_oversized_result(otel_setup):
    exporter, provider = otel_setup

    @tool
    def dump() -> dict:
        """Return far too much."""
        return {"rows": ["x" * 100 for _ in range(500)]}

    await _agent_calling(dump, provider).ask("Dump")

    span = _tool_span_of(exporter)
    recorded = span.attributes["gen_ai.tool.call.result"]
    assert len(recorded) == MAX_TOOL_RESULT_CHARS
    assert recorded.endswith("...[truncated]")
    assert span.attributes["ag2.tool.call.result.truncated"] is True


@pytest.mark.asyncio()
async def test_tool_span_result_cap_is_configurable_per_middleware(otel_setup):
    exporter, provider = otel_setup

    @tool
    def dump() -> dict:
        """Return far too much."""
        return {"rows": ["x" * 100 for _ in range(500)]}

    await _agent_calling(dump, provider, max_tool_result_chars=100_000).ask("Dump")

    span = _tool_span_of(exporter)
    recorded = span.attributes["gen_ai.tool.call.result"]
    assert len(recorded) > MAX_TOOL_RESULT_CHARS
    assert json.loads(recorded) == {"rows": ["x" * 100 for _ in range(500)]}
    assert "ag2.tool.call.result.truncated" not in span.attributes


@pytest.mark.asyncio()
async def test_tool_span_result_cap_none_disables_truncation(otel_setup):
    exporter, provider = otel_setup

    @tool
    def dump() -> str:
        """Return far too much."""
        return "y" * (MAX_TOOL_RESULT_CHARS * 3)

    await _agent_calling(dump, provider, max_tool_result_chars=None).ask("Dump")

    span = _tool_span_of(exporter)
    assert span.attributes["gen_ai.tool.call.result"] == "y" * (MAX_TOOL_RESULT_CHARS * 3)
    assert "ag2.tool.call.result.truncated" not in span.attributes


def test_describe_reports_tool_result_cap():
    assert TelemetryMiddleware().describe().config["max_tool_result_chars"] == MAX_TOOL_RESULT_CHARS
    assert TelemetryMiddleware(max_tool_result_chars=None).describe().config["max_tool_result_chars"] is None


@pytest.mark.asyncio()
async def test_tool_span_omits_structured_result_without_content_capture(otel_setup):
    exporter, provider = otel_setup

    @tool
    def weather() -> dict:
        """Report the weather."""
        return {"city": "Oslo"}

    await _agent_calling(weather, provider, capture_content=False).ask("Weather?")

    attributes = _tool_span_of(exporter).attributes
    assert "gen_ai.tool.call.result" not in attributes
    assert "ag2.tool.call.result.truncated" not in attributes


@pytest.mark.asyncio()
async def test_tool_span_records_single_text_result_verbatim(otel_setup):
    """A lone text part stays an unwrapped plain string."""
    exporter, provider = otel_setup

    @tool
    def greet() -> str:
        """Greet."""
        return "Hello World"

    await _agent_calling(greet, provider).ask("Greet")

    span = _tool_span_of(exporter)
    assert span.attributes["gen_ai.tool.call.result"] == "Hello World"
    assert "ag2.tool.call.result.truncated" not in span.attributes


@pytest.mark.asyncio()
async def test_tool_error_marks_span_error(otel_setup):
    exporter, provider = otel_setup

    @tool
    def fail_tool() -> str:
        """Always fails."""
        raise ValueError("something went wrong")

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent([ToolCallEvent(id="call_1", name="fail_tool", arguments="{}")]),
            ),
            ModelResponse(ModelMessage("Error handled")),
        ),
        tools=[fail_tool],
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    # TestClient re-raises ToolError.error on the next LLM call, so we expect ValueError
    with pytest.raises(ValueError, match="something went wrong"):
        await agent.ask("Do it")

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "tool"]
    assert len(tool_spans) == 1
    span = tool_spans[0]
    assert span.status.status_code.name == "ERROR"


@pytest.mark.asyncio()
async def test_span_parent_child_hierarchy(otel_setup):
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(ModelMessage("Hi!"), usage=Usage(prompt_tokens=5, completion_tokens=3)),
        ),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    await agent.ask("Hello")

    spans = exporter.get_finished_spans()
    turn_span = next(s for s in spans if s.attributes.get("ag2.span.type") == "agent")
    llm_span = next(s for s in spans if s.attributes.get("ag2.span.type") == "llm")

    # LLM span should be a child of the turn span
    assert llm_span.parent is not None
    assert llm_span.parent.span_id == turn_span.context.span_id


@pytest.mark.asyncio()
async def test_capture_content_false_omits_messages(otel_setup):
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(ModelMessage("Secret response")),
        ),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant", capture_content=False)],
    )

    await agent.ask("Secret question")

    spans = exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.attributes.get("ag2.span.type") == "llm")
    assert "gen_ai.input.messages" not in llm_span.attributes
    assert "gen_ai.output.messages" not in llm_span.attributes


@pytest.mark.asyncio()
async def test_capture_content_true_includes_messages(otel_setup):
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(ModelMessage("Hello!")),
        ),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant", capture_content=True)],
    )

    await agent.ask("Hi")

    spans = exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.attributes.get("ag2.span.type") == "llm")
    assert "gen_ai.input.messages" in llm_span.attributes
    assert "gen_ai.output.messages" in llm_span.attributes

    input_msgs = json.loads(llm_span.attributes["gen_ai.input.messages"])
    assert any("Hi" in str(m) for m in input_msgs)


def _llm_spans_in_order(exporter: _InMemorySpanExporter) -> list[ReadableSpan]:
    spans = [s for s in exporter.get_finished_spans() if s.attributes.get("ag2.span.type") == "llm"]
    return sorted(spans, key=lambda s: s.start_time)


def _messages(span: ReadableSpan, attribute: str) -> list[dict]:
    return json.loads(span.attributes[attribute])


def _weather_agent(provider, *, reply: str, **telemetry_kwargs) -> Agent:
    """Agent scripted for one tool-calling turn: call ``get_weather``, then answer."""

    @tool
    def get_weather(city: str) -> str:
        """Return the current weather for a city."""
        return f"15°C and partly cloudy in {city}"

    return Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent([
                    ToolCallEvent(id="call_abc", name="get_weather", arguments='{"city": "Paris"}'),
                ]),
            ),
            ModelResponse(ModelMessage(reply)),
        ),
        tools=[get_weather],
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant", **telemetry_kwargs)],
    )


_USER_MSG = {"content": "What's the weather in Paris?", "role": "user"}
_TOOL_CALL_MSG = {
    "content": None,
    "role": "assistant",
    "tool_calls": [
        {"id": "call_abc", "type": "function", "function": {"arguments": '{"city": "Paris"}', "name": "get_weather"}},
    ],
}


@pytest.mark.asyncio()
async def test_tool_call_turn_records_tool_call_as_output_and_full_history_as_input(otel_setup):
    exporter, provider = otel_setup
    agent = _weather_agent(provider, reply="It's 15°C and partly cloudy in Paris.")

    await agent.ask("What's the weather in Paris?")

    first, second = _llm_spans_in_order(exporter)

    # The call that chose the tool: input is the user message, output the tool call.
    assert _messages(first, "gen_ai.input.messages") == [_USER_MSG]
    assert _messages(first, "gen_ai.output.messages") == [_TOOL_CALL_MSG]

    # The call after the tool ran: input replays everything the model saw.
    assert _messages(second, "gen_ai.input.messages") == [
        _USER_MSG,
        _TOOL_CALL_MSG,
        {"role": "tool", "tool_call_id": "call_abc", "content": "15°C and partly cloudy in Paris"},
    ]
    assert _messages(second, "gen_ai.output.messages") == [
        {"content": "It's 15°C and partly cloudy in Paris.", "role": "assistant"},
    ]


@pytest.mark.asyncio()
async def test_parallel_tool_calls_record_one_tool_message_per_result(otel_setup):
    exporter, provider = otel_setup

    @tool
    def get_weather(city: str) -> str:
        """Return the current weather for a city."""
        return f"18°C in {city}"

    @tool
    def get_current_time(timezone: str) -> str:
        """Return the local time in a timezone."""
        return f"22:15 in {timezone}"

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent([
                    ToolCallEvent(id="call_w", name="get_weather", arguments='{"city": "Paris"}'),
                    ToolCallEvent(id="call_t", name="get_current_time", arguments='{"timezone": "Asia/Tokyo"}'),
                ]),
            ),
            ModelResponse(ModelMessage("Paris is 18°C; it's 22:15 in Tokyo.")),
        ),
        tools=[get_weather, get_current_time],
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    await agent.ask("Weather in Paris and time in Tokyo?")

    first, second = _llm_spans_in_order(exporter)

    (assistant_out,) = _messages(first, "gen_ai.output.messages")
    assert [c["id"] for c in assistant_out["tool_calls"]] == ["call_w", "call_t"]

    history = _messages(second, "gen_ai.input.messages")
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert [c["id"] for c in history[1]["tool_calls"]] == ["call_w", "call_t"]
    tool_msgs = history[2:]
    assert all(m["role"] == "tool" for m in tool_msgs)
    # Parallel results may land in completion order; each result must still be paired to its call.
    assert {(m["tool_call_id"], m["content"]) for m in tool_msgs} == {
        ("call_w", "18°C in Paris"),
        ("call_t", "22:15 in Asia/Tokyo"),
    }


@pytest.mark.asyncio()
async def test_capture_content_false_omits_messages_on_tool_call_turn(otel_setup):
    exporter, provider = otel_setup
    agent = _weather_agent(provider, reply="Done", capture_content=False)

    await agent.ask("What's the weather in Paris?")

    llm_spans = _llm_spans_in_order(exporter)
    assert len(llm_spans) == 2
    for span in llm_spans:
        assert "gen_ai.input.messages" not in span.attributes
        assert "gen_ai.output.messages" not in span.attributes


@pytest.mark.asyncio()
async def test_input_messages_tool_result_honours_max_tool_result_chars(otel_setup):
    exporter, provider = otel_setup

    @tool
    def dump() -> str:
        """Return a large payload."""
        return "x" * 200

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(tool_calls=ToolCallsEvent([ToolCallEvent(id="call_1", name="dump", arguments="{}")])),
            ModelResponse(ModelMessage("Done")),
        ),
        tools=[dump],
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant", max_tool_result_chars=40)],
    )

    await agent.ask("Dump it")

    _, second = _llm_spans_in_order(exporter)
    tool_msg = _messages(second, "gen_ai.input.messages")[-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    assert len(tool_msg["content"]) == 40
    assert tool_msg["content"].endswith("...[truncated]")


@pytest.mark.asyncio()
async def test_input_messages_skip_binary_user_input(otel_setup):
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(ModelResponse(ModelMessage("A picture."))),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    await agent.ask("Describe this", ImageInput(data=b"\x89PNG", media_type="image/png"))

    (span,) = _llm_spans_in_order(exporter)
    assert _messages(span, "gen_ai.input.messages") == [{"content": "Describe this", "role": "user"}]


def test_build_input_messages_renders_tool_error_as_tool_message():
    call = ToolCallEvent(id="call_1", name="get_weather", arguments='{"city": "Tokyo"}')
    events = [
        ModelRequest([TextInput("And Tokyo?")]),
        ModelResponse(tool_calls=ToolCallsEvent([call])),
        ToolResultsEvent([ToolErrorEvent.from_call(call, RuntimeError("boom"))]),
    ]

    user, assistant, tool_error = _build_input_messages(events, None)

    assert user == {"content": "And Tokyo?", "role": "user"}
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "call_1"
    # A failed tool feeds its traceback back to the model, so the trace shows it too.
    assert tool_error["role"] == "tool"
    assert tool_error["tool_call_id"] == "call_1"
    assert "RuntimeError: boom" in tool_error["content"]


def _result_event(parent_id: str, text: str) -> ToolResultEvent:
    return ToolResultEvent(parent_id=parent_id, name="get_weather", result=ToolResult(text))


def test_build_input_messages_includes_compaction_summary():
    """Every provider mapper sends the summary, so the span must record it."""
    events = [
        CompactionSummary(summary="The user asked about Paris.", event_count=4),
        ModelRequest([TextInput("And Tokyo?")]),
    ]

    assert _build_input_messages(events, None) == [
        {"role": "user", "content": "[Summary of earlier conversation]\nThe user asked about Paris."},
        {"content": "And Tokyo?", "role": "user"},
    ]


def test_build_input_messages_falls_back_to_a_loose_tool_result():
    """Recovery path: anthropic, bedrock and zai send a result with no wrapper."""
    call = ToolCallEvent(id="call_1", name="get_weather", arguments='{"city": "Tokyo"}')
    events = [
        ModelRequest([TextInput("Tokyo?")]),
        ModelResponse(tool_calls=ToolCallsEvent([call])),
        _result_event("call_1", "22°C in Tokyo"),
    ]

    assert _build_input_messages(events, None)[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "22°C in Tokyo",
    }


def test_build_input_messages_records_a_wrapped_result_once():
    """History holds both forms; recording both would double every result."""
    call = ToolCallEvent(id="call_1", name="get_weather", arguments="{}")
    result = _result_event("call_1", "18°C")
    events = [
        ModelRequest([TextInput("Weather?")]),
        ModelResponse(tool_calls=ToolCallsEvent([call])),
        result,
        ToolResultsEvent([result]),
    ]

    messages = _build_input_messages(events, None)
    assert [m for m in messages if m["role"] == "tool"] == [
        {"role": "tool", "tool_call_id": "call_1", "content": "18°C"}
    ]


def test_build_input_messages_skips_a_loose_result_without_a_call_id():
    """An unpairable result is dropped, as the bedrock mapper drops it."""
    events = [ModelRequest([TextInput("Hi")]), _result_event("", "orphaned")]

    assert _build_input_messages(events, None) == [{"content": "Hi", "role": "user"}]


def test_build_input_messages_records_a_loose_tool_error():
    call = ToolCallEvent(id="call_1", name="get_weather", arguments="{}")
    events = [ModelResponse(tool_calls=ToolCallsEvent([call])), ToolErrorEvent.from_call(call, RuntimeError("boom"))]

    tool_message = _build_input_messages(events, None)[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert "RuntimeError: boom" in tool_message["content"]


@pytest.mark.asyncio()
async def test_compacted_history_reaches_the_span(otel_setup):
    """End to end: once history is summarised, the span carries the summary."""
    exporter, provider = otel_setup
    summarizer = TestConfig(ModelResponse(ModelMessage("Earlier: the user asked about Paris.")))

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(ModelMessage("First answer")),
            ModelResponse(ModelMessage("Second answer")),
            ModelResponse(ModelMessage("Third answer")),
        ),
        knowledge=KnowledgeConfig(
            store=MemoryKnowledgeStore(),
            compact=SummarizeCompact(target=1, config=summarizer),
            compact_trigger=CompactTrigger(max_events=2),
            expose_tool=False,
            write_event_log=False,
        ),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    reply = await agent.ask("What about Paris?")
    reply = await reply.ask("And Tokyo?")
    # The third turn is the first compacted before its call.
    await reply.ask("And Berlin?")

    inputs = [_messages(s, "gen_ai.input.messages") for s in _llm_spans_in_order(exporter)]
    summaries = [m for msgs in inputs for m in msgs if str(m.get("content", "")).startswith("[Summary of earlier")]
    assert summaries, f"no compaction summary recorded in any chat span: {inputs}"
    assert "Earlier: the user asked about Paris." in summaries[0]["content"]


@pytest.mark.asyncio()
async def test_auto_detect_model_provider_from_response(otel_setup):
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                message=ModelMessage("Hi!"),
                model="gpt-4o-mini-2024-07-18",
                provider="openai",
                finish_reason="stop",
                usage=Usage(prompt_tokens=10, completion_tokens=5),
            ),
        ),
        middleware=[
            TelemetryMiddleware(
                tracer_provider=provider,
                agent_name="assistant",
                # No provider_name or model_name — should auto-detect from response
            )
        ],
    )

    await agent.ask("Hello")

    spans = exporter.get_finished_spans()
    llm_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "llm"]
    assert len(llm_spans) == 1
    span = llm_spans[0]
    assert span.name == "chat gpt-4o-mini-2024-07-18"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4o-mini-2024-07-18"
    assert span.attributes["gen_ai.response.model"] == "gpt-4o-mini-2024-07-18"
    assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
    assert span.attributes["gen_ai.usage.output_tokens"] == 5


@pytest.mark.asyncio()
async def test_tool_span_has_tool_type(otel_setup):
    exporter, provider = otel_setup

    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello {name}"

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent(
                    calls=[ToolCallEvent(id="call_1", name="greet", arguments='{"name": "World"}')]
                ),
            ),
            ModelResponse(ModelMessage("Done")),
        ),
        tools=[greet],
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    await agent.ask("Greet")

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "tool"]
    assert len(tool_spans) == 1
    assert tool_spans[0].attributes["gen_ai.tool.type"] == "function"


@pytest.mark.asyncio()
async def test_constructor_params_override_response(otel_setup):
    """When constructor provides model_name/provider_name, those take precedence."""
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                message=ModelMessage("Hi!"),
                model="gpt-4o-mini-resolved",
                provider="openai",
                finish_reason="stop",
            ),
        ),
        middleware=[
            TelemetryMiddleware(
                tracer_provider=provider,
                agent_name="assistant",
                provider_name="custom-provider",
                model_name="custom-model",
            )
        ],
    )

    await agent.ask("Hello")

    spans = exporter.get_finished_spans()
    llm_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "llm"]
    span = llm_spans[0]
    # Constructor params win for request attributes
    assert span.attributes["gen_ai.provider.name"] == "custom-provider"
    assert span.attributes["gen_ai.request.model"] == "custom-model"
    # Response model still set
    assert span.attributes["gen_ai.response.model"] == "gpt-4o-mini-resolved"


@pytest.mark.asyncio()
async def test_cache_token_usage_attributes(otel_setup):
    """Cache creation/read token counts appear in LLM span attributes."""
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                message=ModelMessage("Hi!"),
                usage=Usage(
                    prompt_tokens=100,
                    completion_tokens=20,
                    cache_creation_input_tokens=80,
                    cache_read_input_tokens=0,
                ),
                model="claude-sonnet-5",
                provider="anthropic",
            ),
        ),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    await agent.ask("Hello")

    spans = exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.attributes.get("ag2.span.type") == "llm")
    assert llm_span.attributes["gen_ai.usage.input_tokens"] == 100
    assert llm_span.attributes["gen_ai.usage.output_tokens"] == 20
    assert llm_span.attributes["gen_ai.usage.cache_creation_input_tokens"] == 80
    # cache_read_input_tokens is 0, so it should NOT be set (guarded by `if usage.get(...)`)
    assert "gen_ai.usage.cache_read_input_tokens" not in llm_span.attributes


@pytest.mark.asyncio()
async def test_cache_read_tokens_when_nonzero(otel_setup):
    """cache_read_input_tokens appears when non-zero (simulates cache hit)."""
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                message=ModelMessage("Hi!"),
                usage=Usage(
                    prompt_tokens=100,
                    completion_tokens=20,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=75,
                ),
                model="claude-sonnet-5",
                provider="anthropic",
            ),
        ),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    await agent.ask("Hello")

    spans = exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.attributes.get("ag2.span.type") == "llm")
    assert llm_span.attributes["gen_ai.usage.cache_read_input_tokens"] == 75
    assert "gen_ai.usage.cache_creation_input_tokens" not in llm_span.attributes


@pytest.mark.asyncio()
async def test_thinking_tokens_when_nonzero(otel_setup):
    """thinking_tokens appears as gen_ai.usage.thinking_tokens when non-zero."""
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                message=ModelMessage("Hi!"),
                usage=Usage(
                    prompt_tokens=100,
                    completion_tokens=20,
                    thinking_tokens=296,
                ),
                model="gemini-3.1-pro-preview",
                provider="google",
            ),
        ),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    await agent.ask("Hello")

    spans = exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.attributes.get("ag2.span.type") == "llm")
    assert llm_span.attributes["gen_ai.usage.thinking_tokens"] == 296


@pytest.mark.asyncio()
async def test_span_attributes_stamped_on_all_spans(otel_setup):
    """Extra span_attributes are applied to every span type the middleware emits."""
    exporter, provider = otel_setup

    @tool
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent(calls=[ToolCallEvent(id="call_1", name="echo", arguments='{"msg": "hi"}')]),
            ),
            ModelResponse(ModelMessage("Done")),
        ),
        tools=[echo],
        middleware=[
            TelemetryMiddleware(
                tracer_provider=provider,
                agent_name="assistant",
                span_attributes={"ag2.org.id": "org-123", "deployment": "prod"},
            )
        ],
    )

    await agent.ask("Go")

    spans = exporter.get_finished_spans()
    for span in spans:
        assert span.attributes.get("ag2.org.id") == "org-123", f"missing on span {span.name!r}"
        assert span.attributes.get("deployment") == "prod", f"missing on span {span.name!r}"


@pytest.mark.asyncio()
async def test_intrinsic_attributes_win_on_collision(otel_setup):
    """Middleware-owned attributes overwrite span_attributes when keys collide."""
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(ModelResponse(ModelMessage("Hi!"))),
        middleware=[
            TelemetryMiddleware(
                tracer_provider=provider,
                agent_name="assistant",
                span_attributes={
                    "ag2.span.type": "SHOULD_BE_OVERWRITTEN",
                    "gen_ai.operation.name": "SHOULD_BE_OVERWRITTEN",
                    "gen_ai.agent.name": "SHOULD_BE_OVERWRITTEN",
                },
            )
        ],
    )

    await agent.ask("Hi")

    spans = exporter.get_finished_spans()
    agent_span = next(s for s in spans if s.attributes.get("ag2.span.type") == "agent")
    assert agent_span.attributes["ag2.span.type"] == "agent"
    assert agent_span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert agent_span.attributes["gen_ai.agent.name"] == "assistant"

    llm_span = next(s for s in spans if s.attributes.get("ag2.span.type") == "llm")
    assert llm_span.attributes["ag2.span.type"] == "llm"
    assert llm_span.attributes["gen_ai.operation.name"] == "chat"


@pytest.mark.asyncio()
async def test_span_attributes_omitted_when_not_provided(otel_setup):
    """No extra attributes appear when span_attributes is not passed."""
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(ModelResponse(ModelMessage("Hi!"))),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    await agent.ask("Hello")

    spans = exporter.get_finished_spans()
    for span in spans:
        assert "ag2.org.id" not in span.attributes


@pytest.mark.asyncio()
async def test_thinking_tokens_omitted_when_zero(otel_setup):
    """thinking_tokens=0 is treated as absent and not exported."""
    exporter, provider = otel_setup

    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                message=ModelMessage("Hi!"),
                usage=Usage(
                    prompt_tokens=100,
                    completion_tokens=20,
                    thinking_tokens=0,
                ),
                model="gemini-3.1-pro-preview",
                provider="google",
            ),
        ),
        middleware=[TelemetryMiddleware(tracer_provider=provider, agent_name="assistant")],
    )

    await agent.ask("Hello")

    spans = exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.attributes.get("ag2.span.type") == "llm")
    assert "gen_ai.usage.thinking_tokens" not in llm_span.attributes
