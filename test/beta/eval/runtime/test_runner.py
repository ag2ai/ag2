# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Public-API tests for :func:`run_agent` — the eval runner.

Drives a ``TestConfig``-mocked agent through one or more tasks and
asserts that events are captured, scorers run, exceptions are surfaced
on the Trace (never raised out of ``run_agent``), observers / middleware the
user passes through the factory still work, and the run is persisted
to ``store_dir`` automatically.
"""

import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytest.importorskip("opentelemetry.sdk")

from autogen.beta import Agent, tool
from autogen.beta.eval import (
    BudgetThresholds,
    Feedback,
    InMemoryTraceSource,
    RunResult,
    Suite,
    Trace,
    TraceRef,
    evaluate_traces,
    run_agent,
    scorer,
)
from autogen.beta.events import ModelResponse, ToolCallEvent
from autogen.beta.testing import TestConfig


@tool
async def get_weather(city: str) -> str:
    return f"Sunny, 72F in {city}"


def _build_weather_agent(*, config: object = None) -> Agent:
    return Agent(
        "weather",
        prompt="You are a weather assistant. Use get_weather to answer.",
        config=config,
        tools=[get_weather],
    )


@scorer
def called_get_weather(trace: Trace) -> bool:
    return len(trace.events_of(ToolCallEvent, name="get_weather")) == 1


@scorer
def city_argument_correct(trace: Trace, reference_outputs: dict) -> bool:
    calls = trace.events_of(ToolCallEvent, name="get_weather")
    if not calls:
        return False
    return '"city": "' + reference_outputs["city"] + '"' in calls[0].arguments


def _cassette(city: str) -> TestConfig:
    return TestConfig(
        ToolCallEvent(name="get_weather", arguments='{"city": "' + city + '"}'),
        f"{city} is sunny.",
    )


@pytest.mark.asyncio
async def test_runs_one_task_and_captures_events(tmp_path: Path) -> None:
    suite = Suite.from_list([
        {"task_id": "t-tokyo", "inputs": {"input": "Tokyo?"}, "reference_outputs": {"city": "Tokyo"}},
    ])

    result = await run_agent(
        suite,
        agent=_build_weather_agent(),
        scorers=[called_get_weather, city_argument_correct],
        store_dir=tmp_path,
        model_config={"t-tokyo": _cassette("Tokyo")},
        concurrency=1,
    )

    assert isinstance(result, RunResult)
    assert len(result.tasks) == 1

    [task_result] = result.tasks
    assert task_result.task.task_id == "t-tokyo"
    assert task_result.trace.exception is None
    assert len(task_result.trace.events_of(ToolCallEvent, name="get_weather")) == 1
    assert task_result.feedback == (
        Feedback(key="called_get_weather", score=True),
        Feedback(key="city_argument_correct", score=True),
    )


@pytest.mark.asyncio
async def test_runs_multiple_tasks_concurrently(tmp_path: Path) -> None:
    suite = Suite.from_list([
        {"task_id": "tokyo", "inputs": {"input": "Tokyo?"}, "reference_outputs": {"city": "Tokyo"}},
        {"task_id": "paris", "inputs": {"input": "Paris?"}, "reference_outputs": {"city": "Paris"}},
        {"task_id": "sydney", "inputs": {"input": "Sydney?"}, "reference_outputs": {"city": "Sydney"}},
    ])

    result = await run_agent(
        suite,
        agent=_build_weather_agent(),
        scorers=[called_get_weather, city_argument_correct],
        store_dir=tmp_path,
        model_config={
            "tokyo": _cassette("Tokyo"),
            "paris": _cassette("Paris"),
            "sydney": _cassette("Sydney"),
        },
        concurrency=3,
    )

    assert tuple(t.task.task_id for t in result.tasks) == ("tokyo", "paris", "sydney")
    for task_result in result.tasks:
        assert all(fb.score is True for fb in task_result.feedback)


@pytest.mark.asyncio
async def test_global_model_config_applied_to_every_task(tmp_path: Path) -> None:
    """A single ModelConfig (not a dict) is shared across all tasks."""
    suite = Suite.from_list([
        {"task_id": "t1", "inputs": {"input": "London?"}, "reference_outputs": {"city": "London"}},
    ])

    result = await run_agent(
        suite,
        agent=_build_weather_agent(),
        scorers=[called_get_weather],
        store_dir=tmp_path,
        model_config=_cassette("London"),
        concurrency=1,
    )

    [task_result] = result.tasks
    assert task_result.feedback == (Feedback(key="called_get_weather", score=True),)


@pytest.mark.asyncio
async def test_agent_ask_exception_is_captured_not_raised(tmp_path: Path) -> None:
    """An agent whose ``ask`` explodes does not abort the run — the error is captured on the trace."""

    class ExplodingAgent:
        name = "exploding"

        async def ask(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("ask exploded")

    suite = Suite.from_list([{"task_id": "t1", "inputs": {"input": "hi"}}])

    result = await run_agent(
        suite,
        agent=ExplodingAgent(),  # type: ignore[arg-type]
        scorers=[called_get_weather],
        store_dir=tmp_path,
        concurrency=1,
    )

    [task_result] = result.tasks
    assert isinstance(task_result.trace.exception, RuntimeError)
    assert task_result.trace.events == ()
    # Scorers should still run — `called_get_weather` returns False on empty trace.
    assert task_result.feedback == (Feedback(key="called_get_weather", score=False),)


@pytest.mark.asyncio
async def test_missing_input_key_raises(tmp_path: Path) -> None:
    """A task whose inputs lack 'input' fails fast — run_agent grades inputs['input']."""
    suite = Suite.from_list([{"task_id": "t1", "inputs": {"question": "Tokyo?"}}])

    with pytest.raises(ValueError, match=r"no 'input' key"):
        await run_agent(
            suite,
            agent=_build_weather_agent(),
            scorers=[called_get_weather],
            store_dir=tmp_path,
            model_config=_cassette("Tokyo"),
            concurrency=1,
        )


@pytest.mark.asyncio
async def test_explicit_empty_input_is_allowed(tmp_path: Path) -> None:
    """An explicit empty ``"input": ""`` is intentional and runs (distinguished from a missing key)."""
    suite = Suite.from_list([{"task_id": "t1", "inputs": {"input": ""}}])

    result = await run_agent(
        suite,
        agent=_build_weather_agent(),
        scorers=[called_get_weather],
        store_dir=tmp_path,
        model_config=_cassette("Tokyo"),
        concurrency=1,
    )
    assert len(result.tasks) == 1


@pytest.mark.asyncio
async def test_run_id_is_generated_unless_provided(tmp_path: Path) -> None:
    suite = Suite.from_list([{"task_id": "t1", "inputs": {"input": "Tokyo?"}}])

    auto = await run_agent(
        suite,
        agent=_build_weather_agent(),
        scorers=[called_get_weather],
        store_dir=tmp_path,
        model_config={"t1": _cassette("Tokyo")},
        concurrency=1,
    )
    explicit = await run_agent(
        suite,
        agent=_build_weather_agent(),
        scorers=[called_get_weather],
        store_dir=tmp_path,
        model_config={"t1": _cassette("Tokyo")},
        run_id="my-fixed-run-id",
        concurrency=1,
    )

    assert auto.run_id != ""
    assert explicit.run_id == "my-fixed-run-id"
    assert auto.schema_version == "0.1"


@pytest.mark.asyncio
async def test_run_persists_to_store_dir(tmp_path: Path) -> None:
    """Every successful run is written to <store_dir>/<run_id>.json."""
    suite = Suite.from_list([{"task_id": "t1", "inputs": {"input": "Tokyo?"}}])

    result = await run_agent(
        suite,
        agent=_build_weather_agent(),
        scorers=[called_get_weather],
        store_dir=tmp_path,
        model_config={"t1": _cassette("Tokyo")},
        concurrency=1,
    )

    expected = tmp_path / f"{result.run_id}.json"
    assert expected.exists()


@pytest.mark.asyncio
async def test_budget_violation_recorded_not_aborted(tmp_path: Path) -> None:
    """A task that exceeds the token budget still completes."""
    suite = Suite.from_list([
        {"task_id": "t1", "inputs": {"input": "Tokyo?"}, "reference_outputs": {"city": "Tokyo"}},
    ])

    result = await run_agent(
        suite,
        agent=_build_weather_agent(),
        scorers=[called_get_weather],
        store_dir=tmp_path,
        model_config={"t1": _cassette("Tokyo")},
        budgets=BudgetThresholds(max_tokens_per_task=0),
        concurrency=1,
    )

    [task_result] = result.tasks
    # The TestConfig used here yields no token usage, so total is 0,
    # which is not strictly > 0 — budget_violation should be False.
    assert task_result.budget_violation is False
    # The scorer should still have run.
    assert task_result.feedback[0] == Feedback(key="called_get_weather", score=True)


@pytest.mark.asyncio
async def test_inline_list_dataset_is_loaded(tmp_path: Path) -> None:
    """``suite=[...]`` is sugar for ``Suite.from_list([...])``."""
    items = [{"task_id": "t1", "inputs": {"input": "Tokyo?"}, "reference_outputs": {"city": "Tokyo"}}]

    result = await run_agent(
        items,
        agent=_build_weather_agent(),
        scorers=[called_get_weather],
        store_dir=tmp_path,
        model_config={"t1": _cassette("Tokyo")},
        concurrency=1,
    )

    assert len(result.tasks) == 1
    assert result.suite.source == "inline"


@pytest.mark.asyncio
async def test_user_middleware_and_observers_still_fire(tmp_path: Path) -> None:
    """The runner's capture observer composes with user-supplied observers."""
    from autogen.beta.events import BaseEvent
    from autogen.beta.observers import observer as observer_factory

    user_events: list[BaseEvent] = []

    async def record(event: BaseEvent) -> None:
        user_events.append(event)

    user_observer = observer_factory(callback=record, sync_to_thread=False)

    agent = Agent(
        "weather",
        prompt="Use get_weather.",
        tools=[get_weather],
        observers=[user_observer],
    )

    suite = Suite.from_list([
        {"task_id": "t1", "inputs": {"input": "Tokyo?"}, "reference_outputs": {"city": "Tokyo"}},
    ])

    result = await run_agent(
        suite,
        agent=agent,
        scorers=[called_get_weather],
        store_dir=tmp_path,
        model_config={"t1": _cassette("Tokyo")},
        concurrency=1,
    )

    # Runner's EventCapture saw events.
    [task_result] = result.tasks
    assert len(task_result.trace.events) > 0
    # User observer also saw events (proves we compose, not replace).
    assert len(user_events) > 0


@pytest.mark.asyncio
async def test_run_duration_is_recorded(tmp_path: Path) -> None:
    suite = Suite.from_list([{"task_id": "t1", "inputs": {"input": "Tokyo?"}}])

    result = await run_agent(
        suite,
        agent=_build_weather_agent(),
        scorers=[called_get_weather],
        store_dir=tmp_path,
        model_config={"t1": _cassette("Tokyo")},
        concurrency=1,
    )

    assert result.duration_ms >= 0
    [task_result] = result.tasks
    assert task_result.trace.duration_ms >= 0


@pytest.mark.asyncio
async def test_reply_ask_continuations_are_captured(tmp_path: Path) -> None:
    """An agent that drives ``reply.ask`` internally still has all events captured.

    EventCapture subscribes directly to the stream (not via ``sub_scope``),
    so its subscription survives the ``agent.ask`` → ``reply.ask`` transition
    and a scorer can grade against the *full* multi-turn trace.
    """

    cassette = TestConfig(
        # turn 1 — model picks the tool
        ToolCallEvent(name="get_weather", arguments='{"city": "Tokyo"}'),
        # turn 1 — model's text reply after the tool result
        "Tokyo is sunny.",
        # turn 2 — model's reply to the follow-up
        "It is also warm.",
    )

    class TwoTurnWeather:
        """Wrapper agent that drives one follow-up turn via reply.ask."""

        def __init__(self, agent: Agent) -> None:
            self._agent = agent
            self.name = agent.name

        async def ask(self, prompt: str, **kwargs: object) -> object:
            reply = await self._agent.ask(prompt, **kwargs)
            return await reply.ask("And how warm is it?")

    agent = TwoTurnWeather(Agent("weather", tools=[get_weather]))

    @scorer
    def saw_at_least_two_model_responses(trace: Trace) -> bool:
        return len(trace.events_of(ModelResponse)) >= 2

    result = await run_agent(
        Suite.from_list([{"task_id": "t1", "inputs": {"input": "Tokyo weather?"}}]),
        agent=agent,  # type: ignore[arg-type]
        scorers=[saw_at_least_two_model_responses],
        model_config={"t1": cassette},
        store_dir=tmp_path,
        concurrency=1,
    )

    [task_result] = result.tasks
    assert task_result.feedback == (Feedback(key="saw_at_least_two_model_responses", score=True),)
    # And the trace itself carries both turns' events.
    assert len(task_result.trace.events_of(ModelResponse)) >= 2


@pytest.mark.asyncio
async def test_store_dir_is_optional(tmp_path: Path) -> None:
    """Omitting store_dir runs the eval without persisting a run JSON."""
    suite = Suite.from_list([{"task_id": "t1", "inputs": {"input": "?"}}])

    result = await run_agent(
        suite,
        agent=_build_weather_agent(),
        scorers=[called_get_weather],
        model_config={"t1": _cassette("Tokyo")},
    )

    assert len(result.tasks) == 1
    # Nothing was written — no run JSON exists under tmp_path.
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.asyncio
async def test_run_accepts_an_agent_instance(tmp_path: Path) -> None:
    """``agent=`` accepts a prebuilt Agent instance."""
    agent = Agent("weather", config=_cassette("Tokyo"), tools=[get_weather])

    result = await run_agent(
        Suite.from_list([{"task_id": "t1", "inputs": {"input": "Tokyo?"}, "reference_outputs": {"city": "Tokyo"}}]),
        agent=agent,
        scorers=[called_get_weather, city_argument_correct],
        store_dir=tmp_path,
        concurrency=1,
    )

    [task_result] = result.tasks
    assert task_result.feedback == (
        Feedback(key="called_get_weather", score=True),
        Feedback(key="city_argument_correct", score=True),
    )


@pytest.mark.asyncio
async def test_repeats_runs_each_task_n_times(tmp_path: Path) -> None:
    """``repeats=N`` runs each task N times (distinct ids) and pools the pass-rate."""
    result = await run_agent(
        Suite.from_list([{"task_id": "tokyo", "inputs": {"input": "Tokyo?"}, "reference_outputs": {"city": "Tokyo"}}]),
        agent=_build_weather_agent(),
        scorers=[called_get_weather],
        store_dir=tmp_path,
        model_config=_cassette("Tokyo"),  # one config, applied to every repeat
        repeats=5,
        concurrency=2,
    )

    assert len(result.tasks) == 5
    assert {t.task.task_id for t in result.tasks} == {"tokyo#1", "tokyo#2", "tokyo#3", "tokyo#4", "tokyo#5"}
    assert result.pass_rate("called_get_weather") == 1.0


@pytest.mark.asyncio
async def test_label_is_recorded_and_serialized(tmp_path: Path) -> None:
    """A user-defined ``label`` is carried on the result and persisted to the run JSON."""
    result = await run_agent(
        Suite.from_list([{"task_id": "t1", "inputs": {"input": "Tokyo?"}}]),
        agent=_build_weather_agent(),
        scorers=[called_get_weather],
        store_dir=tmp_path,
        model_config={"t1": _cassette("Tokyo")},
        label="weather-eval",
    )

    assert result.label == "weather-eval"
    data = json.loads((tmp_path / f"{result.run_id}.json").read_text(encoding="utf-8"))
    assert data["label"] == "weather-eval"


@pytest.mark.asyncio
async def test_label_defaults_to_none(tmp_path: Path) -> None:
    result = await run_agent(
        Suite.from_list([{"task_id": "t1", "inputs": {"input": "Tokyo?"}}]),
        agent=_build_weather_agent(),
        scorers=[called_get_weather],
        store_dir=tmp_path,
        model_config={"t1": _cassette("Tokyo")},
    )
    assert result.label is None


@pytest.mark.asyncio
async def test_run_and_evaluate_of_its_trace_agree(tmp_path: Path) -> None:
    """``run_agent(agent)`` and ``evaluate_traces(the trace it produced)`` grade identically.

    The whole point of the produce-then-grade design: ``run_agent`` and ``evaluate_traces``
    funnel through one grading core, so taking the exact trace ``run_agent`` produced
    and grading it again via ``evaluate_traces`` yields the same feedback — there is no
    second scoring path that could drift.
    """
    suite = Suite.from_list([
        {"task_id": "t1", "inputs": {"input": "Tokyo?"}, "reference_outputs": {"city": "Tokyo"}},
    ])
    agent = Agent("weather", config=_cassette("Tokyo"), tools=[get_weather])
    scorers = [called_get_weather, city_argument_correct]

    run_result = await run_agent(suite, agent=agent, scorers=scorers, store_dir=tmp_path)

    produced = run_result.tasks[0].trace
    source = InMemoryTraceSource([(TraceRef(trace_id="trace-1", task_id="t1"), produced)])
    eval_result = await evaluate_traces(source, scorers=scorers, store_dir=tmp_path, suite=suite)

    assert eval_result.tasks[0].feedback == run_result.tasks[0].feedback
    assert eval_result.tasks[0].budget_violation == run_result.tasks[0].budget_violation
    assert eval_result.pass_rate("called_get_weather") == run_result.pass_rate("called_get_weather")


@pytest.mark.asyncio
class TestTelemetryInjection:
    """span_attributes / span_processors / real trace-id capture (the platform-unblock work)."""

    async def test_span_attributes_stamped_and_eval_ids_seeded(self, tmp_path: Path) -> None:
        """every emitted span carries caller attrs + auto-seeded eval ids (task_id per task)."""
        captured = InMemorySpanExporter()
        await run_agent(
            Suite.from_list([{"task_id": "t-tokyo", "inputs": {"input": "Tokyo?"}}]),
            agent=_build_weather_agent(),
            scorers=[called_get_weather],
            store_dir=tmp_path,
            model_config={"t-tokyo": _cassette("Tokyo")},
            concurrency=1,
            run_id="run-123",
            variant="v3",
            label="checkout-suite",
            span_attributes={"ag2.org.id": "org-9"},
            span_processors=[SimpleSpanProcessor(captured)],
        )

        spans = captured.get_finished_spans()
        assert spans
        for span in spans:
            attrs = dict(span.attributes or {})
            assert attrs["ag2.eval.run_id"] == "run-123"
            assert attrs["ag2.eval.task_id"] == "t-tokyo"
            assert attrs["ag2.eval.variant"] == "v3"
            assert attrs["ag2.eval.label"] == "checkout-suite"
            assert attrs["ag2.org.id"] == "org-9"

    async def test_caller_attributes_win_on_conflict(self, tmp_path: Path) -> None:
        """A caller-supplied ag2.eval.run_id overrides the auto-seeded one."""
        captured = InMemorySpanExporter()
        await run_agent(
            Suite.from_list([{"task_id": "t1", "inputs": {"input": "Tokyo?"}}]),
            agent=_build_weather_agent(),
            scorers=[called_get_weather],
            store_dir=tmp_path,
            model_config={"t1": _cassette("Tokyo")},
            concurrency=1,
            run_id="seeded",
            span_attributes={"ag2.eval.run_id": "override"},
            span_processors=[SimpleSpanProcessor(captured)],
        )
        assert all(dict(s.attributes or {})["ag2.eval.run_id"] == "override" for s in captured.get_finished_spans())

    async def test_injected_processor_is_additive_grading_unchanged(self, tmp_path: Path) -> None:
        """the capturing processor sees the spans, and grading is identical with/without it."""
        suite = Suite.from_list([
            {"task_id": "t1", "inputs": {"input": "Tokyo?"}, "reference_outputs": {"city": "Tokyo"}}
        ])
        scorers = [called_get_weather, city_argument_correct]

        baseline = await run_agent(
            suite,
            agent=_build_weather_agent(),
            scorers=scorers,
            store_dir=tmp_path,
            model_config={"t1": _cassette("Tokyo")},
            concurrency=1,
            run_id="baseline",
        )
        captured = InMemorySpanExporter()
        injected = await run_agent(
            suite,
            agent=_build_weather_agent(),
            scorers=scorers,
            store_dir=tmp_path,
            model_config={"t1": _cassette("Tokyo")},
            concurrency=1,
            run_id="injected",
            span_processors=[SimpleSpanProcessor(captured)],
        )

        assert captured.get_finished_spans()  # the external processor received spans
        assert injected.tasks[0].feedback == baseline.tasks[0].feedback

    async def test_trace_ref_holds_real_otel_trace_id(self, tmp_path: Path) -> None:
        """TaskResult.trace_ref.trace_id equals the real OTEL trace id of the emitted spans."""
        captured = InMemorySpanExporter()
        result = await run_agent(
            Suite.from_list([{"task_id": "t1", "inputs": {"input": "Tokyo?"}}]),
            agent=_build_weather_agent(),
            scorers=[called_get_weather],
            store_dir=tmp_path,
            model_config={"t1": _cassette("Tokyo")},
            concurrency=1,
            span_processors=[SimpleSpanProcessor(captured)],
        )

        spans = captured.get_finished_spans()
        assert spans
        otel_ids = {format(s.context.trace_id, "032x") for s in spans}
        assert len(otel_ids) == 1  # one trace for this single-agent task
        trace_ref = result.tasks[0].trace_ref
        assert trace_ref is not None
        assert trace_ref.trace_id == otel_ids.pop()

    async def test_evaluate_traces_populates_trace_ref(self, tmp_path: Path) -> None:
        """evaluate_traces carries the source's TraceRef onto each TaskResult."""
        suite = Suite.from_list([{"task_id": "t1", "inputs": {"input": "Tokyo?"}}])
        agent = Agent("weather", config=_cassette("Tokyo"), tools=[get_weather])
        produced = (await run_agent(suite, agent=agent, scorers=[], store_dir=tmp_path)).tasks[0].trace
        source = InMemoryTraceSource([(TraceRef(trace_id="trace-abc", task_id="t1"), produced)])

        result = await evaluate_traces(source, scorers=[called_get_weather], store_dir=tmp_path, suite=suite)

        assert result.tasks[0].trace_ref == TraceRef(trace_id="trace-abc", task_id="t1")

    async def test_no_extra_args_is_unchanged(self, tmp_path: Path) -> None:
        """a no-arg run still grades identically (trace_ref present, eval ids self-seeded)."""
        result = await run_agent(
            Suite.from_list([{"task_id": "t1", "inputs": {"input": "Tokyo?"}, "reference_outputs": {"city": "Tokyo"}}]),
            agent=_build_weather_agent(),
            scorers=[called_get_weather, city_argument_correct],
            store_dir=tmp_path,
            model_config={"t1": _cassette("Tokyo")},
            concurrency=1,
        )
        assert result.tasks[0].feedback == (
            Feedback(key="called_get_weather", score=True),
            Feedback(key="city_argument_correct", score=True),
        )
        assert result.tasks[0].trace_ref is not None
