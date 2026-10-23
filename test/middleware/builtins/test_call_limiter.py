# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from ag2 import Agent, Context, MemoryStream
from ag2.events import (
    BaseEvent,
    CallLimitReached,
    DataInput,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResult,
    ToolResultEvent,
)
from ag2.middleware import BaseMiddleware, CallCounter, CallLimiter, LLMCall, Middleware, MiddlewareDescription
from ag2.middleware.builtin.call_limiter import CALL_LIMIT_ERROR_TYPE, CALL_LIMIT_KEY, TEXT_ONLY_PROMPT
from ag2.testing import TestConfig, TrackingConfig

SYNTHESIS = "You have used your call budget"


@dataclass
class Run:
    """What one turn did: tools that ran, results the model saw, limit events, model prompts."""

    ran: Counter[str] = field(default_factory=Counter)
    results: list[ToolResultEvent] = field(default_factory=list)
    limits: list[CallLimitReached] = field(default_factory=list)
    config: TrackingConfig | None = None

    def last_messages(self) -> list[BaseEvent]:
        assert self.config is not None
        return [c.args[0] for c in self.config.mock.call_args_list]

    def error_types(self) -> list[str | None]:
        return [_error(r).get("type") for r in self.results]

    async def record_limit(self, event: CallLimitReached) -> None:
        self.limits.append(event)

    def search(self) -> str:
        """Search."""
        self.ran["search"] += 1
        return "found"


def finish() -> ToolResult:
    return ToolResult("finished", final=True, metadata={"recorded": True})


class PromptRecorder(BaseMiddleware):
    def __init__(self, event: BaseEvent, context: Context, prompts: list[tuple[BaseEvent, ...]]) -> None:
        super().__init__(event, context)
        self.prompts = prompts

    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        self.prompts.append(tuple(events))
        return await call_next(events, context)


def _error(result: ToolResultEvent) -> dict[str, Any]:
    part = result.result.parts[0]
    return part.data.get("error", {}) if isinstance(part, DataInput) else {}


def _text(event: BaseEvent) -> str:
    assert isinstance(event, ModelRequest)
    return " ".join(p.content for p in event.parts if isinstance(p, TextInput))


def _call(name: str) -> ToolCallEvent:
    return ToolCallEvent(name=name)


async def _run(
    limiter: CallLimiter,
    *script: Any,
    run: Run | None = None,
    middleware: Sequence[Any] = (),
) -> tuple[Any, Run]:
    run = run or Run()
    config = TrackingConfig(TestConfig(*script, raise_tool_errors=False))
    run.config = config
    stream = MemoryStream()
    stream.where(CallLimitReached).subscribe(lambda e: run.limits.append(e))
    stream.where(ToolResultEvent).subscribe(lambda e: run.results.append(e))

    def search() -> str:
        """Search."""
        run.ran["search"] += 1
        return "found"

    def shell() -> str:
        """Run a command."""
        run.ran["shell"] += 1
        return "ok"

    def record() -> str:
        """Record the result."""
        run.ran["record"] += 1
        return "recorded"

    agent = Agent("limited", config=config, tools=[search, shell, record], middleware=[*middleware, limiter])
    reply = await agent.ask("Investigate.", stream=stream)
    return reply, run


@pytest.mark.asyncio()
class TestPerToolCaps:
    async def test_a_per_tool_cap_denies_the_tool_and_keeps_the_turn(self) -> None:
        reply, run = await _run(
            CallLimiter(max_tool_calls_per_tool={"search": 1}),
            _call("search"),
            _call("search"),
            _call("shell"),
            "done",
        )

        assert run.ran == {"search": 1, "shell": 1}
        assert run.error_types() == [None, CALL_LIMIT_ERROR_TYPE, None]
        assert "other tools remain available" in _error(run.results[1])["recovery"]
        assert _error(run.results[1])["retryable"] is False
        assert reply.body == "done"
        assert run.limits == []

    async def test_an_agent_that_keeps_retrying_a_capped_tool_still_records_its_artefact(self) -> None:
        """A per-tool cap that ended the turn lost a record call (measured downstream)."""
        reply, run = await _run(
            CallLimiter(max_tool_calls_per_tool={"shell": 1}, exempt={"record"}),
            _call("shell"),
            _call("shell"),
            _call("shell"),
            _call("record"),
            "anomaly map recorded",
        )

        assert run.ran == {"shell": 1, "record": 1}
        assert reply.body == "anomaly map recorded"


@pytest.mark.asyncio()
class TestTotals:
    async def test_a_total_cap_still_ends_the_turn(self) -> None:
        reply, run = await _run(
            CallLimiter(max_tool_calls=1),
            _call("search"),
            _call("shell"),
            "the answer",
        )

        assert run.ran == {"search": 1}
        assert run.error_types() == [None, CALL_LIMIT_ERROR_TYPE]
        assert "answer now" in _error(run.results[1])["recovery"][0]
        assert SYNTHESIS in _text(run.last_messages()[-1])
        assert reply.body == "the answer"
        assert reply.response.metadata[CALL_LIMIT_KEY] == {"reason": "max_tool_calls", "ending": "synthesis"}
        assert [(e.reason, e.ending) for e in run.limits] == [("max_tool_calls", "synthesis")]

    async def test_at_both_caps_the_total_wins(self) -> None:
        _, run = await _run(
            CallLimiter(max_tool_calls=1, max_tool_calls_per_tool={"search": 1}),
            _call("search"),
            _call("search"),
            "the answer",
        )

        assert "max_tool_calls)" in _error(run.results[1])["message"]
        assert [e.reason for e in run.limits] == ["max_tool_calls"]

    async def test_a_refused_call_spends_neither_total_nor_the_per_tool_count(self) -> None:
        counter = CallCounter()
        await _run(
            CallLimiter(max_tool_calls=5, max_tool_calls_per_tool={"search": 1}, counter=counter),
            _call("search"),
            _call("search"),
            _call("search"),
            "done",
        )

        assert counter.tool_calls == 1
        assert counter.per_tool == {"search": 1}
        assert counter.denied == 2
        assert counter.denied_per_tool == {"search": 2}

    async def test_a_refused_tool_call_is_in_the_trace_but_not_charged_to_the_budget(self) -> None:
        counter = CallCounter()
        reply, run = await _run(
            CallLimiter(max_tool_calls_per_tool={"search": 0}, counter=counter),
            _call("search"),
            "done",
        )

        history = await reply.context.stream.history.get_events()
        calls = [c.name for e in history if isinstance(e, ModelResponse) for c in e.tool_calls.calls]
        assert calls == ["search"]
        assert run.error_types() == [CALL_LIMIT_ERROR_TYPE]
        assert counter.tool_calls == 0
        assert run.ran == {}

    async def test_exempt_tools_are_never_counted_or_refused(self) -> None:
        counter = CallCounter()
        _, run = await _run(
            CallLimiter(max_tool_calls=1, exempt={"record"}, counter=counter),
            _call("record"),
            _call("record"),
            _call("search"),
            _call("record"),
            "done",
        )

        assert run.ran == {"record": 3, "search": 1}
        assert counter.tool_calls == 1
        assert run.limits == []

    async def test_a_used_up_model_call_cap_synthesises_before_dispatch(self) -> None:
        counter = CallCounter()
        reply, run = await _run(
            CallLimiter(max_llm_calls=2, counter=counter),
            _call("search"),
            _call("search"),
            "the answer",
        )

        assert run.ran == {"search": 2}
        assert SYNTHESIS in _text(run.last_messages()[-1])
        assert reply.body == "the answer"
        assert counter.llm_calls == 3


@pytest.mark.asyncio()
class TestEndings:
    async def test_an_exempt_final_tool_reports_the_exhausted_ending(self) -> None:
        run = Run()
        stream = MemoryStream()
        stream.where(CallLimitReached).subscribe(run.record_limit)
        agent = Agent(
            "limited",
            config=TestConfig(_call("finish")),
            tools=[finish],
            middleware=[CallLimiter(max_llm_calls=0, exempt={"finish"})],
        )

        reply = await agent.ask("Finish.", stream=stream)

        assert reply.body == "finished"
        assert reply.response.metadata == {
            "recorded": True,
            CALL_LIMIT_KEY: {"reason": "max_llm_calls", "ending": "synthesis"},
        }
        assert [(event.reason, event.ending) for event in run.limits] == [("max_llm_calls", "synthesis")]

    async def test_a_final_reply_of_only_refused_calls_twice_ends_with_the_signal(self) -> None:
        """The case this middleware exists for: the final call answered with tool calls only."""
        reply, run = await _run(
            CallLimiter(max_llm_calls=1),
            _call("search"),
            _call("search"),
            _call("shell"),
        )

        assert run.ran == {"search": 1}
        prompts = [_text(m) for m in run.last_messages()[1:]]
        assert SYNTHESIS in prompts[0]
        assert prompts[1] == TEXT_ONLY_PROMPT
        assert reply.body is None
        assert reply.response.finish_reason == "call_limit"
        assert [(e.reason, e.ending) for e in run.limits] == [("max_llm_calls", "empty")]

    async def test_a_final_reply_keeps_its_text_and_drops_its_tool_calls(self) -> None:
        reply, run = await _run(
            CallLimiter(max_llm_calls=1),
            _call("search"),
            ModelResponse(ModelMessage("partial answer"), tool_calls=ToolCallsEvent([_call("shell")])),
        )

        assert run.ran == {"search": 1}
        assert reply.body == "partial answer"
        assert not reply.response.tool_calls

    async def test_exempt_calls_in_the_final_reply_run_and_the_model_then_answers(self) -> None:
        reply, run = await _run(
            CallLimiter(max_llm_calls=1, exempt={"record"}),
            _call("search"),
            ModelResponse(tool_calls=ToolCallsEvent([_call("record"), _call("shell")])),
            "recorded, and the answer",
        )

        assert run.ran == {"search": 1, "record": 1}
        assert "record" in _text(run.last_messages()[1])
        assert SYNTHESIS in _text(run.last_messages()[2])
        assert reply.body == "recorded, and the answer"
        assert [e.ending for e in run.limits] == ["synthesis"]

    async def test_at_most_two_final_calls_are_made(self) -> None:
        counter = CallCounter()
        reply, run = await _run(
            CallLimiter(max_llm_calls=1, exempt={"record"}, counter=counter),
            _call("search"),
            _call("record"),
            _call("record"),
        )

        assert run.ran == {"search": 1, "record": 2}
        assert len(run.last_messages()) == 3
        assert (counter.llm_calls, counter.llm_denied) == (3, 1)
        assert reply.response.finish_reason == "call_limit"
        assert [e.ending for e in run.limits] == ["empty"]

    async def test_last_text_ending_is_opt_in(self) -> None:
        script = (
            ModelResponse(ModelMessage("Let me search."), tool_calls=ToolCallsEvent([_call("search")])),
            _call("search"),
        )

        empty, _ = await _run(CallLimiter(max_llm_calls=2, on_exhausted="stop"), *script)
        last, run = await _run(CallLimiter(max_llm_calls=2, on_exhausted="stop", ending="last_text"), *script)

        assert empty.body is None
        assert last.body == "Let me search."
        assert last.response.metadata[CALL_LIMIT_KEY] == {"reason": "max_llm_calls", "ending": "last_text"}
        assert last.response.finish_reason == "call_limit"
        assert [e.ending for e in run.limits] == ["last_text"]

    async def test_a_failed_final_call_ends_the_turn_and_reports_the_error(self) -> None:
        reply, run = await _run(
            CallLimiter(max_llm_calls=1),
            _call("search"),
            TimeoutError("gateway timed out"),
        )

        assert reply.response.finish_reason == "call_limit"
        assert [(e.ending, e.error_type, e.error) for e in run.limits] == [
            ("empty", "TimeoutError", "gateway timed out")
        ]

    async def test_a_cancelled_final_call_is_not_swallowed(self) -> None:
        with pytest.raises(asyncio.CancelledError):
            await _run(CallLimiter(max_llm_calls=1), _call("search"), asyncio.CancelledError())

    async def test_synthesis_text_only_and_empty_reply_prompts_reach_the_model(self) -> None:
        """Both prompts ride the call as the last request, and neither is stored in history."""
        reply, run = await _run(
            CallLimiter(max_llm_calls=1),
            _call("search"),
            ModelResponse(),
            "the answer",
        )

        assert reply.body is None
        history = await reply.context.stream.history.get_events()
        assert not any(SYNTHESIS in _text(e) for e in history if isinstance(e, ModelRequest))
        assert SYNTHESIS in _text(run.last_messages()[1])
        assert [e.ending for e in run.limits] == ["synthesis"]


@pytest.mark.asyncio()
class TestClock:
    async def test_wall_clock_expiry_allows_exactly_one_synthesis_call(self) -> None:
        now = [0.0]
        counter = CallCounter(clock=lambda: now[0])

        async def tick(call_next: Any, event: ToolCallEvent, context: Context) -> Any:
            result = await call_next(event, context)
            now[0] += 60
            return result

        reply, run = await _run(
            CallLimiter(max_seconds=30, counter=counter),
            _call("search"),
            "the answer",
            middleware=[Middleware(_Ticking, tick=tick)],
        )

        assert run.ran == {"search": 1}
        assert SYNTHESIS in _text(run.last_messages()[-1])
        assert reply.body == "the answer"
        assert [e.reason for e in run.limits] == ["max_seconds"]
        assert counter.llm_calls == 2

    async def test_wall_clock_expiry_under_stop_calls_the_model_no_further(self) -> None:
        now = [0.0]
        counter = CallCounter(clock=lambda: now[0])

        async def tick(call_next: Any, event: ToolCallEvent, context: Context) -> Any:
            result = await call_next(event, context)
            now[0] += 60
            return result

        reply, run = await _run(
            CallLimiter(max_seconds=30, on_exhausted="stop", counter=counter),
            _call("search"),
            middleware=[Middleware(_Ticking, tick=tick)],
        )

        assert len(run.last_messages()) == 1
        assert reply.response.finish_reason == "call_limit"
        assert counter.llm_denied == 1

    async def test_a_shared_counter_clock_runs_from_its_start(self) -> None:
        now = [100.0]
        counter = CallCounter(clock=lambda: now[0], started=40.0)

        reply, run = await _run(CallLimiter(max_seconds=30, on_exhausted="stop", counter=counter))

        assert run.last_messages() == []
        assert [e.reason for e in run.limits] == ["max_seconds"]
        assert reply.response.finish_reason == "call_limit"


class _Ticking(BaseMiddleware):
    def __init__(self, event: BaseEvent, context: Context, tick: Any) -> None:
        super().__init__(event, context)
        self._tick = tick

    async def on_tool_execution(self, call_next: Any, event: ToolCallEvent, context: Context) -> Any:
        return await self._tick(call_next, event, context)


@pytest.mark.asyncio()
class TestRemainingNote:
    async def test_remaining_notes_do_not_hide_the_last_text_ending(self) -> None:
        reply, _ = await _run(
            CallLimiter(max_llm_calls=2, show_remaining=True, on_exhausted="stop", ending="last_text"),
            ModelResponse(ModelMessage("Let me search."), tool_calls=ToolCallsEvent([_call("search")])),
            _call("search"),
        )

        assert reply.body == "Let me search."
        assert reply.response.metadata[CALL_LIMIT_KEY] == {"reason": "max_llm_calls", "ending": "last_text"}

    async def test_remaining_notes_are_replayed_as_part_of_the_cached_prefix(self) -> None:
        prompts: list[tuple[BaseEvent, ...]] = []
        run = Run(config=TrackingConfig(TestConfig(_call("search"), _call("search"), "done")))
        assert run.config is not None
        agent = Agent(
            "limited",
            config=run.config,
            tools=[run.search],
            middleware=[
                CallLimiter(max_tool_calls=3, max_llm_calls=5, show_remaining=True),
                Middleware(PromptRecorder, prompts=prompts),
            ],
        )

        reply = await agent.ask("Investigate.")

        notes = [_text(m) for m in run.last_messages()]
        assert notes == [
            "[call limit] Remaining: 3 of 3 tool calls; 4 of 5 model calls. Plan to answer before these run out.",
            "[call limit] Remaining: 2 of 3 tool calls; 3 of 5 model calls. Plan to answer before these run out.",
            "[call limit] Remaining: 1 of 3 tool calls; 2 of 5 model calls. Plan to answer before these run out.",
        ]
        history = await reply.context.stream.history.get_events()
        saved_notes = [_text(e) for e in history if isinstance(e, ModelRequest) and "[call limit]" in _text(e)]
        assert saved_notes == notes
        assert prompts[1][: len(prompts[0])] == prompts[0]
        assert prompts[2][: len(prompts[1])] == prompts[1]

    async def test_the_synthesis_call_spent_past_the_llm_cap_leaves_nothing_remaining(self) -> None:
        counter = CallCounter()
        _, run = await _run(
            CallLimiter(max_llm_calls=1, show_remaining=True, counter=counter),
            _call("search"),
            "the answer",
        )

        # The final call is spent past the cap, so nothing is left to report.
        assert counter.llm_calls == 2
        assert run.last_messages()[-1] != run.last_messages()[0]
        assert "0 of 1 model calls" in _text(run.last_messages()[0])


@pytest.mark.asyncio()
class TestSharing:
    async def test_stacked_policies_reserve_a_shared_counter_only_once(self) -> None:
        counter = CallCounter()
        _, run = await _run(
            CallLimiter(max_tool_calls_per_tool={"search": 2}, counter=counter),
            _call("search"),
            "done",
            middleware=[CallLimiter(max_tool_calls=5, counter=counter)],
        )

        assert run.ran == {"search": 1}
        assert (counter.tool_calls, counter.llm_calls, counter.per_tool) == (1, 2, {"search": 1})

    async def test_parallel_calls_reserve_the_budget_before_execution(self) -> None:
        counter = CallCounter()
        _, run = await _run(
            CallLimiter(max_tool_calls=2, counter=counter),
            [_call("search"), _call("search"), _call("search")],
            "done",
        )

        assert run.ran == {"search": 2}
        assert (counter.tool_calls, counter.denied) == (2, 1)

    async def test_delegated_agents_have_independent_limiter_groups(self) -> None:
        run = Run()
        child_counter, parent_counter = CallCounter(), CallCounter()
        child = Agent(
            "child",
            config=TestConfig(_call("search"), "child done"),
            tools=[run.search],
            middleware=[CallLimiter(max_tool_calls=1, counter=child_counter)],
        )
        parent = Agent(
            "parent",
            config=TestConfig(ToolCallEvent(name="task_child", arguments='{"objective": "search"}'), "done"),
            tools=[child.as_tool(description="Investigate.")],
            middleware=[CallLimiter(max_tool_calls_per_tool={"search": 0}, counter=parent_counter)],
        )

        reply = await parent.ask("Investigate.")

        assert reply.body == "done"
        assert run.ran == {"search": 1}
        assert parent_counter.per_tool == {"task_child": 1}
        assert child_counter.per_tool == {"search": 1}

    async def test_inner_refusals_do_not_spend_the_outer_tool_budget(self) -> None:
        outer, inner = CallCounter(), CallCounter()
        reply, run = await _run(
            CallLimiter(max_tool_calls_per_tool={"search": 0}, counter=inner),
            _call("search"),
            _call("search"),
            "done",
            middleware=[CallLimiter(max_tool_calls=10, counter=outer)],
        )

        assert reply.body == "done"
        assert run.ran == {}
        assert (outer.tool_calls, outer.denied) == (0, 2)
        assert (inner.tool_calls, inner.denied) == (0, 2)

    async def test_outer_counter_includes_each_inner_synthesis_call(self) -> None:
        outer, inner = CallCounter(), CallCounter()
        _, run = await _run(
            CallLimiter(max_llm_calls=0, counter=inner),
            _call("search"),
            _call("search"),
            middleware=[CallLimiter(max_llm_calls=10, counter=outer)],
        )

        assert len(run.last_messages()) == 2
        assert (outer.llm_calls, inner.llm_calls) == (2, 2)

    async def test_outer_model_cap_is_checked_before_an_inner_synthesis_retry(self) -> None:
        outer = CallCounter()
        reply, run = await _run(
            CallLimiter(max_llm_calls=0),
            _call("search"),
            _call("search"),
            middleware=[CallLimiter(max_llm_calls=1, on_exhausted="stop", counter=outer)],
        )

        assert len(run.last_messages()) == 1
        assert outer.llm_calls == 1
        assert reply.response.finish_reason == "call_limit"

    async def test_a_counter_in_dependencies_is_shared_across_turns(self) -> None:
        counter = CallCounter()
        limiter = CallLimiter(max_tool_calls=1, on_exhausted="stop", counter=lambda ctx: ctx.dependencies["calls"])

        # Each ask replays the script from the start.
        config = TestConfig(_call("search"), "done")
        ran: list[str] = []

        def search() -> str:
            """Search."""
            ran.append("search")
            return "found"

        agent = Agent("limited", config=config, tools=[search], middleware=[limiter])
        await agent.ask("one", dependencies={"calls": counter})
        reply = await agent.ask("two", dependencies={"calls": counter})

        assert ran == ["search"]
        assert counter.tool_calls == 1
        assert counter.denied == 1
        assert reply.response.finish_reason == "call_limit"

    async def test_a_default_counter_resets_every_turn(self) -> None:
        config = TestConfig(_call("search"), "done")
        ran: list[str] = []

        def search() -> str:
            """Search."""
            ran.append("search")
            return "found"

        agent = Agent("limited", config=config, tools=[search], middleware=[CallLimiter(max_tool_calls=1)])
        await agent.ask("one")
        await agent.ask("two")

        assert ran == ["search", "search"]

    async def test_an_inner_limiter_lets_the_outer_final_call_through(self) -> None:
        """A run-wide limiter outside a per-agent one: the outer's final call is not refused twice."""
        outer, inner = CallCounter(), CallCounter()
        reply, run = await _run(
            CallLimiter(max_llm_calls=2, counter=inner),
            _call("search"),
            _call("shell"),
            "the answer",
            middleware=[CallLimiter(max_tool_calls=1, counter=outer)],
        )

        # The outer limiter refuses the second tool and synthesises. The inner one is
        # at its own cap by then, and passes that final call through uncounted.
        assert run.ran == {"search": 1}
        assert reply.body == "the answer"
        assert SYNTHESIS in _text(run.last_messages()[-1])
        assert (outer.llm_calls, inner.llm_calls, inner.llm_denied) == (3, 2, 0)
        assert [e.reason for e in run.limits] == ["max_tool_calls"]


class TestConfiguration:
    @pytest.mark.parametrize(
        "options",
        [
            {"synthesis_prompt": "Summarise the evidence."},
            {"tool_limit_result": "This tool is unavailable."},
            {"exhausted_result": {"error": "Budget exhausted"}},
        ],
    )
    def test_custom_response_settings_are_not_reported_as_complete(self, options: dict[str, Any]) -> None:
        assert CallLimiter(**options).describe().complete is False

    def test_describe(self) -> None:
        assert CallLimiter(max_tool_calls=3, exempt={"b", "a"}).describe() == MiddlewareDescription(
            kind="CallLimiter",
            config={
                "max_tool_calls": 3,
                "max_tool_calls_per_tool": {},
                "max_llm_calls": None,
                "max_seconds": None,
                "exempt": ["a", "b"],
                "on_exhausted": "synthesise",
                "ending": "empty",
                "show_remaining": False,
                "shared_counter": False,
            },
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_tool_calls": -1},
            {"max_llm_calls": -1},
            {"max_tool_calls_per_tool": {"search": -1}},
            {"max_seconds": -1},
            {"on_exhausted": "hand_back"},
            {"ending": "canned"},
        ],
    )
    def test_invalid_configuration_is_rejected(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            CallLimiter(**kwargs)

    @pytest.mark.asyncio()
    async def test_refusal_results_are_configurable(self) -> None:
        _, run = await _run(
            CallLimiter(
                max_tool_calls=1,
                max_tool_calls_per_tool={"search": 0},
                tool_limit_result="no more searching",
                exhausted_result=lambda call: {"stop": call.name},
            ),
            _call("search"),
            _call("shell"),
            _call("shell"),
            "done",
        )

        results = [r.result.parts[0] for r in run.results]
        assert isinstance(results[0], TextInput) and results[0].content == "no more searching"
        assert isinstance(results[2], DataInput) and results[2].data == {"stop": "shell"}
