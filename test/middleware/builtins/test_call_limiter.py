# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from dirty_equals import IsPartialDict

from ag2 import Agent, Context, MemoryStream
from ag2.context import StreamId
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
    Usage,
    UsageEvent,
)
from ag2.history import MemoryStorage
from ag2.middleware import (
    BaseMiddleware,
    CallCounter,
    CallLimiter,
    ConditionalMiddleware,
    LLMCall,
    Middleware,
    MiddlewareDescription,
    RetryMiddleware,
)
from ag2.middleware.builtin.call_limiter import CALL_LIMIT_ERROR_TYPE, CALL_LIMIT_KEY, TEXT_ONLY_PROMPT
from ag2.testing import TestConfig, TrackingConfig

SYNTHESIS = "You have used your call budget"


@dataclass
class Run:
    """What one turn did: tools that ran, results the model saw, limit events, model prompts."""

    ran: Counter[str] = field(default_factory=Counter)
    results: list[ToolResultEvent] = field(default_factory=list)
    limits: list[CallLimitReached] = field(default_factory=list)
    response_metadata: list[dict[str, Any]] = field(default_factory=list)
    config: TrackingConfig | None = None

    def last_messages(self) -> list[BaseEvent]:
        assert self.config is not None
        return [c.args[0] for c in self.config.mock.call_args_list]

    def error_types(self) -> list[str | None]:
        return [_error(r).get("type") for r in self.results]

    async def record_limit(self, event: CallLimitReached) -> None:
        self.limits.append(event)

    async def record_result(self, event: ToolResultEvent) -> None:
        self.results.append(event)

    async def record_response(self, event: ModelResponse) -> None:
        self.response_metadata.append(dict(event.metadata))

    def search(self) -> str:
        """Search."""
        self.ran["search"] += 1
        return "found"

    def shell(self) -> str:
        """Run a command."""
        self.ran["shell"] += 1
        return "ok"

    def record(self) -> str:
        """Record the result."""
        self.ran["record"] += 1
        return "recorded"


def finish() -> ToolResult:
    return ToolResult("finished", final=True, metadata={"recorded": True})


async def immediate_finish() -> ToolResult:
    return ToolResult("finished", final=True, metadata={"recorded": True})


def search_and_enqueue(context: Context) -> str:
    """Search and add details to the ongoing turn."""
    context.enqueue("More details for the ongoing search.")
    return "found"


class PromptRecorder(BaseMiddleware):
    def __init__(self, event: BaseEvent, context: Context, prompts: list[tuple[BaseEvent, ...]]) -> None:
        super().__init__(event, context)
        self.prompts = prompts

    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        self.prompts.append(tuple(events))
        return await call_next(events, context)


class EnqueueAfterText:
    def __init__(self) -> None:
        self.sent = False

    async def __call__(self, event: ModelResponse, context: Context) -> None:
        if not self.sent and event.message is not None and event.message.content == "partial answer":
            self.sent = True
            context.enqueue("More details for this turn.")


class SnapshotStorage(MemoryStorage):
    async def get_history(self, stream_id: StreamId) -> list[BaseEvent]:
        return list(await super().get_history(stream_id))


class IteratorStorage(MemoryStorage):
    async def get_history(self, stream_id: StreamId) -> Iterable[BaseEvent]:
        return iter(list(await super().get_history(stream_id)))


def _counter_from_context(context: Context) -> CallCounter:
    return context.dependencies["calls"]


def _stop_result(call: ToolCallEvent) -> dict[str, Any]:
    return {"stop": call.name}


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
    stream.where(CallLimitReached).subscribe(run.record_limit)
    stream.where(ToolResultEvent).subscribe(run.record_result)

    agent = Agent(
        "limited", config=config, tools=[run.search, run.shell, run.record], middleware=[*middleware, limiter]
    )
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
    async def test_total_exhaustion_starts_synthesis_before_another_tool_is_requested(self) -> None:
        counter = CallCounter()
        reply, run = await _run(
            CallLimiter(max_tool_calls=1, exempt={"record"}, counter=counter),
            _call("search"),
            *[_call("record") for _ in range(5)],
            "done",
        )

        assert counter.llm_calls == 3
        assert run.ran == {"search": 1, "record": 2}
        assert reply.response.finish_reason == "call_limit"
        assert [e.reason for e in run.limits] == ["max_tool_calls"]

    async def test_a_total_cap_still_ends_the_turn(self) -> None:
        reply, run = await _run(
            CallLimiter(max_tool_calls=1),
            [_call("search"), _call("shell")],
            "the answer",
        )

        assert run.ran == {"search": 1}
        assert Counter(run.error_types()) == Counter([None, CALL_LIMIT_ERROR_TYPE])
        refused = next(r for r in run.results if _error(r))
        assert _error(refused) == IsPartialDict({"recovery": ["answer now from the information you already have"]})
        assert SYNTHESIS in _text(run.last_messages()[-1])
        assert reply.body == "the answer"
        assert reply.response.metadata[CALL_LIMIT_KEY] == {"reason": "max_tool_calls", "ending": "synthesis"}
        assert [(e.reason, e.ending) for e in run.limits] == [("max_tool_calls", "synthesis")]

    async def test_at_both_caps_the_total_wins(self) -> None:
        _, run = await _run(
            CallLimiter(max_tool_calls=1, max_tool_calls_per_tool={"search": 1}),
            [_call("search"), _call("search")],
            "the answer",
        )

        refused = next(r for r in run.results if _error(r))
        assert "max_tool_calls)" in _error(refused)["message"]
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
        assert [e.reason for e in run.limits] == ["max_tool_calls"]

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
    async def test_input_enqueued_after_text_keeps_the_last_text_in_the_turn(self) -> None:
        stream = MemoryStream()
        stream.where(ModelResponse).subscribe(EnqueueAfterText())
        agent = Agent(
            "limited",
            config=TestConfig("partial answer"),
            middleware=[CallLimiter(max_llm_calls=1, on_exhausted="stop", ending="last_text")],
        )

        reply = await agent.ask("Investigate.", stream=stream)

        assert reply.body == "partial answer"
        assert reply.response.finish_reason == "call_limit"
        assert reply.response.metadata[CALL_LIMIT_KEY] == {"reason": "max_llm_calls", "ending": "last_text"}

    async def test_last_text_is_not_reused_from_an_earlier_ask(self) -> None:
        stream = MemoryStream()
        counter = CallCounter()
        agent = Agent(
            "limited",
            config=TestConfig("earlier answer"),
            middleware=[CallLimiter(max_llm_calls=1, on_exhausted="stop", ending="last_text", counter=counter)],
        )
        await agent.ask("First question.", stream=stream)

        reply = await agent.ask("Next question.", stream=stream)

        assert reply.body is None
        assert reply.response.finish_reason == "call_limit"

    async def test_enqueued_input_does_not_hide_the_last_text_in_the_turn(self) -> None:
        agent = Agent(
            "limited",
            config=TestConfig(
                ModelResponse(
                    ModelMessage("Let me search."),
                    tool_calls=ToolCallsEvent([_call("search_and_enqueue")]),
                )
            ),
            tools=[search_and_enqueue],
            middleware=[CallLimiter(max_llm_calls=1, on_exhausted="stop", ending="last_text")],
        )

        reply = await agent.ask("Investigate.")

        assert reply.body == "Let me search."
        assert reply.response.metadata[CALL_LIMIT_KEY] == {"reason": "max_llm_calls", "ending": "last_text"}

    @pytest.mark.parametrize("finish_first", [False, True])
    async def test_total_refusal_and_final_tool_in_one_batch_announce_exhaustion(self, finish_first: bool) -> None:
        run = Run()
        counter = CallCounter()
        calls = [_call("search"), _call("search")]
        calls.insert(0 if finish_first else len(calls), _call("immediate_finish"))
        stream = MemoryStream()
        stream.where(CallLimitReached).subscribe(run.record_limit)
        stream.where(ModelResponse).subscribe(run.record_response)
        agent = Agent(
            "limited",
            config=TestConfig(calls),
            tools=[run.search, immediate_finish],
            middleware=[CallLimiter(max_tool_calls=1, exempt={"immediate_finish"}, counter=counter)],
        )

        reply = await agent.ask("Finish.", stream=stream)

        assert reply.body == "finished"
        assert counter.denied == 1
        assert reply.response.metadata == {
            "recorded": True,
            CALL_LIMIT_KEY: {"reason": "max_tool_calls", "ending": "final_tool"},
        }
        assert run.response_metadata[-1] == reply.response.metadata
        assert [(event.reason, event.ending) for event in run.limits] == [("max_tool_calls", "final_tool")]

    @pytest.mark.parametrize("on_exhausted", ["stop", "synthesise"])
    async def test_last_text_ending_accepts_iterator_storage(self, on_exhausted: str) -> None:
        run = Run()
        agent = Agent(
            "limited",
            config=TestConfig(
                ModelResponse(ModelMessage("Let me search."), tool_calls=ToolCallsEvent([_call("search")])),
                TimeoutError("final call failed"),
            ),
            tools=[run.search],
            middleware=[CallLimiter(max_llm_calls=1, on_exhausted=on_exhausted, ending="last_text")],
        )

        reply = await agent.ask("Investigate.", stream=MemoryStream(storage=IteratorStorage()))

        assert reply.body == "Let me search."
        assert reply.response.finish_reason == "call_limit"
        assert reply.response.metadata[CALL_LIMIT_KEY] == {"reason": "max_llm_calls", "ending": "last_text"}

    @pytest.mark.parametrize("ending", ["text", "tools", "error"])
    async def test_discarded_synthesis_usage_is_reported_once(self, ending: str) -> None:
        first = ModelResponse(
            tool_calls=ToolCallsEvent([_call("search")]),
            usage=Usage(total_tokens=15),
            model="test-model",
            provider="test-provider",
        )
        second = (
            TimeoutError("failed")
            if ending == "error"
            else ModelResponse(
                ModelMessage("done") if ending == "text" else None,
                tool_calls=ToolCallsEvent([_call("search")]) if ending == "tools" else ToolCallsEvent(),
                usage=Usage(total_tokens=25),
                model="test-model",
                provider="test-provider",
            )
        )
        reply, _ = await _run(CallLimiter(max_llm_calls=0), first, second)

        usage = await reply.usage()
        assert usage.total.total_tokens == (15 if ending == "error" else 40)
        history = await reply.context.stream.history.get_events()
        measured = [e for e in history if isinstance(e, UsageEvent)]
        assert [e.usage.total_tokens for e in measured] == ([15] if ending == "error" else [15, 25])
        assert all((e.model, e.provider) == ("test-model", "test-provider") for e in measured)

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


@dataclass
class Clock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now


class _Ticking(BaseMiddleware):
    def __init__(self, event: BaseEvent, context: Context, clock: Clock) -> None:
        super().__init__(event, context)
        self._clock = clock

    async def on_tool_execution(self, call_next: Any, event: ToolCallEvent, context: Context) -> Any:
        result = await call_next(event, context)
        self._clock.now += 60
        return result


@pytest.mark.asyncio()
class TestClock:
    async def test_wall_clock_expiry_allows_exactly_one_synthesis_call(self) -> None:
        clock = Clock()
        counter = CallCounter(clock=clock)
        reply, run = await _run(
            CallLimiter(max_seconds=30, counter=counter),
            _call("search"),
            "the answer",
            middleware=[Middleware(_Ticking, clock=clock)],
        )

        assert run.ran == {"search": 1}
        assert SYNTHESIS in _text(run.last_messages()[-1])
        assert reply.body == "the answer"
        assert [e.reason for e in run.limits] == ["max_seconds"]
        assert counter.llm_calls == 2

    async def test_wall_clock_expiry_under_stop_calls_the_model_no_further(self) -> None:
        clock = Clock()
        counter = CallCounter(clock=clock)
        reply, run = await _run(
            CallLimiter(max_seconds=30, on_exhausted="stop", counter=counter),
            _call("search"),
            middleware=[Middleware(_Ticking, clock=clock)],
        )

        assert len(run.last_messages()) == 1
        assert reply.response.finish_reason == "call_limit"
        assert counter.llm_denied == 1

    async def test_a_shared_counter_clock_runs_from_its_start(self) -> None:
        counter = CallCounter(clock=Clock(100.0), started=40.0)
        reply, run = await _run(CallLimiter(max_seconds=30, on_exhausted="stop", counter=counter))

        assert run.last_messages() == []
        assert [e.reason for e in run.limits] == ["max_seconds"]
        assert reply.response.finish_reason == "call_limit"


@pytest.mark.asyncio()
class TestRemainingNote:
    @pytest.mark.parametrize("failures", [1, 2])
    async def test_retry_notes_preserve_the_cached_prefix_with_snapshot_storage(self, failures: int) -> None:
        prompts: list[tuple[BaseEvent, ...]] = []
        run = Run()
        config = TrackingConfig(TestConfig(*[TimeoutError("retry") for _ in range(failures)], _call("search"), "done"))
        agent = Agent(
            "limited",
            config=config,
            tools=[run.search],
            middleware=[
                RetryMiddleware(),
                CallLimiter(max_llm_calls=5, show_remaining=True),
                Middleware(PromptRecorder, prompts=prompts),
            ],
        )

        reply = await agent.ask("Investigate.", stream=MemoryStream(storage=SnapshotStorage()))

        assert reply.body == "done"
        assert len(prompts) == config.mock.call_count == failures + 2
        assert all(later[: len(earlier)] == earlier for earlier, later in zip(prompts, prompts[1:]))

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
    async def test_conditional_limiter_is_rejected_in_either_position(self) -> None:
        conditional = ConditionalMiddleware(CallLimiter(max_tool_calls=0), ModelRequest)
        for middleware in ([conditional, CallLimiter()], [CallLimiter(), conditional]):
            agent = Agent("limited", config=TestConfig("done"), middleware=middleware)
            with pytest.raises(ValueError, match="requires the full turn"):
                await agent.ask("Investigate.")

    async def test_retries_are_charged_and_budget_notes_do_not_prevent_retry(self) -> None:
        counter = CallCounter()
        reply, run = await _run(
            CallLimiter(max_llm_calls=2, show_remaining=True, counter=counter),
            TimeoutError("first"),
            TimeoutError("second"),
            "final answer",
            middleware=[RetryMiddleware()],
        )

        assert counter.llm_calls == 3
        assert len(run.last_messages()) == 3
        assert reply.body == "final answer"
        assert [e.reason for e in run.limits] == ["max_llm_calls"]

    async def test_retries_respect_stop_without_an_extra_provider_call(self) -> None:
        counter = CallCounter()
        reply, run = await _run(
            CallLimiter(max_llm_calls=1, show_remaining=True, on_exhausted="stop", counter=counter),
            TimeoutError("first"),
            "never reached",
            middleware=[RetryMiddleware()],
        )

        assert counter.llm_calls == len(run.last_messages()) == 1
        assert reply.response.finish_reason == "call_limit"

    async def test_retry_inside_limiter_is_rejected(self) -> None:
        agent = Agent(
            "limited",
            config=TestConfig("done"),
            middleware=[CallLimiter(), RetryMiddleware()],
        )
        with pytest.raises(ValueError, match="RetryMiddleware must wrap"):
            await agent.ask("Investigate.")

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
        limiter = CallLimiter(max_tool_calls=1, on_exhausted="stop", counter=_counter_from_context)

        # Each ask replays the script from the start.
        config = TestConfig(_call("search"), "done")
        run = Run()
        agent = Agent("limited", config=config, tools=[run.search], middleware=[limiter])
        await agent.ask("one", dependencies={"calls": counter})
        reply = await agent.ask("two", dependencies={"calls": counter})

        assert run.ran == {"search": 1}
        assert counter.tool_calls == 1
        assert counter.denied == 0
        assert reply.response.finish_reason == "call_limit"

    async def test_a_default_counter_resets_every_turn(self) -> None:
        config = TestConfig(_call("search"), "done")
        run = Run()
        agent = Agent("limited", config=config, tools=[run.search], middleware=[CallLimiter(max_tool_calls=1)])
        await agent.ask("one")
        await agent.ask("two")

        assert run.ran == {"search": 2}

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

        # The outer budget is spent after search. Its two final calls bypass
        # the inner policy, which counts only the initial model call.
        assert run.ran == {"search": 1}
        assert reply.body == "the answer"
        assert SYNTHESIS in _text(run.last_messages()[1])
        assert (outer.llm_calls, inner.llm_calls, inner.llm_denied) == (3, 1, 0)
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
                exhausted_result=_stop_result,
            ),
            _call("search"),
            [_call("shell"), _call("shell")],
            "done",
        )

        results = [r.result.parts[0] for r in run.results]
        assert isinstance(results[0], TextInput) and results[0].content == "no more searching"
        assert any(isinstance(result, DataInput) and result.data == {"stop": "shell"} for result in results[1:])
