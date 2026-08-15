# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

import pytest

from ag2 import Agent, Context, tool
from ag2.events import (
    ModelMessage,
    ModelResponse,
    ObserverAlert,
    ObserverCompleted,
    ObserverStarted,
    Severity,
    ToolCallEvent,
    Usage,
    UsageEvent,
)
from ag2.observers import BaseObserver, LoopDetector, TokenMonitor
from ag2.stream import MemoryStream
from ag2.testing import TestConfig
from ag2.tools.subagents import persistent_stream
from ag2.tools.subagents.run_task import run_task
from ag2.watch import EventWatch

# Every real provider populates ``total_tokens``; fixtures that omit it would
# pass against a guard counting nothing.
_BILLED = Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110)

# High enough that no alert fires, so a case can assert on the running total alone.
_NO_THRESHOLD = 10**9


@tool
def _flaky() -> str:
    """A downstream API that is down.

    Failure is triggered by a tool that raises — the real-world flaky-dependency
    case — rather than by exhausting the test double's scripted responses, which
    would surface as a ``StopIteration`` artefact.
    """
    raise RuntimeError("downstream API is down")


def _armed_guard() -> tuple[MemoryStream, Context, TokenMonitor]:
    """A registered guard on a fresh stream, with thresholds that never fire.

    Lets a case assert on the running total alone.
    """
    stream = MemoryStream()
    ctx = Context(stream=stream)
    monitor = TokenMonitor(warn_threshold=_NO_THRESHOLD, alert_threshold=_NO_THRESHOLD)
    monitor.register(ExitStack(), ctx)
    return stream, ctx, monitor


@pytest.mark.asyncio
class TestTokenMonitor:
    """The guard reads ``UsageEvent`` — the framework's accounting record.

    The cases below that cover delegated and maintenance spend drive the
    framework rather than hand-sending events, because each defect they cover is
    about *which* event the framework actually produces. A hand-sent event would
    let them pass against a guard still reading a derived one.
    """

    async def test_no_signal_below_threshold(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        monitor.register(ExitStack(), ctx)

        # Spend 50 tokens — below threshold
        await ctx.send(UsageEvent(Usage(total_tokens=50)))

        assert len(signals) == 0
        assert monitor.total_tokens == 50

    async def test_warning_at_threshold(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        monitor.register(ExitStack(), ctx)

        await ctx.send(UsageEvent(Usage(total_tokens=110)))

        assert len(signals) == 1
        assert signals[0].severity is Severity.WARNING
        assert "token-monitor" in signals[0].source

    async def test_critical_at_threshold(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        monitor.register(ExitStack(), ctx)

        # Jump straight past both thresholds
        await ctx.send(UsageEvent(Usage(total_tokens=250)))

        # Should emit CRITICAL (not WARNING since critical is checked first)
        assert len(signals) == 1
        assert signals[0].severity is Severity.CRITICAL

    async def test_reset_clears_counter_and_allows_rewarning(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        monitor.register(ExitStack(), ctx)

        await ctx.send(UsageEvent(Usage(total_tokens=110)))
        assert monitor.total_tokens == 110
        assert len(signals) == 1

        monitor.reset()
        assert monitor.total_tokens == 0

        # Warning must fire again after reset
        await ctx.send(UsageEvent(Usage(total_tokens=110)))
        assert len(signals) == 2

    async def test_subtask_rollup_usage(self) -> None:
        """A sub-agent's spend reaches the parent as a ``"subtask"`` rollup."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        monitor.register(ExitStack(), ctx)

        await ctx.send(UsageEvent(Usage(total_tokens=60), kind="subtask", label="task-1"))

        assert monitor.total_tokens == 60
        assert len(signals) == 0

    async def test_subtask_rollup_triggers_warning(self) -> None:
        """Sub-agent tokens count toward thresholds."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        monitor.register(ExitStack(), ctx)

        await ctx.send(UsageEvent(Usage(total_tokens=120), kind="subtask", label="task-1"))

        assert len(signals) == 1
        assert signals[0].severity is Severity.WARNING

    async def test_cumulative_across_model_call_and_subtask(self) -> None:
        """The agent's own calls and its sub-agents' rollups accumulate together."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        monitor.register(ExitStack(), ctx)

        await ctx.send(UsageEvent(Usage(total_tokens=60)))
        await ctx.send(UsageEvent(Usage(total_tokens=50), kind="subtask", label="task-1"))

        assert monitor.total_tokens == 110
        assert len(signals) == 1
        assert signals[0].severity is Severity.WARNING

    async def test_empty_usage_ignored(self) -> None:
        """Events with no usage data should not affect counters."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        monitor.register(ExitStack(), ctx)

        await ctx.send(UsageEvent())
        await ctx.send(UsageEvent(Usage(), kind="subtask", label="task-1"))

        assert monitor.total_tokens == 0

    async def test_warning_only_emitted_once(self) -> None:
        """Warning alert should fire only once, not on every subsequent event."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=500)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        monitor.register(ExitStack(), ctx)

        await ctx.send(UsageEvent(Usage(total_tokens=110)))
        await ctx.send(UsageEvent(Usage(total_tokens=50)))

        assert len(signals) == 1

    async def test_failed_subtask_spend_reaches_the_guard(self) -> None:
        """A sub-task that bills and then dies is the scenario cost control exists for.

        ``TaskFailed`` carries no usage and never will; the spend reaches the
        parent as the ``"subtask"`` rollup the sub-task runner already emits
        before the terminal event.
        """
        stream, ctx, monitor = _armed_guard()

        worker = Agent(
            "worker",
            config=TestConfig(
                ModelResponse(usage=_BILLED, tool_calls=[{"id": "1", "name": "flaky", "arguments": "{}"}]),
            ),
            tools=[_flaky],
        )

        result = await run_task(worker, "go", parent_context=ctx)

        assert result.completed is False
        assert monitor.total_tokens == 110

    async def test_failed_subtask_alone_trips_the_warning(self) -> None:
        """The alert must fire on spend that failed, not only on spend that succeeded."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=500)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))
        monitor.register(ExitStack(), ctx)

        worker = Agent(
            "worker",
            config=TestConfig(
                ModelResponse(usage=_BILLED, tool_calls=[{"id": "1", "name": "flaky", "arguments": "{}"}]),
            ),
            tools=[_flaky],
        )

        await run_task(worker, "go", parent_context=ctx)

        assert [s.severity for s in signals] == [Severity.WARNING]

    async def test_repeated_delegations_do_not_inflate_the_total(self) -> None:
        """``TaskCompleted.usage`` is a cumulative snapshot; the guard accumulates with ``+=``.

        Reading the snapshot made three delegations of 110 report 660. The
        rollup carries this invocation's spend, so the total is linear.
        """
        stream, ctx, monitor = _armed_guard()

        worker = Agent(
            "worker",
            config=TestConfig(*(ModelResponse(ModelMessage("done"), usage=_BILLED) for _ in range(3))),
        )
        factory = persistent_stream()

        for _ in range(3):
            await run_task(worker, "go", parent_context=ctx, stream=factory(worker, ctx))

        assert monitor.total_tokens == 330

    async def test_agents_own_model_call_still_counts_once(self) -> None:
        """The main loop emits both the accounting event and the response — count one.

        This is the double-count guard: reading both sources would report 220.
        """
        stream, ctx, monitor = _armed_guard()

        agent = Agent("solo", config=TestConfig(ModelResponse(ModelMessage("hi"), usage=_BILLED)))
        await agent.ask("hello", stream=stream)

        assert monitor.total_tokens == 110

    @pytest.mark.parametrize("kind", ["compaction", "aggregation", "model_call"])
    async def test_maintenance_and_live_session_spend_counts(self, kind: str) -> None:
        """History compaction, memory aggregation and the live clients report usage only.

        Compaction calls the model on a throwaway stream and surfaces just the
        accounting event onto the agent's; live clients map realtime usage
        straight to it. None produce a response event, so all were invisible.
        """
        stream, ctx, monitor = _armed_guard()

        await ctx.send(UsageEvent(_BILLED, kind=kind))

        assert monitor.total_tokens == 110

    async def test_subtask_that_spent_nothing_moves_nothing(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))
        monitor.register(ExitStack(), ctx)

        worker = Agent("worker", config=TestConfig(ModelResponse(ModelMessage("done"))))

        await run_task(worker, "go", parent_context=ctx)

        assert monitor.total_tokens == 0
        assert signals == []


@pytest.mark.asyncio
class TestTokenMonitorTotalFallback:
    """A call with counts is never counted as free.

    Every shipped provider mapper populates ``total_tokens``, but the shared
    ``Usage`` type does not synthesize it — so usage assembled by hand, as a
    custom client or a fixture would, used to register as zero tokens and left
    the guard silently inert.
    """

    async def test_counts_prompt_plus_completion_when_no_total_reported(self) -> None:
        stream, ctx, monitor = _armed_guard()

        await ctx.send(UsageEvent(Usage(prompt_tokens=100, completion_tokens=10)))

        assert monitor.total_tokens == 110

    async def test_reported_total_is_used_rather_than_recomputed(self) -> None:
        """Providers whose total is not simply prompt plus completion must be believed."""
        stream, ctx, monitor = _armed_guard()

        await ctx.send(UsageEvent(Usage(prompt_tokens=100, completion_tokens=10, total_tokens=999)))

        assert monitor.total_tokens == 999

    async def test_usage_with_neither_moves_nothing(self) -> None:
        stream, ctx, monitor = _armed_guard()

        await ctx.send(UsageEvent(Usage(cache_read_input_tokens=50)))

        assert monitor.total_tokens == 0

    async def test_fallback_can_trip_a_threshold(self) -> None:
        """The fallback is reachable behaviour, not a number nobody reads."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=500)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))
        monitor.register(ExitStack(), ctx)

        await ctx.send(UsageEvent(Usage(prompt_tokens=100, completion_tokens=10)))

        assert [s.severity for s in signals] == [Severity.WARNING]

    async def test_usage_type_does_not_synthesize_a_total(self) -> None:
        """The fallback lives in the guard; the shared value type is untouched."""
        assert Usage(prompt_tokens=100, completion_tokens=10).total_tokens is None

    async def test_a_partial_total_does_not_understate(self) -> None:
        """Summed usage adds totals field-wise, so one omission drags the sum low.

        A rollup covering two calls where only one provider reported a total
        carries ``total_tokens`` for that call alone — below the prompt and
        completion counts it is supposed to cover.
        """
        stream, ctx, monitor = _armed_guard()

        reported = Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110)
        omitted = Usage(prompt_tokens=50, completion_tokens=5)
        rollup = reported + omitted
        assert rollup.total_tokens == 110  # the sum under-reports its own parts

        await ctx.send(UsageEvent(rollup, kind="subtask", label="worker"))

        assert monitor.total_tokens == 165

    async def test_a_total_above_the_two_counts_is_believed(self) -> None:
        """Reasoning tokens can legitimately put the total above prompt plus completion."""
        stream, ctx, monitor = _armed_guard()

        await ctx.send(UsageEvent(Usage(prompt_tokens=100, completion_tokens=10, total_tokens=400)))

        assert monitor.total_tokens == 400


@pytest.mark.asyncio
class TestLoopDetector:
    async def test_no_signal_below_threshold(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        detector = LoopDetector(repeat_threshold=3)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        detector.register(ExitStack(), ctx)

        # Only 2 identical calls — below threshold of 3
        await ctx.send(ToolCallEvent(name="search", arguments="q"))
        await ctx.send(ToolCallEvent(name="search", arguments="q"))

        assert len(signals) == 0

    async def test_signals_on_loop(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        detector = LoopDetector(repeat_threshold=3)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        detector.register(ExitStack(), ctx)

        # 3 identical calls — should trigger
        for _ in range(3):
            await ctx.send(ToolCallEvent(name="search", arguments="q"))

        assert len(signals) == 1
        assert signals[0].severity is Severity.WARNING
        assert "loop" in signals[0].message.lower()

    async def test_different_calls_no_signal(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        detector = LoopDetector(repeat_threshold=3)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        detector.register(ExitStack(), ctx)

        # Different calls — no loop
        await ctx.send(ToolCallEvent(name="search", arguments="q1"))
        await ctx.send(ToolCallEvent(name="search", arguments="q2"))
        await ctx.send(ToolCallEvent(name="search", arguments="q3"))

        assert len(signals) == 0

    async def test_reset_clears_history_and_allows_redetection(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        detector = LoopDetector(repeat_threshold=3)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        detector.register(ExitStack(), ctx)

        for _ in range(3):
            await ctx.send(ToolCallEvent(name="search", arguments="q"))
        assert len(signals) == 1

        detector.reset()

        # Same sequence must trigger again after reset
        for _ in range(3):
            await ctx.send(ToolCallEvent(name="search", arguments="q"))
        assert len(signals) == 2


class _SelfAwareObserver(BaseObserver):
    """Observer that watches ``ObserverStarted``/``ObserverCompleted`` on itself."""

    def __init__(self, name: str = "self-aware") -> None:
        super().__init__(name, watch=EventWatch(ObserverStarted | ObserverCompleted))
        self.started_seen: list[str] = []
        self.completed_seen: list[str] = []

    async def process(self, events, ctx) -> None:
        for event in events:
            if isinstance(event, ObserverStarted):
                self.started_seen.append(event.name)
            elif isinstance(event, ObserverCompleted):
                self.completed_seen.append(event.name)
        return None


@pytest.mark.asyncio
class TestObserverLifecycleSelfVisibility:
    """An observer subscribed to its own lifecycle events must receive them.

    ``ObserverStarted`` is emitted *after* the observer registers on the
    stream so the observer itself can react to its own start; the same
    contract applies to ``ObserverCompleted`` (emitted *before* unregister).
    """

    async def test_observer_sees_own_started_and_completed(self) -> None:
        obs = _SelfAwareObserver()
        agent = Agent(
            "with-obs",
            config=TestConfig(ModelResponse(ModelMessage("hello"))),
            observers=[obs],
        )
        await agent.ask("hi")

        assert obs.started_seen == ["self-aware"]
        assert obs.completed_seen == ["self-aware"]

    async def test_external_subscriber_also_sees_started(self) -> None:
        stream = MemoryStream()
        started: list[ObserverStarted] = []
        stream.where(ObserverStarted).subscribe(lambda e: started.append(e))

        agent = Agent(
            "lifecycle",
            config=TestConfig(ModelResponse(ModelMessage("hi"))),
            observers=[_SelfAwareObserver(name="alpha")],
        )
        await agent.ask("go", stream=stream)

        assert len(started) == 1
        assert started[0].name == "alpha"
