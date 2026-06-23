# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

import pytest

from autogen.beta import Agent, Context
from autogen.beta.events import (
    ModelMessage,
    ModelResponse,
    ObserverAlert,
    ObserverCompleted,
    ObserverStarted,
    Severity,
    TaskCompleted,
    ToolCallEvent,
    Usage,
)
from autogen.beta.observers import BaseObserver, LoopDetector, TokenMonitor
from autogen.beta.stream import MemoryStream
from autogen.beta.testing import TestConfig
from autogen.beta.watch import EventWatch


@pytest.mark.asyncio
class TestTokenMonitor:
    async def test_no_signal_below_threshold(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        monitor.register(ExitStack(), ctx)

        # Send a response with 50 tokens — below threshold
        event = ModelResponse(usage=Usage(total_tokens=50))
        await ctx.send(event)

        assert len(signals) == 0
        assert monitor.total_tokens == 50

    async def test_warning_at_threshold(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        monitor.register(ExitStack(), ctx)

        await ctx.send(ModelResponse(usage=Usage(total_tokens=110)))

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
        await ctx.send(ModelResponse(usage=Usage(total_tokens=250)))

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

        await ctx.send(ModelResponse(usage=Usage(total_tokens=110)))
        assert monitor.total_tokens == 110
        assert len(signals) == 1

        monitor.reset()
        assert monitor.total_tokens == 0

        # Warning must fire again after reset
        await ctx.send(ModelResponse(usage=Usage(total_tokens=110)))
        assert len(signals) == 2

    async def test_task_completed_usage(self) -> None:
        """TaskCompleted carries a Usage object — monitor must handle it."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        monitor.register(ExitStack(), ctx)

        await ctx.send(
            TaskCompleted(
                task_id="t1",
                agent_name="task-1",
                objective="x",
                result="done",
                task_stream=stream.id,
                usage=Usage(total_tokens=60),
            )
        )

        assert monitor.total_tokens == 60
        assert len(signals) == 0

    async def test_task_completed_triggers_warning(self) -> None:
        """TaskCompleted tokens should count toward thresholds."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        monitor.register(ExitStack(), ctx)

        await ctx.send(
            TaskCompleted(
                task_id="t1",
                agent_name="task-1",
                objective="x",
                result="done",
                task_stream=stream.id,
                usage=Usage(total_tokens=120),
            )
        )

        assert len(signals) == 1
        assert signals[0].severity is Severity.WARNING

    async def test_cumulative_across_model_and_task(self) -> None:
        """Tokens from ModelResponse and TaskCompleted accumulate together."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        monitor.register(ExitStack(), ctx)

        await ctx.send(ModelResponse(usage=Usage(total_tokens=60)))
        await ctx.send(
            TaskCompleted(
                task_id="t1",
                agent_name="task-1",
                objective="x",
                result="done",
                task_stream=stream.id,
                usage=Usage(total_tokens=50),
            )
        )

        assert monitor.total_tokens == 110
        assert len(signals) == 1
        assert signals[0].severity is Severity.WARNING

    async def test_empty_usage_ignored(self) -> None:
        """Events with no usage data should not affect counters."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        monitor.register(ExitStack(), ctx)

        # ModelResponse with default (empty) Usage
        await ctx.send(ModelResponse())
        # TaskCompleted with default empty Usage
        await ctx.send(
            TaskCompleted(
                task_id="t1",
                agent_name="task-1",
                objective="x",
                result="done",
                task_stream=stream.id,
            )
        )

        assert monitor.total_tokens == 0

    async def test_warning_only_emitted_once(self) -> None:
        """Warning alert should fire only once, not on every subsequent event."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=500)

        signals: list = []
        stream.where(ObserverAlert).subscribe(lambda e: signals.append(e))

        monitor.register(ExitStack(), ctx)

        await ctx.send(ModelResponse(usage=Usage(total_tokens=110)))
        await ctx.send(ModelResponse(usage=Usage(total_tokens=50)))

        assert len(signals) == 1


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
