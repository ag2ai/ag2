# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

import pytest

from autogen.beta import LoopDetector, TokenMonitor
from autogen.beta.context import ConversationContext as Context
from autogen.beta.events import ModelResponse, TaskCompleted, ToolCallEvent
from autogen.beta.events.alert import ObserverAlert, Severity
from autogen.beta.events.conditions import TypeCondition
from autogen.beta.events.types import Usage
from autogen.beta.stream import MemoryStream


class TestTokenMonitor:
    @pytest.mark.asyncio
    async def test_no_signal_below_threshold(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.subscribe(
            lambda e: signals.append(e),
            condition=TypeCondition(ObserverAlert),
        )

        monitor.register(ExitStack(), ctx)

        # Send a response with 50 tokens — below threshold
        event = ModelResponse(usage=Usage(total_tokens=50))
        await stream.send(event, ctx)

        assert len(signals) == 0
        assert monitor.total_tokens == 50

    @pytest.mark.asyncio
    async def test_warning_at_threshold(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.subscribe(
            lambda e: signals.append(e),
            condition=TypeCondition(ObserverAlert),
        )

        monitor.register(ExitStack(), ctx)

        await stream.send(ModelResponse(usage=Usage(total_tokens=110)), ctx)

        assert len(signals) == 1
        assert signals[0].severity == Severity.WARNING
        assert "token-monitor" in signals[0].source

    @pytest.mark.asyncio
    async def test_critical_at_threshold(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.subscribe(
            lambda e: signals.append(e),
            condition=TypeCondition(ObserverAlert),
        )

        monitor.register(ExitStack(), ctx)

        # Jump straight past both thresholds
        await stream.send(ModelResponse(usage=Usage(total_tokens=250)), ctx)

        # Should emit CRITICAL (not WARNING since critical is checked first)
        assert len(signals) == 1
        assert signals[0].severity == Severity.CRITICAL

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)
        monitor._total_tokens = 150
        monitor._warned = True
        monitor.reset()
        assert monitor.total_tokens == 0
        assert monitor._warned is False

    @pytest.mark.asyncio
    async def test_task_completed_usage_dict(self) -> None:
        """TaskCompleted carries usage as a plain dict — monitor must handle it."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.subscribe(
            lambda e: signals.append(e),
            condition=TypeCondition(ObserverAlert),
        )

        monitor.register(ExitStack(), ctx)

        await stream.send(
            TaskCompleted(
                task_id="t1",
                agent_name="task-1",
                objective="x",
                result="done",
                task_stream=stream.id,
                usage={"total_tokens": 60},
            ),
            ctx,
        )

        assert monitor.total_tokens == 60
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_task_completed_triggers_warning(self) -> None:
        """TaskCompleted tokens should count toward thresholds."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.subscribe(
            lambda e: signals.append(e),
            condition=TypeCondition(ObserverAlert),
        )

        monitor.register(ExitStack(), ctx)

        await stream.send(
            TaskCompleted(
                task_id="t1",
                agent_name="task-1",
                objective="x",
                result="done",
                task_stream=stream.id,
                usage={"total_tokens": 120},
            ),
            ctx,
        )

        assert len(signals) == 1
        assert signals[0].severity == Severity.WARNING

    @pytest.mark.asyncio
    async def test_cumulative_across_model_and_task(self) -> None:
        """Tokens from ModelResponse and TaskCompleted accumulate together."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        signals: list = []
        stream.subscribe(
            lambda e: signals.append(e),
            condition=TypeCondition(ObserverAlert),
        )

        monitor.register(ExitStack(), ctx)

        await stream.send(ModelResponse(usage=Usage(total_tokens=60)), ctx)
        await stream.send(
            TaskCompleted(
                task_id="t1",
                agent_name="task-1",
                objective="x",
                result="done",
                task_stream=stream.id,
                usage={"total_tokens": 50},
            ),
            ctx,
        )

        assert monitor.total_tokens == 110
        assert len(signals) == 1
        assert signals[0].severity == Severity.WARNING

    @pytest.mark.asyncio
    async def test_empty_usage_ignored(self) -> None:
        """Events with no usage data should not affect counters."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        monitor.register(ExitStack(), ctx)

        # ModelResponse with default (empty) Usage
        await stream.send(ModelResponse(), ctx)
        # TaskCompleted with default empty usage dict
        await stream.send(
            TaskCompleted(
                task_id="t1",
                agent_name="task-1",
                objective="x",
                result="done",
                task_stream=stream.id,
            ),
            ctx,
        )

        assert monitor.total_tokens == 0

    @pytest.mark.asyncio
    async def test_warning_only_emitted_once(self) -> None:
        """Warning alert should fire only once, not on every subsequent event."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        monitor = TokenMonitor(warn_threshold=100, alert_threshold=500)

        signals: list = []
        stream.subscribe(
            lambda e: signals.append(e),
            condition=TypeCondition(ObserverAlert),
        )

        monitor.register(ExitStack(), ctx)

        await stream.send(ModelResponse(usage=Usage(total_tokens=110)), ctx)
        await stream.send(ModelResponse(usage=Usage(total_tokens=50)), ctx)

        assert len(signals) == 1


class TestLoopDetector:
    @pytest.mark.asyncio
    async def test_no_signal_below_threshold(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        detector = LoopDetector(repeat_threshold=3)

        signals: list = []
        stream.subscribe(
            lambda e: signals.append(e),
            condition=TypeCondition(ObserverAlert),
        )

        detector.register(ExitStack(), ctx)

        # Only 2 identical calls — below threshold of 3
        await stream.send(ToolCallEvent(name="search", arguments="q"), ctx)
        await stream.send(ToolCallEvent(name="search", arguments="q"), ctx)

        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_signals_on_loop(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        detector = LoopDetector(repeat_threshold=3)

        signals: list = []
        stream.subscribe(
            lambda e: signals.append(e),
            condition=TypeCondition(ObserverAlert),
        )

        detector.register(ExitStack(), ctx)

        # 3 identical calls — should trigger
        for _ in range(3):
            await stream.send(ToolCallEvent(name="search", arguments="q"), ctx)

        assert len(signals) == 1
        assert signals[0].severity == Severity.WARNING
        assert "loop" in signals[0].message.lower()

    @pytest.mark.asyncio
    async def test_different_calls_no_signal(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        detector = LoopDetector(repeat_threshold=3)

        signals: list = []
        stream.subscribe(
            lambda e: signals.append(e),
            condition=TypeCondition(ObserverAlert),
        )

        detector.register(ExitStack(), ctx)

        # Different calls — no loop
        await stream.send(ToolCallEvent(name="search", arguments="q1"), ctx)
        await stream.send(ToolCallEvent(name="search", arguments="q2"), ctx)
        await stream.send(ToolCallEvent(name="search", arguments="q3"), ctx)

        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        detector = LoopDetector()
        detector._history.append(("a", "b"))
        detector._flagged.add(("a", "b"))
        detector.reset()
        assert len(detector._history) == 0
        assert len(detector._flagged) == 0
