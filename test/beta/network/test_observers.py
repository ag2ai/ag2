# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.context import Context
from autogen.beta.events import ModelResponse, ToolCallEvent
from autogen.beta.events.conditions import TypeCondition
from autogen.beta.network.observers import LoopDetector, TokenMonitor
from autogen.beta.network.primitives.signal import Severity, Signal
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
            condition=TypeCondition(Signal),
        )

        monitor.attach(stream, ctx)

        # Send a response with 50 tokens — below threshold
        event = ModelResponse(usage={"total_tokens": 50})
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
            condition=TypeCondition(Signal),
        )

        monitor.attach(stream, ctx)

        await stream.send(ModelResponse(usage={"total_tokens": 110}), ctx)

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
            condition=TypeCondition(Signal),
        )

        monitor.attach(stream, ctx)

        # Jump straight past both thresholds
        await stream.send(ModelResponse(usage={"total_tokens": 250}), ctx)

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


class TestLoopDetector:
    @pytest.mark.asyncio
    async def test_no_signal_below_threshold(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        detector = LoopDetector(repeat_threshold=3)

        signals: list = []
        stream.subscribe(
            lambda e: signals.append(e),
            condition=TypeCondition(Signal),
        )

        detector.attach(stream, ctx)

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
            condition=TypeCondition(Signal),
        )

        detector.attach(stream, ctx)

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
            condition=TypeCondition(Signal),
        )

        detector.attach(stream, ctx)

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
