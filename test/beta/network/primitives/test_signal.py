# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.context import Context
from autogen.beta.events.conditions import TypeCondition
from autogen.beta.network.primitives.signal import (
    CallHandler,
    EmitToStream,
    HaltEvent,
    HaltOnFatal,
    InjectToPrompt,
    Severity,
    Signal,
)
from autogen.beta.stream import MemoryStream


class TestSignalCreation:
    def test_create_signal(self) -> None:
        s = Signal(source="monitor", severity=Severity.WARNING, message="High load")
        assert s.source == "monitor"
        assert s.severity == Severity.WARNING
        assert s.message == "High load"
        assert s.data == {}

    def test_create_signal_with_data(self) -> None:
        s = Signal(source="mon", severity="custom", message="msg", data={"key": "val"})
        assert s.data == {"key": "val"}
        assert s.severity == "custom"  # Accepts any string

    def test_severity_values(self) -> None:
        assert Severity.INFO == "info"
        assert Severity.WARNING == "warning"
        assert Severity.CRITICAL == "critical"
        assert Severity.FATAL == "fatal"


class TestInjectToPrompt:
    @pytest.mark.asyncio
    async def test_non_fatal_appends_to_prompt(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        policy = InjectToPrompt()

        signals = [
            Signal(source="token-mon", severity=Severity.WARNING, message="High token usage"),
            Signal(source="loop-det", severity=Severity.CRITICAL, message="Loop detected"),
        ]
        await policy.deliver(signals, ctx)

        assert len(ctx.prompt) == 1
        assert "[OBSERVER MONITORING ALERTS]" in ctx.prompt[0]
        assert "token-mon" in ctx.prompt[0]
        assert "loop-det" in ctx.prompt[0]

    @pytest.mark.asyncio
    async def test_fatal_emits_halt_event(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        policy = InjectToPrompt()
        halt_events: list = []

        stream.subscribe(
            lambda e: halt_events.append(e),
            condition=TypeCondition(HaltEvent),
        )

        signals = [
            Signal(source="safety", severity=Severity.FATAL, message="Critical safety violation"),
        ]
        await policy.deliver(signals, ctx)

        assert len(halt_events) == 1
        assert halt_events[0].source == "safety"
        assert "FATAL" in halt_events[0].reason

    @pytest.mark.asyncio
    async def test_empty_signals_no_change(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        policy = InjectToPrompt()

        await policy.deliver([], ctx)
        assert len(ctx.prompt) == 0


class TestEmitToStream:
    @pytest.mark.asyncio
    async def test_emits_each_signal(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        policy = EmitToStream()
        received: list = []

        stream.subscribe(
            lambda e: received.append(e),
            condition=TypeCondition(Signal),
        )

        signals = [
            Signal(source="a", severity=Severity.INFO, message="msg1"),
            Signal(source="b", severity=Severity.WARNING, message="msg2"),
        ]
        await policy.deliver(signals, ctx)
        assert len(received) == 2


class TestCallHandler:
    @pytest.mark.asyncio
    async def test_calls_handler(self) -> None:
        received: list = []

        async def handler(signals):
            received.extend(signals)

        policy = CallHandler(handler)
        stream = MemoryStream()
        ctx = Context(stream=stream)

        signals = [Signal(source="a", severity=Severity.INFO, message="test")]
        await policy.deliver(signals, ctx)
        assert len(received) == 1


class TestHaltOnFatal:
    @pytest.mark.asyncio
    async def test_halts_on_fatal_delegates_rest(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        halt_events: list = []

        stream.subscribe(
            lambda e: halt_events.append(e),
            condition=TypeCondition(HaltEvent),
        )

        inner_received: list = []

        async def inner_handler(signals):
            inner_received.extend(signals)

        policy = HaltOnFatal(inner=CallHandler(inner_handler))

        signals = [
            Signal(source="a", severity=Severity.WARNING, message="warn"),
            Signal(source="b", severity=Severity.FATAL, message="die"),
        ]
        await policy.deliver(signals, ctx)

        # Non-fatal delivered first, then FATAL halts
        assert len(halt_events) == 1
        assert len(inner_received) == 1
        assert inner_received[0].severity == Severity.WARNING

    @pytest.mark.asyncio
    async def test_no_fatal_delegates_to_inner(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)

        inner_received: list = []

        async def inner_handler(signals):
            inner_received.extend(signals)

        policy = HaltOnFatal(inner=CallHandler(inner_handler))

        signals = [
            Signal(source="a", severity=Severity.WARNING, message="warn"),
        ]
        await policy.deliver(signals, ctx)
        assert len(inner_received) == 1
