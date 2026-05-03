# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from contextlib import ExitStack

import pytest

from autogen.beta import Context
from autogen.beta.events import ModelMessage, ObserverAlert, Severity, ToolCallEvent
from autogen.beta.observer import BaseObserver
from autogen.beta.stream import MemoryStream
from autogen.beta.watch import EventWatch


class DummyObserver(BaseObserver):
    """Test observer that signals on any matching event."""

    def __init__(self, name: str = "test-observer"):
        super().__init__(name, watch=EventWatch(ToolCallEvent))
        self.process_count = 0

    async def process(self, events, ctx) -> ObserverAlert | None:
        self.process_count += 1
        return ObserverAlert(
            source=self.name,
            severity=Severity.WARNING,
            message=f"Saw {len(events)} event(s)",
        )


class NullObserver(BaseObserver):
    """Observer that never signals."""

    def __init__(self):
        super().__init__("null-observer", watch=EventWatch(ToolCallEvent))

    async def process(self, events, ctx) -> ObserverAlert | None:
        return None


@pytest.mark.asyncio
class TestBaseObserver:
    async def test_attach_and_process(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        obs = DummyObserver()

        signals: list = []

        @stream.where(ObserverAlert).subscribe()
        def on_alert(e: ObserverAlert) -> None:
            signals.append(e)

        with ExitStack() as stack:
            obs.register(stack, ctx)
            await stream.send(ToolCallEvent(name="search", arguments="{}"), ctx)

            assert obs.process_count == 1
            assert len(signals) == 1
            assert signals[0].source == "test-observer"
            assert signals[0].severity is Severity.WARNING

    async def test_detach_stops_processing(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        obs = DummyObserver()

        stack = ExitStack()
        obs.register(stack, ctx)
        stack.close()

        await stream.send(ToolCallEvent(name="search", arguments="{}"), ctx)
        assert obs.process_count == 0

    async def test_null_signal_not_emitted(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        obs = NullObserver()

        signals: list = []

        @stream.where(ObserverAlert).subscribe()
        def on_alert(e: ObserverAlert) -> None:
            signals.append(e)

        with ExitStack() as stack:
            obs.register(stack, ctx)
            await stream.send(ToolCallEvent(name="search", arguments="{}"), ctx)

            assert len(signals) == 0

    async def test_only_matching_events(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        obs = DummyObserver()

        with ExitStack() as stack:
            obs.register(stack, ctx)

            # ModelMessage doesn't match ToolCallEvent watch
            await stream.send(ModelMessage(content="hello"), ctx)
            assert obs.process_count == 0

            await stream.send(ToolCallEvent(name="t", arguments="{}"), ctx)
            assert obs.process_count == 1


class _CrashingObserver(BaseObserver):
    """Observer whose ``process()`` always raises."""

    def __init__(self) -> None:
        super().__init__("crasher", watch=EventWatch(ModelMessage))

    async def process(self, events, ctx):
        raise RuntimeError("observer exploded")


class _NullModelMessageObserver(BaseObserver):
    def __init__(self) -> None:
        super().__init__("null", watch=EventWatch(ModelMessage))

    async def process(self, events, ctx):
        return None


@pytest.mark.asyncio
class TestObserverExceptionHandling:
    async def test_observer_process_exception_is_caught(self, caplog) -> None:
        observer = _CrashingObserver()
        stream = MemoryStream()
        ctx = Context(stream=stream)

        with ExitStack() as stack:
            observer.register(stack, ctx)

            with caplog.at_level(logging.ERROR):
                await stream.send(ModelMessage(content="trigger"), ctx)
                await asyncio.sleep(0.01)

        assert any("process() failed" in r.message for r in caplog.records)

    async def test_observer_returns_none_no_signal(self) -> None:
        observer = _NullModelMessageObserver()
        stream = MemoryStream()
        ctx = Context(stream=stream)

        signals: list[ObserverAlert] = []

        async def _capture(event: ObserverAlert, _ctx: Context) -> None:
            signals.append(event)

        stream.where(ObserverAlert).subscribe(_capture)

        with ExitStack() as stack:
            observer.register(stack, ctx)
            await stream.send(ModelMessage(content="test"), ctx)
            await asyncio.sleep(0.01)

        assert len(signals) == 0
