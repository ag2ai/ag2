# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from contextlib import ExitStack

import pytest

from ag2 import Agent, Context, MemoryStream, Plugin, observer
from ag2.events import BaseEvent, ModelMessage, ObserverAlert, Severity
from ag2.observers import BaseObserver, LoopDetector, TokenMonitor
from ag2.testing import TestConfig
from ag2.watch import AnyOf, CadenceWatch, DelayWatch, EventWatch, IntervalWatch, Watch


class WaitingCallback:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.settled = asyncio.Event()
        self.release = asyncio.Event()

    async def handle(self, event: BaseEvent) -> None:
        self.started.set()
        try:
            await self.release.wait()
        finally:
            self.settled.set()


class RecoveringObserver(BaseObserver):
    def __init__(self, watch: Watch, timeout: float | None = 0.02) -> None:
        super().__init__("recovering", watch, timeout=timeout)
        self.calls = 0
        self.started = asyncio.Event()
        self.settled = asyncio.Event()
        self.recovered = asyncio.Event()

    async def process(self, events: list[BaseEvent], ctx: Context) -> ObserverAlert | None:
        self.calls += 1
        self.started.set()
        if self.calls == 1:
            try:
                await asyncio.Event().wait()
            finally:
                self.settled.set()
        self.recovered.set()
        return ObserverAlert(source=self.name, severity=Severity.INFO, message="recovered")


class SignallingDelayWatch(DelayWatch):
    def __init__(self) -> None:
        super().__init__(0.005)
        self.disarmed = asyncio.Event()

    def disarm(self) -> None:
        super().disarm()
        self.disarmed.set()


class AlertObserver(BaseObserver):
    async def process(self, events: list[BaseEvent], ctx: Context) -> ObserverAlert | None:
        return ObserverAlert(source=self.name, severity=Severity.INFO, message="ready")


@pytest.mark.asyncio
@pytest.mark.parametrize("registration", ["factory", "decorator", "agent", "plugin"])
@pytest.mark.parametrize("filtered", [False, True])
async def test_callback_timeout_through_observer_helpers(registration: str, filtered: bool) -> None:
    callback = WaitingCallback()
    condition = ModelMessage if filtered else None
    agent = Agent("test", config=TestConfig("done"))
    if registration == "factory":
        agent.add_observer(observer(condition, callback.handle, timeout=0.02))
    elif registration == "decorator":
        agent.add_observer(observer(condition, timeout=0.02)(callback.handle))
    elif registration == "agent":
        agent.observer(condition, timeout=0.02)(callback.handle)
    else:
        plugin = Plugin()
        plugin.observer(condition, callback.handle, timeout=0.02)
        agent = Agent("test", config=TestConfig("done"), plugins=[plugin])

    with pytest.raises(asyncio.TimeoutError):
        await agent.ask("trigger")

    assert callback.started.is_set()
    assert callback.settled.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["event", "cadence_count", "cadence_timer", "interval", "delay", "composite"])
async def test_base_observer_timeout_is_logged_for_every_trigger(kind: str, caplog: pytest.LogCaptureFixture) -> None:
    if kind == "event":
        watch = EventWatch(ModelMessage)
    elif kind == "cadence_count":
        watch = CadenceWatch(n=1, condition=ModelMessage)
    elif kind == "cadence_timer":
        watch = CadenceWatch(max_wait=0.005, condition=ModelMessage)
    elif kind == "interval":
        watch = IntervalWatch(0.005)
    elif kind == "delay":
        watch = SignallingDelayWatch()
    else:
        watch = AnyOf(IntervalWatch(0.005))
    obs = RecoveringObserver(watch)
    stream = MemoryStream()
    ctx = Context(stream)
    alerts: asyncio.Queue[ObserverAlert] = asyncio.Queue()
    stream.where(ObserverAlert).subscribe(alerts.put_nowait, sync_to_thread=False)

    with caplog.at_level(logging.ERROR), ExitStack() as stack:
        obs.register(stack, ctx)
        await ctx.send(ModelMessage("first"))
        await asyncio.wait_for(obs.settled.wait(), timeout=5)

        if kind in ("event", "cadence_count", "cadence_timer"):
            await ctx.send(ModelMessage("second"))
        if kind == "delay":
            assert isinstance(watch, SignallingDelayWatch)
            await asyncio.wait_for(watch.disarmed.wait(), timeout=5)
            assert alerts.empty()
        else:
            alert = await asyncio.wait_for(alerts.get(), timeout=5)
            assert alert == ObserverAlert(source="recovering", severity=Severity.INFO, message="recovered")
            assert obs.calls >= 2

        assert any(
            "Observer 'recovering' process() failed" in record.message
            and record.exc_info is not None
            and isinstance(record.exc_info[1], asyncio.TimeoutError)
            for record in caplog.records
        )

    assert not watch.is_armed


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 60])
async def test_base_observer_caller_cancellation_is_not_logged(
    caplog: pytest.LogCaptureFixture, timeout: float | None
) -> None:
    obs = RecoveringObserver(EventWatch(ModelMessage), timeout=timeout)
    stream = MemoryStream()
    with ExitStack() as stack:
        obs.register(stack, Context(stream))
        task = asyncio.create_task(Context(stream).send(ModelMessage("blocked")))
        try:
            await obs.started.wait()
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    assert obs.settled.is_set()
    assert not caplog.records


@pytest.mark.asyncio
async def test_base_observer_timeout_excludes_returned_alert_delivery(caplog: pytest.LogCaptureFixture) -> None:
    stream = MemoryStream()
    ctx = Context(stream)
    callback = WaitingCallback()
    stream.where(ObserverAlert).subscribe(callback.handle)
    obs = AlertObserver("alert", EventWatch(ModelMessage), timeout=0.02)

    with ExitStack() as stack:
        obs.register(stack, ctx)
        task = asyncio.create_task(ctx.send(ModelMessage("trigger")))
        try:
            await callback.started.wait()
            await asyncio.sleep(0.04)
            assert not task.done()
        finally:
            callback.release.set()
            await task

    assert callback.settled.is_set()
    assert not caplog.records


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("kind", ["callback", "filtered_callback", "base", "loop_detector", "token_monitor"])
def test_invalid_observer_timeout_is_rejected(timeout: float, kind: str) -> None:
    with pytest.raises(ValueError, match="finite number > 0 seconds or None"):
        if kind == "callback":
            observer(None, WaitingCallback().handle, timeout=timeout)
        elif kind == "filtered_callback":
            observer(ModelMessage, WaitingCallback().handle, timeout=timeout)
        elif kind == "base":
            RecoveringObserver(EventWatch(ModelMessage), timeout=timeout)
        elif kind == "loop_detector":
            LoopDetector(timeout=timeout)
        else:
            TokenMonitor(timeout=timeout)
