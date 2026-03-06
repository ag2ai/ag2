# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.context import Context
from autogen.beta.events import ModelResponse
from autogen.beta.satellites.builtins.token_monitor import TokenMonitor
from autogen.beta.satellites.events import SatelliteFlag, Severity, TaskSatelliteResult
from autogen.beta.stream import MemoryStream


@pytest.mark.asyncio
async def test_token_monitor_warns_at_threshold():
    stream = MemoryStream()
    ctx = Context(stream)
    monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)
    monitor.attach(stream, ctx)

    # Send events below threshold
    await ctx.send(ModelResponse(usage={"total_tokens": 50}))
    assert monitor.total_tokens == 50

    # Cross warning threshold
    flags: list[SatelliteFlag] = []
    original_send = ctx.stream.send

    async def capture_send(event, ctx):
        if isinstance(event, SatelliteFlag):
            flags.append(event)
        await original_send(event, ctx)

    ctx.stream.send = capture_send
    await ctx.send(ModelResponse(usage={"total_tokens": 60}))

    assert monitor.total_tokens == 110
    assert len(flags) == 1
    assert flags[0].severity == Severity.WARNING

    monitor.detach()


@pytest.mark.asyncio
async def test_token_monitor_alerts_at_threshold():
    stream = MemoryStream()
    ctx = Context(stream)
    monitor = TokenMonitor(warn_threshold=50, alert_threshold=100)
    monitor.attach(stream, ctx)

    flags: list[SatelliteFlag] = []
    original_send = ctx.stream.send

    async def capture_send(event, ctx):
        if isinstance(event, SatelliteFlag):
            flags.append(event)
        await original_send(event, ctx)

    ctx.stream.send = capture_send

    # Jump straight past both thresholds
    await ctx.send(ModelResponse(usage={"total_tokens": 150}))
    assert monitor.total_tokens == 150

    # Should emit CRITICAL (not WARNING, since critical takes priority)
    assert len(flags) == 1
    assert flags[0].severity == Severity.CRITICAL

    monitor.detach()


@pytest.mark.asyncio
async def test_token_monitor_tracks_task_satellite_usage():
    stream = MemoryStream()
    ctx = Context(stream)
    monitor = TokenMonitor(warn_threshold=200, alert_threshold=500)
    monitor.attach(stream, ctx)

    await ctx.send(ModelResponse(usage={"total_tokens": 50}))
    await ctx.send(
        TaskSatelliteResult(
            task="test",
            satellite_name="sat-1",
            result="done",
            usage={"total_tokens": 100},
        )
    )

    assert monitor.total_tokens == 150
    monitor.detach()


@pytest.mark.asyncio
async def test_token_monitor_no_flag_below_threshold():
    stream = MemoryStream()
    ctx = Context(stream)
    monitor = TokenMonitor(warn_threshold=1000, alert_threshold=2000)
    monitor.attach(stream, ctx)

    result = await monitor.process(
        [ModelResponse(usage={"total_tokens": 10})], ctx
    )
    assert result is None
    monitor.detach()


@pytest.mark.asyncio
async def test_token_monitor_persists_after_detach():
    stream = MemoryStream()
    ctx = Context(stream)
    monitor = TokenMonitor(warn_threshold=50, alert_threshold=100)
    monitor.attach(stream, ctx)

    await ctx.send(ModelResponse(usage={"total_tokens": 75}))
    assert monitor.total_tokens == 75

    monitor.detach()
    # Counters persist after detach so callers can read them
    assert monitor.total_tokens == 75

    monitor.reset()
    assert monitor.total_tokens == 0


@pytest.mark.asyncio
async def test_token_monitor_flags_only_once():
    stream = MemoryStream()
    ctx = Context(stream)
    monitor = TokenMonitor(warn_threshold=10, alert_threshold=100)
    monitor.attach(stream, ctx)

    flags: list[SatelliteFlag] = []
    original_send = ctx.stream.send

    async def capture_send(event, ctx):
        if isinstance(event, SatelliteFlag):
            flags.append(event)
        await original_send(event, ctx)

    ctx.stream.send = capture_send

    # Multiple events crossing warning threshold
    await ctx.send(ModelResponse(usage={"total_tokens": 15}))
    await ctx.send(ModelResponse(usage={"total_tokens": 15}))
    await ctx.send(ModelResponse(usage={"total_tokens": 15}))

    # Only one WARNING flag
    warning_flags = [f for f in flags if f.severity == Severity.WARNING]
    assert len(warning_flags) == 1

    monitor.detach()
