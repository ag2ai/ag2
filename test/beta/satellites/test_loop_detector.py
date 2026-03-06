# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.context import Context
from autogen.beta.events import ToolCall
from autogen.beta.satellites.builtins.loop_detector import LoopDetector
from autogen.beta.satellites.events import SatelliteFlag, Severity
from autogen.beta.stream import MemoryStream


@pytest.mark.asyncio
async def test_loop_detector_flags_repeated_calls():
    stream = MemoryStream()
    ctx = Context(stream)
    detector = LoopDetector(repeat_threshold=3)
    detector.attach(stream, ctx)

    flags: list[SatelliteFlag] = []
    original_send = ctx.stream.send

    async def capture_send(event, ctx):
        if isinstance(event, SatelliteFlag):
            flags.append(event)
        await original_send(event, ctx)

    ctx.stream.send = capture_send

    # Three identical tool calls
    for _ in range(3):
        await ctx.send(ToolCall(name="search", arguments='{"q": "test"}'))

    assert len(flags) == 1
    assert flags[0].severity == Severity.WARNING
    assert "search" in flags[0].message

    detector.detach()


@pytest.mark.asyncio
async def test_loop_detector_no_flag_for_varied_calls():
    stream = MemoryStream()
    ctx = Context(stream)
    detector = LoopDetector(repeat_threshold=3)
    detector.attach(stream, ctx)

    flags: list[SatelliteFlag] = []
    original_send = ctx.stream.send

    async def capture_send(event, ctx):
        if isinstance(event, SatelliteFlag):
            flags.append(event)
        await original_send(event, ctx)

    ctx.stream.send = capture_send

    # Different tool calls
    await ctx.send(ToolCall(name="search", arguments='{"q": "a"}'))
    await ctx.send(ToolCall(name="search", arguments='{"q": "b"}'))
    await ctx.send(ToolCall(name="search", arguments='{"q": "c"}'))

    assert len(flags) == 0
    detector.detach()


@pytest.mark.asyncio
async def test_loop_detector_different_tools_no_flag():
    stream = MemoryStream()
    ctx = Context(stream)
    detector = LoopDetector(repeat_threshold=3)
    detector.attach(stream, ctx)

    flags: list[SatelliteFlag] = []
    original_send = ctx.stream.send

    async def capture_send(event, ctx):
        if isinstance(event, SatelliteFlag):
            flags.append(event)
        await original_send(event, ctx)

    ctx.stream.send = capture_send

    await ctx.send(ToolCall(name="search", arguments='{"q": "test"}'))
    await ctx.send(ToolCall(name="calculate", arguments='{"x": 1}'))
    await ctx.send(ToolCall(name="search", arguments='{"q": "test"}'))

    assert len(flags) == 0
    detector.detach()


@pytest.mark.asyncio
async def test_loop_detector_flags_only_once_per_pattern():
    stream = MemoryStream()
    ctx = Context(stream)
    detector = LoopDetector(repeat_threshold=2)
    detector.attach(stream, ctx)

    flags: list[SatelliteFlag] = []
    original_send = ctx.stream.send

    async def capture_send(event, ctx):
        if isinstance(event, SatelliteFlag):
            flags.append(event)
        await original_send(event, ctx)

    ctx.stream.send = capture_send

    # Same pattern repeated many times
    for _ in range(5):
        await ctx.send(ToolCall(name="fetch", arguments='{}'))

    # Only one flag for this pattern
    assert len(flags) == 1
    detector.detach()


@pytest.mark.asyncio
async def test_loop_detector_reset():
    stream = MemoryStream()
    ctx = Context(stream)
    detector = LoopDetector(repeat_threshold=3)
    detector.attach(stream, ctx)

    await ctx.send(ToolCall(name="search", arguments='{}'))
    await ctx.send(ToolCall(name="search", arguments='{}'))

    detector.detach()
    # State persists after detach
    assert len(detector._history) == 2

    detector.reset()
    assert len(detector._history) == 0
    assert len(detector._flagged) == 0
