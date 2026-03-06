# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ModelResponse, ToolCall
from autogen.beta.satellites.triggers import EveryNEvents, OnEvent
from autogen.beta.stream import MemoryStream


class _Marker(BaseEvent):
    value: int


@pytest.mark.asyncio
async def test_on_event_fires_immediately():
    stream = MemoryStream()
    ctx = Context(stream)
    collected: list[list[BaseEvent]] = []

    async def callback(events: list[BaseEvent], ctx: Context) -> None:
        collected.append(events)

    trigger = OnEvent(_Marker)
    sub_id = trigger.attach(stream, callback)

    await ctx.send(_Marker(value=1))
    await ctx.send(_Marker(value=2))

    assert len(collected) == 2
    assert collected[0][0].value == 1
    assert collected[1][0].value == 2

    trigger.detach(stream, sub_id)


@pytest.mark.asyncio
async def test_on_event_ignores_other_types():
    stream = MemoryStream()
    ctx = Context(stream)
    collected: list[list[BaseEvent]] = []

    async def callback(events: list[BaseEvent], ctx: Context) -> None:
        collected.append(events)

    trigger = OnEvent(_Marker)
    trigger.attach(stream, callback)

    await ctx.send(ModelResponse())
    assert len(collected) == 0

    await ctx.send(_Marker(value=42))
    assert len(collected) == 1


@pytest.mark.asyncio
async def test_every_n_events_buffers():
    stream = MemoryStream()
    ctx = Context(stream)
    batches: list[list[BaseEvent]] = []

    async def callback(events: list[BaseEvent], ctx: Context) -> None:
        batches.append(events)

    trigger = EveryNEvents(3)
    trigger.attach(stream, callback)

    await ctx.send(_Marker(value=1))
    await ctx.send(_Marker(value=2))
    assert len(batches) == 0

    await ctx.send(_Marker(value=3))
    assert len(batches) == 1
    assert len(batches[0]) == 3


@pytest.mark.asyncio
async def test_every_n_events_with_condition():
    stream = MemoryStream()
    ctx = Context(stream)
    batches: list[list[BaseEvent]] = []

    async def callback(events: list[BaseEvent], ctx: Context) -> None:
        batches.append(events)

    trigger = EveryNEvents(2, condition=_Marker)
    trigger.attach(stream, callback)

    await ctx.send(ModelResponse())
    await ctx.send(_Marker(value=1))
    assert len(batches) == 0

    await ctx.send(_Marker(value=2))
    assert len(batches) == 1
    assert len(batches[0]) == 2


@pytest.mark.asyncio
async def test_every_n_events_detach_clears_buffer():
    stream = MemoryStream()
    ctx = Context(stream)
    batches: list[list[BaseEvent]] = []

    async def callback(events: list[BaseEvent], ctx: Context) -> None:
        batches.append(events)

    trigger = EveryNEvents(5)
    sub_id = trigger.attach(stream, callback)

    await ctx.send(_Marker(value=1))
    await ctx.send(_Marker(value=2))

    trigger.detach(stream, sub_id)
    assert trigger._buffer == []
