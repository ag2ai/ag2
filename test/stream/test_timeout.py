# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from collections.abc import AsyncGenerator
from contextlib import suppress
from typing import Annotated

import pytest
from fast_depends import Depends

from ag2 import Context, MemoryStream
from ag2.events import ModelMessage, ToolCallEvent


class WaitingHandler:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.settled = asyncio.Event()

    async def handle(self, event: ModelMessage) -> ModelMessage:
        self.started.set()
        try:
            await self.release.wait()
            return event
        finally:
            self.settled.set()


class ThreadHandler:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.settled = threading.Event()

    def handle(self, event: ModelMessage) -> None:
        self.started.set()
        try:
            self.release.wait()
        finally:
            self.settled.set()


class RecordingHandler:
    def __init__(self) -> None:
        self.events: list[ModelMessage] = []

    def handle(self, event: ModelMessage) -> None:
        self.events.append(event)


async def _resource(ctx: Context) -> AsyncGenerator[str]:
    try:
        if ctx.dependencies["phase"] == "setup":
            await ctx.dependencies["handler"].handle(ModelMessage("setup"))
        yield "resource"
    finally:
        try:
            if ctx.dependencies["phase"] == "cleanup":
                await ctx.dependencies["handler"].handle(ModelMessage("cleanup"))
        finally:
            ctx.dependencies["cleaned_up"].set()


async def _use_resource(event: ModelMessage, ctx: Context, resource: Annotated[str, Depends(_resource)]) -> None:
    if ctx.dependencies["phase"] == "handler":
        await ctx.dependencies["handler"].handle(event)


def _raise_timeout(event: ModelMessage) -> None:
    raise TimeoutError("upstream request timed out")


def _transform(event: ModelMessage) -> ModelMessage:
    return ModelMessage(event.content.upper())


@pytest.mark.asyncio
async def test_successful_timed_handlers_preserve_filtering_and_event_replacement() -> None:
    stream = MemoryStream()
    ctx = Context(stream)
    recorder = RecordingHandler()
    stream.where(ModelMessage).subscribe(_transform, interrupt=True, timeout=1, sync_to_thread=False)
    stream.where(ModelMessage).subscribe(recorder.handle, timeout=1, sync_to_thread=False)

    await ctx.send(ToolCallEvent(name="ignored"))
    await ctx.send(ModelMessage("first"))
    await ctx.send(ModelMessage("second"))

    assert recorder.events == [ModelMessage("FIRST"), ModelMessage("SECOND")]


@pytest.mark.asyncio
@pytest.mark.parametrize("registration", ["subscribe", "decorator", "scope"])
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("interrupt", [False, True])
async def test_timeout_through_subscription_helpers(registration: str, filtered: bool, interrupt: bool) -> None:
    stream = MemoryStream()
    target = stream.where(ModelMessage) if filtered else stream
    handler = WaitingHandler()
    recorder = RecordingHandler()
    ctx = Context(stream)

    if registration == "subscribe":
        sub_id = target.subscribe(handler.handle, interrupt=interrupt, timeout=0.02)
    elif registration == "decorator":
        sub_id = target.subscribe(interrupt=interrupt, timeout=0.02)(handler.handle)
    else:
        with target.sub_scope(handler.handle, interrupt=interrupt, timeout=0.02), pytest.raises(asyncio.TimeoutError):
            await ctx.send(ModelMessage("blocked"))
        stream.subscribe(recorder.handle, sync_to_thread=False)
        await ctx.send(ModelMessage("after scope"))
        assert recorder.events == [ModelMessage("after scope")]
        assert handler.settled.is_set()
        return

    stream.subscribe(recorder.handle, sync_to_thread=False)
    with pytest.raises(asyncio.TimeoutError):
        await ctx.send(ModelMessage("blocked"))
    assert recorder.events == []
    assert handler.started.is_set()
    assert handler.settled.is_set()

    target.unsubscribe(sub_id)
    await ctx.send(ModelMessage("after unsubscribe"))
    assert recorder.events == [ModelMessage("after unsubscribe")]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["setup", "handler", "cleanup"])
async def test_timeout_covers_dependencies_and_cleanup(phase: str) -> None:
    stream = MemoryStream()
    handler = WaitingHandler()
    cleaned_up = asyncio.Event()
    ctx = Context(stream, dependencies={"phase": phase, "handler": handler, "cleaned_up": cleaned_up})
    stream.subscribe(_use_resource, timeout=0.02)

    with pytest.raises(asyncio.TimeoutError):
        await ctx.send(ModelMessage("blocked"))

    assert handler.started.is_set()
    assert handler.settled.is_set()
    assert cleaned_up.is_set()


@pytest.mark.asyncio
async def test_each_invocation_gets_its_own_timeout() -> None:
    stream = MemoryStream()
    ctx = Context(stream)
    handler = WaitingHandler()
    stream.subscribe(handler.handle, timeout=0.02)

    for _ in range(2):
        handler.settled.clear()
        with pytest.raises(asyncio.TimeoutError):
            await ctx.send(ModelMessage("blocked"))
        assert handler.settled.is_set()

    handler.release.set()
    await ctx.send(ModelMessage("released"))


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 60])
async def test_caller_cancellation_propagates(timeout: float | None) -> None:
    stream = MemoryStream()
    ctx = Context(stream)
    handler = WaitingHandler()
    stream.subscribe(handler.handle, timeout=timeout)
    task = asyncio.create_task(ctx.send(ModelMessage("blocked")))
    try:
        await handler.started.wait()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert handler.settled.is_set()


@pytest.mark.asyncio
async def test_handler_raised_timeout_keeps_original_message() -> None:
    stream = MemoryStream()
    stream.subscribe(_raise_timeout, timeout=1)

    with pytest.raises(TimeoutError, match="^upstream request timed out$"):
        await Context(stream).send(ModelMessage("trigger"))


@pytest.mark.asyncio
async def test_sync_timeout_leaves_thread_running_until_released() -> None:
    stream = MemoryStream()
    handler = ThreadHandler()
    stream.subscribe(handler.handle, timeout=0.1)
    task = asyncio.create_task(Context(stream).send(ModelMessage("blocked")))
    try:
        assert await asyncio.to_thread(handler.started.wait, 5)
        with pytest.raises(asyncio.TimeoutError):
            await task
        assert not handler.settled.is_set()
    finally:
        handler.release.set()
        with suppress(asyncio.TimeoutError):
            await task
        assert await asyncio.to_thread(handler.settled.wait, 5)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("filtered", [False, True])
def test_invalid_timeout_is_rejected(timeout: float, filtered: bool) -> None:
    stream = MemoryStream()
    target = stream.where(ModelMessage) if filtered else stream
    with pytest.raises(ValueError, match="finite number > 0 seconds or None"):
        target.subscribe(_raise_timeout, timeout=timeout)
