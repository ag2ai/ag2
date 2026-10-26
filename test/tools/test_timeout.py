# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from collections.abc import AsyncGenerator
from typing import Annotated

import pytest
from fast_depends import Depends
from freezegun import freeze_time

from ag2 import Agent, Context, tool
from ag2.events import ToolCallEvent, ToolErrorEvent, ToolResultEvent, ToolResultsEvent
from ag2.middleware import ToolExecution, ToolResultType
from ag2.testing import TestConfig, TrackingConfig


async def _wait_for_release(context: Context) -> str:
    context.dependencies["started"].set()
    try:
        await context.dependencies["release"].wait()
        return "released"
    finally:
        context.dependencies["settled"].set()


def _immediate() -> str:
    return "ok"


def _sync_wait_for_release(context: Context) -> str:
    context.dependencies["started"].set()
    try:
        context.dependencies["release"].wait()
        return "late result"
    finally:
        context.dependencies["settled"].set()


def _raise_timeout() -> str:
    raise TimeoutError("upstream request timed out")


async def _resource(context: Context) -> AsyncGenerator[str]:
    try:
        yield "resource"
    finally:
        await asyncio.sleep(0)
        context.dependencies["cleaned_up"].set()


async def _use_resource(context: Context, resource: Annotated[str, Depends(_resource)]) -> str:
    return await _wait_for_release(context)


async def _wait_in_middleware(call_next: ToolExecution, event: ToolCallEvent, context: Context) -> ToolResultType:
    context.dependencies["started"].set()
    await context.dependencies["release"].wait()
    return await call_next(event, context)


@pytest.mark.asyncio
async def test_timeout_returns_error_and_parallel_sibling_result() -> None:
    call = ToolCallEvent(name="work")
    sibling = ToolCallEvent(name="sibling")
    config = TrackingConfig(TestConfig([call, sibling], "done", raise_tool_errors=False))
    settled = asyncio.Event()
    cleaned_up = asyncio.Event()
    started = asyncio.Event()
    agent = Agent(
        "test",
        config=config,
        tools=[tool(_immediate, name="sibling")],
        dependencies={
            "started": started,
            "release": asyncio.Event(),
            "settled": settled,
            "cleaned_up": cleaned_up,
        },
    )
    agent.tool(name="work", timeout=30)(_use_resource)

    # Scanning Daytona's lazy exports imports Pydantic models under FakeDatetime.
    with freeze_time(ignore=["daytona"]) as clock:
        task = asyncio.create_task(agent.ask("Run both tools"))
        await started.wait()
        clock.tick(31)
        reply = await task

    assert reply.body == "done"
    assert settled.is_set()
    assert cleaned_up.is_set()
    results = config.mock.call_args_list[1].args[0]
    assert isinstance(results, ToolResultsEvent)
    assert results.results == [
        ToolErrorEvent.from_call(call, TimeoutError("Tool 'work' timed out after 30 seconds")),
        ToolResultEvent.from_call(sibling, "ok"),
    ]


@pytest.mark.parametrize("timeout", [0, float("nan")], ids=["non_positive", "non_finite"])
def test_invalid_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="`timeout` must be a finite number > 0 seconds or None"):
        tool(_immediate, timeout=timeout)


@pytest.mark.asyncio
async def test_sync_timeout_stops_waiting_but_thread_finishes_later() -> None:
    started = threading.Event()
    release = threading.Event()
    settled = threading.Event()
    call = ToolCallEvent(name="work")
    config = TestConfig(call, "done", raise_tool_errors=False)
    agent = Agent(
        "test",
        config=config,
        tools=[tool(_sync_wait_for_release, name="work", timeout=30)],
        dependencies={"started": started, "release": release, "settled": settled},
    )
    with freeze_time(ignore=["daytona"]) as clock:
        task = asyncio.create_task(agent.ask("Run work"))
        try:
            await asyncio.to_thread(started.wait)
            clock.tick(31)
            reply = await task
            assert reply.body == "done"
            assert not settled.is_set()
        finally:
            release.set()
            await asyncio.to_thread(settled.wait)


@pytest.mark.asyncio
async def test_caller_cancellation_propagates() -> None:
    started = asyncio.Event()
    settled = asyncio.Event()
    agent = Agent(
        "test",
        config=TestConfig(ToolCallEvent(name="work"), "done", raise_tool_errors=False),
        tools=[tool(_wait_for_release, name="work", timeout=60)],
        dependencies={"started": started, "release": asyncio.Event(), "settled": settled},
    )
    task = asyncio.create_task(agent.ask("Run work"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert settled.is_set()


@pytest.mark.asyncio
async def test_tool_raised_timeout_keeps_original_message() -> None:
    agent = Agent(
        "test", config=TestConfig(ToolCallEvent(name="work")), tools=[tool(_raise_timeout, name="work", timeout=1)]
    )
    with pytest.raises(TimeoutError, match="^upstream request timed out$"):
        await agent.ask("Run work")


@pytest.mark.asyncio
async def test_timeout_starts_after_tool_middleware() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    agent = Agent(
        "test",
        config=TestConfig(ToolCallEvent(name="work"), "done"),
        tools=[tool(_immediate, name="work", timeout=30, middleware=[_wait_in_middleware])],
        dependencies={"started": started, "release": release},
    )
    with freeze_time(ignore=["daytona"]) as clock:
        task = asyncio.create_task(agent.ask("Run work"))
        try:
            await started.wait()
            clock.tick(31)
            # Let expired timer callbacks and their waiters run.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
            reply = await task
    assert reply.body == "done"
