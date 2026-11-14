# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager

import pytest

from ag2 import Agent, Context, MemoryStream
from ag2.events import BaseEvent, HaltEvent, ObserverCompleted, TaskCancelled, TaskStarted, ToolCallEvent
from ag2.middleware import BaseMiddleware, Middleware
from ag2.testing import TestConfig


class CleanupScope(Middleware):
    def __init__(self, cleaning: asyncio.Event, release: asyncio.Event, cleaned: asyncio.Event) -> None:
        super().__init__(BaseMiddleware)
        self.cleaning, self.release, self.cleaned = cleaning, release, cleaned

    @asynccontextmanager
    async def scope(self, event: BaseEvent, context: Context) -> AsyncGenerator[None]:
        try:
            yield
        finally:
            self.cleaning.set()
            await self.release.wait()
            self.cleaned.set()


class CleanupObserver:
    def __init__(self, cleaning: asyncio.Event, release: asyncio.Event, cleaned: asyncio.Event) -> None:
        self.cleaning, self.release, self.cleaned = cleaning, release, cleaned

    def register(self, stack: ExitStack | AsyncExitStack, context: Context) -> None:
        stack.enter_context(context.stream.where(ObserverCompleted).sub_scope(self.close, sync_to_thread=False))

    async def close(self) -> None:
        self.cleaning.set()
        await self.release.wait()
        self.cleaned.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("repeats", [1, 2, 3])
async def test_repeated_targeted_halt_preserves_async_tool_cleanup(nested: bool, repeats: int) -> None:
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()
    starts: list[TaskStarted] = []
    cleaned_at_terminal: list[bool] = []

    async def record(event: TaskStarted) -> None:
        starts.append(event)

    async def record_terminal(event: TaskCancelled) -> None:
        cleaned_at_terminal.append(cleaned.is_set())

    async def work() -> str:
        """Hold work until cancelled, then finish asynchronous cleanup."""
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            cleaned.set()
        return "unreachable"

    child = Agent("child", config=TestConfig(ToolCallEvent(name="work", arguments="{}")), tools=[work])
    if nested:
        child = Agent(
            "middle",
            config=TestConfig(ToolCallEvent(name="task_child", arguments='{"objective":"work"}'), "unreachable"),
            tools=[child.as_tool(description="Work")],
        )
    parent = Agent(
        "parent",
        config=TestConfig(ToolCallEvent(name=f"task_{child.name}", arguments='{"objective":"work"}'), "done"),
        tools=[child.as_tool(description="Work")],
    )
    stream = MemoryStream()
    stream.where(TaskStarted).subscribe(record, sync_to_thread=False)
    stream.where(TaskCancelled).subscribe(record_terminal, sync_to_thread=False)
    turn = asyncio.create_task(parent.ask("start", stream=stream))
    await asyncio.wait_for(entered.wait(), 2)
    try:
        await starts[0].origin.send(HaltEvent(reason="stop", source="test"))
        await asyncio.wait_for(cleaning.wait(), 2)
        for _ in range(repeats - 1):
            await starts[0].origin.send(HaltEvent(reason="stop again", source="test"))
    finally:
        release.set()
    reply = await asyncio.wait_for(turn, 2)
    assert reply.body == "done"
    assert cleaned.is_set()
    cancelled = [event for event in await reply.history.get_events() if isinstance(event, TaskCancelled)]
    assert len(cancelled) == (2 if nested else 1)
    assert cleaned_at_terminal == [True] * len(cancelled)
    assert all(event.reason == "stop" for event in cancelled)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_caller", [False, True])
@pytest.mark.parametrize("phase", ["tool", "middleware", "observer"])
async def test_ancestor_cancellation_waits_for_already_halted_leaf(cancel_caller: bool, phase: str) -> None:
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()
    starts: list[TaskStarted] = []
    terminals: list[tuple[str, str, bool]] = []

    async def record(event: TaskStarted) -> None:
        starts.append(event)

    async def record_terminal(event: TaskCancelled) -> None:
        terminals.append((event.agent_name, event.reason, cleaned.is_set()))

    async def work() -> str:
        """Hold the leaf's cleanup until the cancellation requests are delivered."""
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            if phase == "tool":
                cleaning.set()
                await release.wait()
                cleaned.set()
        return "unreachable"

    leaf = Agent(
        "leaf",
        config=TestConfig(ToolCallEvent(name="work", arguments="{}")),
        tools=[work],
        middleware=[CleanupScope(cleaning, release, cleaned)] if phase == "middleware" else (),
        observers=[CleanupObserver(cleaning, release, cleaned)] if phase == "observer" else (),
    )
    middle = Agent(
        "middle",
        config=TestConfig(ToolCallEvent(name="task_leaf", arguments='{"objective":"work"}'), "unreachable"),
        tools=[leaf.as_tool(description="Work")],
    )
    parent = Agent(
        "parent",
        config=TestConfig(ToolCallEvent(name="task_middle", arguments='{"objective":"work"}'), "done"),
        tools=[middle.as_tool(description="Work")],
    )
    stream = MemoryStream()
    stream.where(TaskStarted).subscribe(record, sync_to_thread=False)
    stream.where(TaskCancelled).subscribe(record_terminal, sync_to_thread=False)
    turn = asyncio.create_task(parent.ask("start", stream=stream))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await starts[-1].origin.send(HaltEvent(reason="leaf stop", source="test"))
        await asyncio.wait_for(cleaning.wait(), 2)
        if cancel_caller:
            turn.cancel()
        else:
            await starts[0].origin.send(HaltEvent(reason="ancestor stop", source="test"))
    finally:
        release.set()
    if cancel_caller:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(turn, 2)
    else:
        reply = await asyncio.wait_for(turn, 2)
        assert reply.body == "done"
    assert cleaned.is_set()
    assert [name for name, _, _ in terminals] == ["leaf", "middle"]
    assert terminals[0] == ("leaf", "leaf stop", True)
    assert all(was_cleaned for _, _, was_cleaned in terminals)


@pytest.mark.asyncio
async def test_cancelling_delegation_propagates_when_halted_child_returns() -> None:
    cleaning = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()
    starts: list[TaskStarted] = []

    async def record(event: TaskStarted) -> None:
        starts.append(event)

    child = Agent("child", config=TestConfig("done"), middleware=[CleanupScope(cleaning, release, cleaned)])
    delegate = child.as_tool(description="Work")
    stream = MemoryStream()
    stream.where(TaskStarted).subscribe(record, sync_to_thread=False)
    call = ToolCallEvent(name=delegate.name, arguments='{"objective":"work"}')
    waiter = asyncio.create_task(delegate(call, Context(stream)))
    try:
        # The reply is ready, but ask is still waiting for middleware cleanup.
        await asyncio.wait_for(cleaning.wait(), 2)
        await starts[0].origin.send(HaltEvent(reason="stop", source="test"))
        waiter.cancel()
    finally:
        release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiter, 2)
    assert cleaned.is_set()
