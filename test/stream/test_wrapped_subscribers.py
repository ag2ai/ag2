# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, overload

import pytest

from ag2 import Agent, MemoryStream, tool
from ag2.context import SubId
from ag2.events import BaseEvent, ModelResponse, ToolCallEvent, ToolResultEvent
from ag2.events.conditions import Condition
from ag2.testing import TestConfig


class AuditedStream(MemoryStream):
    """Wraps every subscriber to record the events it handles, as instrumentation does."""

    def __init__(self) -> None:
        self.audited: list[BaseEvent] = []
        super().__init__()

    @overload
    def subscribe(
        self,
        func: Callable[..., Any],
        *,
        interrupt: bool = False,
        sync_to_thread: bool = True,
        timeout: float | None = None,
        condition: Condition | None = None,
    ) -> SubId: ...

    @overload
    def subscribe(
        self,
        func: None = None,
        *,
        interrupt: bool = False,
        sync_to_thread: bool = True,
        timeout: float | None = None,
        condition: Condition | None = None,
    ) -> Callable[[Callable[..., Any]], SubId]: ...

    def subscribe(
        self,
        func: Callable[..., Any] | None = None,
        *,
        interrupt: bool = False,
        sync_to_thread: bool = True,
        timeout: float | None = None,
        condition: Condition | None = None,
    ) -> Callable[[Callable[..., Any]], SubId] | SubId:
        if func is None:
            return super().subscribe(
                interrupt=interrupt, sync_to_thread=sync_to_thread, timeout=timeout, condition=condition
            )

        @wraps(func)
        async def audited(event: BaseEvent, *args: Any, **kwargs: Any) -> Any:
            self.audited.append(event)
            result = func(event, *args, **kwargs)
            return await result if inspect.isawaitable(result) else result

        return super().subscribe(
            audited, interrupt=interrupt, sync_to_thread=sync_to_thread, timeout=timeout, condition=condition
        )


@pytest.mark.asyncio
async def test_agent_runs_on_a_stream_that_wraps_its_subscribers() -> None:
    calls: list[tuple[int, int]] = []

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        calls.append((a, b))
        return a + b

    agent = Agent(
        "calculator",
        config=TestConfig(ToolCallEvent(name="add", arguments='{"a": 1, "b": 2}'), "The sum is 3."),
        tools=[add],
    )
    stream = AuditedStream()

    reply = await agent.ask("What is 1 + 2?", stream=stream)

    assert reply.body == "The sum is 3."
    assert calls == [(1, 2)]
    assert {ToolCallEvent, ToolResultEvent, ModelResponse} <= {type(e) for e in stream.audited}
    assert {ToolCallEvent, ToolResultEvent} <= {type(e) for e in await stream.history.get_events()}
