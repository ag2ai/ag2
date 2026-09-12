# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A served turn's lifetime belongs to the server, not to the HTTP exchange.

Nothing here is user-visible: it is the prefactor the interrupt work stands on.
An interrupt is delivered as the *terminating* event of its exchange, so the
response closes by design while the agent is still suspended mid-function — and
under the old shape, where the agent ran inside the exchange's task group, that
is exactly when it was cancelled.

This is the one place in that work driven through the single-exchange seam
rather than in-process HTTP: it ships before any interrupt exists, so at the HTTP
seam there is nothing yet that can end an exchange early. The response is closed
by hand instead, and what is asserted is that the agent reaches code past it.
"""

import asyncio
import gc
from collections.abc import AsyncIterator
from typing import Any

import pytest
from ag_ui.core import UserMessage

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.events import ToolCallEvent
from ag2.testing import TestConfig
from ag2.tools import tool
from test.ag_ui.utils import create_run_input

pytestmark = pytest.mark.asyncio

# Every wait below is on an event another task sets. The bound is only so a
# regression fails the test instead of hanging CI until the suite timeout.
_NEVER = 5.0


async def _drain_until(events: AsyncIterator[str], event_type: str) -> None:
    """Read the response until ``event_type`` has been delivered."""
    async for raw in events:
        if f'"{event_type}"' in raw:
            return
    raise AssertionError(f"the run ended without emitting {event_type}")


async def test_turn_runs_on_past_the_end_of_its_exchange() -> None:
    gate, reached = asyncio.Event(), asyncio.Event()

    @tool
    async def park() -> str:
        """Wait for the test to let the turn through."""
        await gate.wait()
        reached.set()
        return "through"

    agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="park", arguments="{}"), "done"), tools=[park])
    stream = AGUIStream(agent)

    events = stream.dispatch(create_run_input(UserMessage(id="m1", content="go")))
    await _drain_until(events, "TOOL_CALL_ARGS")
    await events.aclose()

    gate.set()
    async with asyncio.timeout(_NEVER):
        await reached.wait()

    await stream.aclose()


async def test_a_turn_nobody_awaits_does_not_report_an_unretrieved_exception() -> None:
    """A turn left running raises where nobody waits; that must be consumed."""
    gate, reached = asyncio.Event(), asyncio.Event()

    @tool
    async def park() -> str:
        """Fail once the test lets the turn through."""
        await gate.wait()
        reached.set()
        raise RuntimeError("nobody is listening")

    agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="park", arguments="{}")), tools=[park])
    stream = AGUIStream(agent)

    reported: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reported.append(context))
    try:
        events = stream.dispatch(create_run_input(UserMessage(id="m1", content="go")))
        await _drain_until(events, "TOOL_CALL_ARGS")
        await events.aclose()

        gate.set()
        async with asyncio.timeout(_NEVER):
            await reached.wait()

        await stream.aclose()
        # Let the turn unwind and be collected: an unretrieved task exception is
        # only reported when the task is finalised, which no assertion can force.
        for _ in range(3):
            await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous)

    assert reported == []


async def test_a_turn_still_running_is_cancelled_on_shutdown() -> None:
    gate, parked, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    @tool
    async def park() -> str:
        """Park until cancelled, and say so on the way out."""
        parked.set()
        try:
            await gate.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "through"

    agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="park", arguments="{}"), "done"), tools=[park])
    stream = AGUIStream(agent)

    events = stream.dispatch(create_run_input(UserMessage(id="m1", content="go")))
    await _drain_until(events, "TOOL_CALL_ARGS")
    await events.aclose()
    async with asyncio.timeout(_NEVER):
        await parked.wait()

    await stream.aclose()

    async with asyncio.timeout(_NEVER):
        await cancelled.wait()
