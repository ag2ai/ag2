# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Concurrency contract for ``Agent.ask``.

* Two ``ask()`` calls on the *same* stream serialise — turn N+1 cannot
  start until turn N has finished, so tool subscribers from turn N can't
  bleed into turn N+1.
* Two ``ask()`` calls on *distinct* streams may overlap — there is no
  global lock on the Agent.
"""

import asyncio

import pytest

from autogen.beta import Agent, tool
from autogen.beta.events import ModelMessage, ModelResponse, ToolCallEvent
from autogen.beta.stream import MemoryStream
from autogen.beta.testing import TestConfig


@pytest.mark.asyncio
class TestSharedStreamSerialization:
    async def test_same_stream_turns_do_not_overlap(self) -> None:
        active = 0
        peak = 0

        @tool
        async def slow_step() -> str:
            """Tool that holds for one event-loop tick to expose any overlap."""
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0)
            finally:
                active -= 1
            return "ok"

        config = TestConfig(
            ToolCallEvent(name="slow_step", arguments="{}"),
            ModelResponse(ModelMessage("done-1")),
            ToolCallEvent(name="slow_step", arguments="{}"),
            ModelResponse(ModelMessage("done-2")),
        )
        agent = Agent("shared", config=config, tools=[slow_step])
        stream = MemoryStream()

        await asyncio.gather(
            agent.ask("first", stream=stream),
            agent.ask("second", stream=stream),
        )

        assert peak == 1, "turns on a shared stream must not overlap"

    async def test_distinct_streams_may_overlap(self) -> None:
        gate = asyncio.Event()
        arrivals = 0

        @tool
        async def gated_step() -> str:
            """First arrival opens the gate; both must reach this point."""
            nonlocal arrivals
            arrivals += 1
            if arrivals == 2:
                gate.set()
            await asyncio.wait_for(gate.wait(), timeout=1.0)
            return "ok"

        config = TestConfig(
            ToolCallEvent(name="gated_step", arguments="{}"),
            ModelResponse(ModelMessage("a")),
            ToolCallEvent(name="gated_step", arguments="{}"),
            ModelResponse(ModelMessage("b")),
        )
        agent = Agent("fresh", config=config, tools=[gated_step])

        await asyncio.gather(
            agent.ask("first", stream=MemoryStream()),
            agent.ask("second", stream=MemoryStream()),
        )

        assert arrivals == 2
