# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Concurrency contract for ``Agent.ask``.

* Two ``ask()`` calls on the *same* stream serialise — turn N+1 cannot
  start until turn N has finished, so tool subscribers from turn N can't
  bleed into turn N+1. "Same stream" means the same ``id``, which is what
  ``History`` is keyed by — not the same Python object.
* Two ``ask()`` calls on *distinct* streams may overlap — there is no
  global lock on the Agent.
"""

import asyncio
from collections.abc import Callable
from uuid import uuid4

import pytest

from ag2 import Agent, tool
from ag2.events import ModelMessage, ModelResponse, ToolCallEvent
from ag2.history import MemoryStorage
from ag2.stream import MemoryStream
from ag2.testing import TestConfig


def _one_object() -> tuple[MemoryStream, MemoryStream]:
    """Both turns handed the very same stream object."""
    stream = MemoryStream()
    return stream, stream


def _two_objects_one_identity() -> tuple[MemoryStream, MemoryStream]:
    """A stream reconstructed from storage, beside the live one.

    Same ``(id, storage)`` pair, so the two handles share one history — which
    makes them one stream, whatever the caller is holding.
    """
    storage = MemoryStorage()
    stream_id = uuid4()
    return (
        MemoryStream(storage=storage, id=stream_id),
        MemoryStream(storage=storage, id=stream_id),
    )


@pytest.mark.asyncio
class TestSharedStreamSerialization:
    @pytest.mark.parametrize(
        "handles",
        [_one_object, _two_objects_one_identity],
        ids=["one-object", "two-objects-one-id"],
    )
    async def test_turns_on_one_stream_do_not_overlap(
        self, handles: Callable[[], tuple[MemoryStream, MemoryStream]]
    ) -> None:
        active = 0
        peak = 0

        @tool
        async def slow_step() -> str:
            """Tool that holds long enough for a racing turn to arrive."""
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.05)
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
        first, second = handles()

        await asyncio.gather(
            agent.ask("first", stream=first),
            agent.ask("second", stream=second),
        )

        assert peak == 1, "turns on one stream must not overlap, whichever object holds it"

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
