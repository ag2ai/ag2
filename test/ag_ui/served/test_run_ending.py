# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A served run's body ends cleanly after its terminal event, however the run ended."""

import asyncio

import pytest

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.events import ToolCallEvent
from ag2.testing import TestConfig
from test.ag_ui.harness import every, exploding_agent, outcome_of, types_of
from test.ag_ui.serving import app_for, post_run, run_body

pytestmark = pytest.mark.asyncio


async def test_a_tool_that_raises_ends_the_body_after_run_error() -> None:
    # `post_run` raises on a transport error: an app that re-raises after
    # RUN_ERROR aborts the chunked body under it.
    frames = await post_run(app_for(AGUIStream(exploding_agent())), run_body(thread_id="t1", run_id="r1"))

    assert types_of(frames)[-1] == "RUN_ERROR"


def _stalling_worker(started: asyncio.Event) -> Agent:
    worker = Agent("worker", config=TestConfig(ToolCallEvent(name="stall", arguments="{}"), "done"))

    @worker.tool
    async def stall() -> str:
        """Wait for something that never comes."""
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    return worker


async def test_closing_the_stream_during_a_live_run_ends_it_as_cancelled() -> None:
    started = asyncio.Event()
    worker = _stalling_worker(started)
    parent = Agent(
        "parent",
        config=TestConfig(ToolCallEvent(name="task_worker", arguments='{"objective": "stall"}'), "done"),
        tools=[worker.as_tool(description="Delegate to the worker.")],
    )
    stream = AGUIStream(parent)

    run = asyncio.ensure_future(post_run(app_for(stream), run_body(thread_id="t1", run_id="r1")))
    await started.wait()
    await stream.aclose()
    frames = await run

    assert types_of(frames)[-1] == "RUN_FINISHED"
    assert outcome_of(frames) == {"type": "cancelled"}
    [invocation] = every(frames, "SUBAGENT_STARTED")
    [closed] = every(frames, "SUBAGENT_ERROR")
    assert closed["subagentRunId"] == invocation["subagentRunId"]
