# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A run served over A2UI's AG-UI transport always ends, and its body closes cleanly.

The same contract as ``test/ag_ui/served/test_run_ending.py``, against the other
AG-UI server: the exchange underneath is shared, and what is tested here is that
this transport reaches it.
"""

import asyncio

import pytest

pytest.importorskip("ag_ui")
pytest.importorskip("starlette")

from ag2 import Agent  # noqa: E402
from ag2.a2ui import A2UIServer  # noqa: E402
from ag2.a2ui.transports import AgUiTransport  # noqa: E402
from ag2.events import ToolCallEvent  # noqa: E402
from ag2.testing import TestConfig  # noqa: E402
from test.ag_ui.harness import every, exploding_agent, outcome_of, types_of  # noqa: E402
from test.ag_ui.serving import post_run, run_body, shut_down  # noqa: E402

pytestmark = pytest.mark.asyncio


async def test_a_part_that_cannot_be_decoded_ends_the_run_with_run_error() -> None:
    server = A2UIServer(Agent("ui", config=TestConfig("never reached")), transport=AgUiTransport())
    body = run_body(thread_id="t1", run_id="r1")
    body["messages"] = [
        {
            "id": "m1",
            "role": "user",
            "content": [{"type": "image", "source": {"type": "data", "value": "a", "mimeType": "image/png"}}],
        }
    ]

    frames = await post_run(server, body)

    assert types_of(frames) == ["RUN_STARTED", "RUN_ERROR"]


async def test_a_tool_that_raises_ends_the_body_after_run_error() -> None:
    server = A2UIServer(exploding_agent(), transport=AgUiTransport(), validate_responses=False)

    frames = await post_run(server, run_body(thread_id="t1", run_id="r1"))

    assert types_of(frames)[-1] == "RUN_ERROR"


async def test_shutting_down_during_a_live_run_ends_it_as_cancelled() -> None:
    started = asyncio.Event()
    worker = Agent("worker", config=TestConfig(ToolCallEvent(name="stall", arguments="{}"), "done"))

    @worker.tool
    async def stall() -> str:
        """Wait for something that never comes."""
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    parent = Agent(
        "ui",
        config=TestConfig(ToolCallEvent(name="task_worker", arguments='{"objective": "stall"}'), "done"),
        tools=[worker.as_tool(description="Delegate to the worker.")],
    )
    server = A2UIServer(parent, transport=AgUiTransport(), validate_responses=False)

    run = asyncio.ensure_future(post_run(server, run_body(thread_id="t1", run_id="r1")))
    await started.wait()
    await shut_down(server)
    frames = await run

    assert outcome_of(frames) == {"type": "cancelled"}
    [invocation] = every(frames, "SUBAGENT_STARTED")
    [closed] = every(frames, "SUBAGENT_ERROR")
    assert closed["subagentRunId"] == invocation["subagentRunId"]
