# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A client's `state` comes back from a run whole.

A `STATE_SNAPSHOT` replaces the client's state wholesale, so every one the server
sends is the complete state: the client's keys and the server's variables together.
"""

from typing import Any

import pytest
from ag_ui.core import UserMessage

from ag2 import Agent, Context
from ag2.ag_ui import AGUIStream
from ag2.events import ToolCallEvent
from ag2.testing import TestConfig
from test.ag_ui.harness import dispatch_run, every, outcome_of, run_input

pytestmark = pytest.mark.asyncio


def _snapshots(frames: list[dict[str, Any]]) -> list[Any]:
    return [frame["snapshot"] for frame in every(frames, "STATE_SNAPSHOT")]


async def test_every_snapshot_holds_the_client_s_keys_and_the_server_s_variables() -> None:
    agent = Agent("test_agent", config=TestConfig("done"))
    incoming = run_input(UserMessage(id="m1", content="hi"), state={"draft": "hello"})

    frames = await dispatch_run(AGUIStream(agent), incoming, variables={"user_id": "u1"})

    assert _snapshots(frames) == [{"draft": "hello", "user_id": "u1"}]


async def test_a_run_that_changes_a_variable_sends_the_whole_state_back() -> None:
    agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="count"), "done"))

    @agent.tool
    def count(context: Context) -> str:
        """Count a visit."""
        context.variables["visits"] = 1
        return "counted"

    incoming = run_input(UserMessage(id="m1", content="hi"), state={"draft": "hello"})

    frames = await dispatch_run(AGUIStream(agent), incoming)

    assert _snapshots(frames) == [{"draft": "hello", "visits": 1}]


async def test_a_run_that_changes_nothing_sends_the_client_nothing_to_replace() -> None:
    agent = Agent("test_agent", config=TestConfig("done"))
    incoming = run_input(UserMessage(id="m1", content="hi"), state={"draft": "hello"})

    frames = await dispatch_run(AGUIStream(agent), incoming)

    assert _snapshots(frames) == []


async def test_a_null_value_in_state_is_kept() -> None:
    agent = Agent("test_agent", config=TestConfig("done"))
    incoming = run_input(UserMessage(id="m1", content="hi"), state={"keep": None})

    frames = await dispatch_run(AGUIStream(agent), incoming, variables={"user_id": "u1"})

    assert _snapshots(frames) == [{"keep": None, "user_id": "u1"}]


@pytest.mark.parametrize("state", [[1, 2], "draft", 3])
async def test_a_state_that_is_not_an_object_is_served_and_left_as_it_is(state: Any) -> None:
    agent = Agent("test_agent", config=TestConfig("done"))
    incoming = run_input(UserMessage(id="m1", content="hi"), state=state)

    frames = await dispatch_run(AGUIStream(agent), incoming, variables={"user_id": "u1"})

    assert outcome_of(frames) == {"type": "success"}
    assert _snapshots(frames) == []
