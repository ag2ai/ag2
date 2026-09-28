# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A delegated agent asks the human a question, across the runs that ask and answer it.

The invocation that asks is closed as suspended before its run ends, and the
resuming run announces it again, under the same id, before anything else about it.
"""

from typing import Any

import pytest
from dirty_equals import IsPartialDict

from ag2 import Agent, Context
from ag2.ag_ui import AGUIStream
from ag2.events import ToolCallEvent
from ag2.testing import TestConfig
from test.ag_ui.harness import every, only, outcome_of, sole_interrupt, types_of
from test.ag_ui.serving import QUESTION, abandon, answer, app_for, post_run, run_body

pytestmark = pytest.mark.asyncio


def _delegating_to_an_asker() -> Agent:
    worker = Agent("worker", config=TestConfig(ToolCallEvent(name="ask_human", arguments="{}"), "worked it out"))

    @worker.tool
    async def ask_human(context: Context) -> str:
        """Ask the human."""
        return await context.input(QUESTION)

    return Agent(
        "parent",
        config=TestConfig(ToolCallEvent(name="task_worker", arguments='{"objective": "find out"}'), "summarised"),
        tools=[worker.as_tool(description="Delegate to the worker.")],
    )


def _open_invocations(events: list[dict[str, Any]]) -> set[str]:
    open_ids: set[str] = set()
    for event in events:
        if event["type"] == "SUBAGENT_STARTED":
            open_ids.add(event["subagentRunId"])
        elif event["type"] in ("SUBAGENT_FINISHED", "SUBAGENT_ERROR"):
            open_ids.discard(event["subagentRunId"])
    return open_ids


async def test_the_asking_invocation_is_suspended_before_the_run_ends_on_its_question() -> None:
    app = app_for(AGUIStream(_delegating_to_an_asker()))

    events = await post_run(app, run_body(thread_id="t1", run_id="r1"))

    started = only(events, "SUBAGENT_STARTED")
    interrupt = sole_interrupt(events)
    assert interrupt == IsPartialDict({"subagentRunId": started["subagentRunId"], "message": QUESTION})
    assert only(events, "SUBAGENT_FINISHED") == IsPartialDict({
        "subagentRunId": started["subagentRunId"],
        "outcome": {"type": "suspended", "interruptIds": [interrupt["id"]]},
    })
    assert types_of(events)[-2:] == ["SUBAGENT_FINISHED", "RUN_FINISHED"]
    assert _open_invocations(events) == set()


async def test_the_resuming_run_announces_it_again_and_finishes_it() -> None:
    app = app_for(AGUIStream(_delegating_to_an_asker()))

    first = await post_run(app, run_body(thread_id="t1", run_id="r1"))
    second = await post_run(
        app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(sole_interrupt(first), "blue"))
    )

    invocation = only(first, "SUBAGENT_STARTED")["subagentRunId"]
    assert types_of(second)[:2] == ["RUN_STARTED", "SUBAGENT_STARTED"]
    assert only(second, "SUBAGENT_STARTED") == IsPartialDict({
        "subagentRunId": invocation,
        "name": "worker",
        "description": "find out",
    })
    assert only(second, "SUBAGENT_FINISHED") == IsPartialDict({"subagentRunId": invocation, "result": "worked it out"})
    assert outcome_of(second) == {"type": "success"}
    assert _open_invocations(second) == set()


async def test_abandoning_its_question_closes_it_before_the_cancelled_run_ends() -> None:
    app = app_for(AGUIStream(_delegating_to_an_asker()))

    first = await post_run(app, run_body(thread_id="t1", run_id="r1"))
    second = await post_run(
        app, run_body(thread_id="t1", run_id="r2", text=None, resume=abandon(sole_interrupt(first)))
    )

    invocation = only(first, "SUBAGENT_STARTED")["subagentRunId"]
    assert types_of(second) == ["RUN_STARTED", "SUBAGENT_STARTED", "SUBAGENT_ERROR", "RUN_FINISHED"]
    assert every(second, "SUBAGENT_ERROR") == [IsPartialDict({"subagentRunId": invocation})]
    assert outcome_of(second) == {"type": "cancelled"}


async def test_a_question_the_parent_asks_itself_names_no_invocation() -> None:
    parent = Agent("parent", config=TestConfig(ToolCallEvent(name="ask_human", arguments="{}"), "done"))

    @parent.tool
    async def ask_human(context: Context) -> str:
        """Ask the human."""
        return await context.input(QUESTION)

    events = await post_run(app_for(AGUIStream(parent)), run_body(thread_id="t1", run_id="r1"))

    assert "subagentRunId" not in sole_interrupt(events)
