# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""What the served agent declares about itself, before and at the start of each run."""

import asyncio

import pytest
from ag_ui.core import PROTOCOL_VERSION
from dirty_equals import IsPartialDict

from ag2 import Agent, Context
from ag2.ag_ui import UNSUPPORTED_PROTOCOL_VERSION, AGUIStream
from ag2.events import BaseEvent, ClientToolCallEvent, ToolCallEvent
from ag2.middleware import BaseMiddleware, Middleware, ToolExecution, ToolResultType
from ag2.observers import observer
from ag2.testing import TestConfig
from test.ag_ui.harness import every, only, outcome_of, sole_interrupt, types_of, weather_tool
from test.ag_ui.serving import QUESTION, answer, app_for, ask_once, asking_agent, post_run, run_body

pytestmark = pytest.mark.asyncio


async def test_a_refused_version_leaves_a_held_question_answerable() -> None:
    """No run starts, so nothing the thread holds is abandoned by it."""
    agent, asked = asking_agent()
    app = app_for(AGUIStream(agent))

    interrupt = await ask_once(app)
    refused = await post_run(app, {**run_body(thread_id="t1", run_id="r2"), "protocolVersion": "2.0"})
    resumed = await post_run(app, run_body(thread_id="t1", run_id="r3", text=None, resume=answer(interrupt, "blue")))

    assert types_of(refused) == ["RUN_ERROR"]
    assert only(refused, "RUN_ERROR") == IsPartialDict({"code": UNSUPPORTED_PROTOCOL_VERSION})
    assert outcome_of(resumed) == {"type": "success"}
    assert asked.answers == ["blue"]


async def test_a_resuming_run_declares_the_version_too() -> None:
    agent, _ = asking_agent()
    app = app_for(AGUIStream(agent))

    first = await post_run(app, run_body(thread_id="t1", run_id="r1"))
    second = await post_run(
        app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(sole_interrupt(first), "blue"))
    )

    assert only(second, "RUN_STARTED") == IsPartialDict({"protocolVersion": PROTOCOL_VERSION})


class _HoldsClientCalls(BaseMiddleware):
    """Holds each client tool call until `release` is set."""

    def __init__(self, event: BaseEvent, context: Context, release: asyncio.Event) -> None:
        super().__init__(event, context)
        self.release = release

    async def on_tool_execution(
        self, call_next: ToolExecution, event: ToolCallEvent, context: Context
    ) -> ToolResultType:
        if event.name == "get_weather":
            await self.release.wait()
        return await call_next(event, context)


async def test_a_client_call_made_while_the_question_waited_is_pending_on_the_resumed_run() -> None:
    release, made = asyncio.Event(), asyncio.Event()
    agent = Agent(
        "test_agent",
        config=TestConfig([
            ToolCallEvent(name="ask_human", arguments="{}"),
            ToolCallEvent(name="get_weather", arguments='{"location":"Paris"}'),
        ]),
        middleware=[Middleware(_HoldsClientCalls, release=release)],
        observers=[observer(ClientToolCallEvent, lambda _event: made.set(), sync_to_thread=False)],
    )

    @agent.tool
    async def ask_human(context: Context) -> str:
        """Ask the human."""
        return await context.input(QUESTION)

    app = app_for(AGUIStream(agent))
    tools = [weather_tool().model_dump(by_alias=True)]

    first = await post_run(app, {**run_body(thread_id="t1", run_id="r1"), "tools": tools})
    release.set()
    await made.wait()
    resume = run_body(thread_id="t1", run_id="r2", text=None, resume=answer(sole_interrupt(first), "blue"))
    second = await post_run(app, {**resume, "tools": tools})

    assert every(first, "TOOL_CALL_CHUNK") == []
    [chunk] = every(second, "TOOL_CALL_CHUNK")
    assert outcome_of(second) == {"type": "success", "pendingToolCallIds": [chunk["toolCallId"]]}


async def test_a_client_call_made_before_the_question_is_not_pending_on_the_resumed_run() -> None:
    """The call went out in the run that paused: the run that finishes lists only what it started."""
    made = asyncio.Event()
    agent = Agent(
        "test_agent",
        config=TestConfig([
            ToolCallEvent(name="get_weather", arguments='{"location":"Paris"}'),
            ToolCallEvent(name="ask_human", arguments="{}"),
        ]),
        observers=[observer(ClientToolCallEvent, lambda _event: made.set(), sync_to_thread=False)],
    )

    @agent.tool
    async def ask_human(context: Context) -> str:
        """Ask the human."""
        # Parallel calls can start in either order; arrange the scenario this test asserts.
        await made.wait()
        return await context.input(QUESTION)

    app = app_for(AGUIStream(agent))
    tools = [weather_tool().model_dump(by_alias=True)]

    first = await post_run(app, {**run_body(thread_id="t1", run_id="r1"), "tools": tools})
    resume = run_body(thread_id="t1", run_id="r2", text=None, resume=answer(sole_interrupt(first), "blue"))
    second = await post_run(app, {**resume, "tools": tools})

    assert len(every(first, "TOOL_CALL_CHUNK")) == 1
    assert every(second, "TOOL_CALL_CHUNK") == []
    assert outcome_of(second) == {"type": "success"}
