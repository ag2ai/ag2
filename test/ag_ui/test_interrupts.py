# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A served agent asks its AG-UI client a question, and is answered.

Driven the way a client drives it: two sequential POSTs against the built ASGI
application, over in-process HTTP. Nothing here asserts how a paused turn is
held — only what a client can see on the wire, and whether the agent's own code
advanced.
"""

import asyncio

import httpx
import pytest
from ag_ui.core import Interrupt, UserMessage
from anyio import create_memory_object_stream
from dirty_equals import IsPartialDict, IsStr

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.events import BaseEvent, HumanInputRequest, HumanMessage
from ag2.exceptions import HumanInputError
from ag2.testing import TestConfig
from test.ag_ui.utils import collect_events, create_run_input

# Only so a regression fails the test instead of hanging CI until the suite timeout.
_NEVER = 5.0

pytest.importorskip("starlette")

from ag2.ag_ui.interrupts import ServedTurn, ServedTurns, TurnOutput  # noqa: E402
from test.ag_ui.driving import (  # noqa: E402
    QUESTION,
    answer,
    app_for,
    asking_agent,
    only,
    outcome_of,
    post_run,
    run_body,
    sole_interrupt,
    types_of,
)

pytestmark = pytest.mark.asyncio


class TestOneQuestionOneAnswer:
    async def test_the_question_ends_the_exchange_as_an_interrupt(self) -> None:
        agent, _ = asking_agent()
        app = app_for(AGUIStream(agent))

        events = await post_run(app, run_body(thread_id="t1", run_id="r1"))

        assert types_of(events)[-1] == "RUN_FINISHED"
        assert "RUN_ERROR" not in types_of(events)
        assert outcome_of(events) == {
            "type": "interrupt",
            "interrupts": [
                IsPartialDict({
                    "id": IsStr(),
                    "reason": "human_input",
                    "message": QUESTION,
                    "expiresAt": IsStr(),
                })
            ],
        }

    async def test_a_later_run_on_the_same_thread_answers_it(self) -> None:
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent))

        first = await post_run(app, run_body(thread_id="t1", run_id="r1"))

        second = await post_run(
            app,
            run_body(thread_id="t1", run_id="r2", text=None, resume=answer(sole_interrupt(first), "blue")),
        )

        assert asked.answers == ["blue"]
        assert outcome_of(second) == {"type": "success"}
        assert only(second, "RUN_STARTED") == IsPartialDict({"threadId": "t1", "runId": "r2"})
        assert only(second, "RUN_FINISHED") == IsPartialDict({"threadId": "t1", "runId": "r2"})

    async def test_the_agent_carries_on_from_where_it_stopped(self) -> None:
        """The answer reaches the waiting call, not a restarted turn."""
        agent, _ = asking_agent()
        app = app_for(AGUIStream(agent))

        first = await post_run(app, run_body(thread_id="t1", run_id="r1"))
        second = await post_run(
            app,
            run_body(thread_id="t1", run_id="r2", text=None, resume=answer(sole_interrupt(first), "blue")),
        )

        # The tool's own return value, carrying the answer, reaches the wire in
        # the resuming exchange — so the call returned rather than being re-run.
        assert only(second, "TOOL_CALL_RESULT")["content"] == "blue"
        assert types_of(first).count("TOOL_CALL_START") == 1
        assert "TOOL_CALL_START" not in types_of(second)


async def test_a_second_question_is_asked_and_answered() -> None:
    agent, asked = asking_agent(questions=(QUESTION, "And your favourite number?"))
    app = app_for(AGUIStream(agent))

    first = await post_run(app, run_body(thread_id="t1", run_id="r1"))
    second = await post_run(
        app,
        run_body(thread_id="t1", run_id="r2", text=None, resume=answer(sole_interrupt(first), "blue")),
    )

    assert sole_interrupt(second) == IsPartialDict({"message": "And your favourite number?"})
    assert sole_interrupt(second)["id"] != sole_interrupt(first)["id"]

    third = await post_run(
        app,
        run_body(thread_id="t1", run_id="r3", text=None, resume=answer(sole_interrupt(second), "7")),
    )

    assert asked.answers == ["blue", "7"]
    assert outcome_of(third) == {"type": "success"}


class TestTheHeldTurnIsFoundByThread:
    async def test_a_resume_on_another_thread_is_refused(self) -> None:
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent))

        first = await post_run(app, run_body(thread_id="t1", run_id="r1"))
        events = await post_run(
            app,
            run_body(thread_id="other", run_id="r2", text=None, resume=answer(sole_interrupt(first), "blue")),
        )

        assert types_of(events)[-1] == "RUN_ERROR"
        assert asked.answers == []

    async def test_retrieving_a_held_turn_removes_it(self) -> None:
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent))

        first = await post_run(app, run_body(thread_id="t1", run_id="r1"))
        interrupt = sole_interrupt(first)

        await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, "blue")))
        again = await post_run(
            app,
            run_body(thread_id="t1", run_id="r3", text=None, resume=answer(interrupt, "red")),
        )

        assert types_of(again)[-1] == "RUN_ERROR"
        assert asked.answers == ["blue"]


class TestRunsThatAskNothing:
    async def test_a_completing_run_states_a_success_outcome(self) -> None:
        agent = Agent("test_agent", config=TestConfig("hello"))

        events = await post_run(app_for(AGUIStream(agent)), run_body(thread_id="t1", run_id="r1"))

        assert outcome_of(events) == {"type": "success"}

    async def test_the_agents_own_hook_answers_in_process(self) -> None:
        """A caller who supplied a hook keeps today's behaviour exactly."""

        async def hook(event: HumanInputRequest) -> str:
            return "green"

        agent, asked = asking_agent(hitl_hook=hook)

        events = await post_run(app_for(AGUIStream(agent)), run_body(thread_id="t1", run_id="r1"))

        assert asked.answers == ["green"]
        assert outcome_of(events) == {"type": "success"}
        assert "RUN_ERROR" not in types_of(events)

    async def test_a_hook_passed_to_dispatch_answers_in_process(self) -> None:
        """Driven at the single-exchange seam: there is no round trip to express."""
        agent, asked = asking_agent()

        async def hook(event: HumanInputRequest) -> HumanMessage:
            return HumanMessage("green")

        events = await collect_events(
            AGUIStream(agent),
            create_run_input(UserMessage(id="m1", content="go")),
            hitl_hook=hook,
        )

        assert asked.answers == ["green"]
        assert outcome_of(events) == {"type": "success"}


async def test_the_agent_says_it_speaks_the_interrupt_protocol() -> None:
    agent, _ = asking_agent()
    app = app_for(AGUIStream(agent))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ag-ui.test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert response.json() == IsPartialDict({
        "humanInTheLoop": IsPartialDict({"supported": True, "interrupts": True}),
    })


async def test_a_second_question_is_refused_and_the_first_still_answers() -> None:
    """A turn carries one outstanding question, and says so when asked for two.

    At the registry seam rather than over in-process HTTP: driving two genuinely
    concurrent asks from the wire needs an agent running parallel subtasks, and
    what is under test is the turn's own invariant, not anything a client sees.

    Serving several at once needs one outcome carrying them all and a resume
    routed per interrupt. Until then the second ask must fail loudly: overwriting
    the slot orphans the first future, hanging that branch until the turn's
    deadline, and sends the second question out on an exchange that already
    ended on the first.
    """
    send, _receive = create_memory_object_stream[BaseEvent](max_buffer_size=10)
    turns = ServedTurns()
    turn = ServedTurn(TurnOutput(thread_id="thread-1", run_id="run-1", send=send))

    def question(n: int) -> Interrupt:
        return Interrupt(id=f"interrupt-{n}", reason="human_input", message=f"Q{n}?")

    first = asyncio.create_task(turns.ask(turn, question(1)))
    await asyncio.sleep(0)  # let the first ask park on its question

    with pytest.raises(HumanInputError, match="already waiting on interrupt interrupt-1"):
        await turns.ask(turn, question(2))

    assert turn.outstanding is not None
    assert turn.outstanding.id == "interrupt-1"

    turn.deliver("still answerable")
    assert await asyncio.wait_for(first, timeout=_NEVER) == "still answerable"
