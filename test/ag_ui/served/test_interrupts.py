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
from ag_ui.core import UserMessage
from dirty_equals import IsPartialDict, IsStr

from ag2 import Agent, Context
from ag2.ag_ui import AGUIStream
from ag2.events import HumanInputRequest, HumanMessage, ToolCallEvent
from ag2.testing import TestConfig
from test.ag_ui.harness import dispatch_run, every, only, outcome_of, run_input, sole_interrupt, types_of
from test.ag_ui.serving import QUESTION, Clock, answer, app_for, ask_once, asking_agent, post_run, run_body

# Only so a regression fails the test instead of hanging CI until the suite timeout.
_NEVER = 5.0

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
                    "reason": "input_required",
                    "message": QUESTION,
                    "expiresAt": IsStr(),
                })
            ],
        }

    async def test_the_question_declares_what_an_answer_may_be(self) -> None:
        """A client is told the shape it must send, rather than finding out by refusal."""
        agent, _ = asking_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)

        assert interrupt["responseSchema"] == IsPartialDict({"type": "string"})

    async def test_the_tool_call_the_question_paused_is_completed_by_the_later_run(self) -> None:
        """The call is closed in the run that paused it; its result arrives in the later one.

        `@ag-ui/client` refuses a `RUN_FINISHED` while a call is still open, and
        refuses a `TOOL_CALL_END` for a call its own run never started — so the
        end belongs to the first run and only the result to the second.
        """
        agent, _ = asking_agent()
        app = app_for(AGUIStream(agent))

        first = await post_run(app, run_body(thread_id="t1", run_id="r1"))
        second = await post_run(
            app,
            run_body(thread_id="t1", run_id="r2", text=None, resume=answer(sole_interrupt(first), "blue")),
        )

        assert [t for t in types_of(first) if t.startswith("TOOL_CALL")] == [
            "TOOL_CALL_START",
            "TOOL_CALL_ARGS",
            "TOOL_CALL_END",
        ]
        assert [t for t in types_of(second) if t.startswith("TOOL_CALL")] == ["TOOL_CALL_RESULT"]
        # The same call, under a run id the run that opened it never mentioned.
        started = only(first, "TOOL_CALL_START")
        assert only(second, "TOOL_CALL_RESULT") == IsPartialDict({"toolCallId": started["toolCallId"]})
        assert {e["runId"] for e in second if "runId" in e} == {"r2"}

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

    async def test_two_resumes_racing_one_thread_cannot_both_drive_it(self) -> None:
        """Sequentially this is "already answered"; at once it is a race on one turn.

        Which of the two wins is not the contract — that exactly one does, and
        that the turn is driven once, is.
        """
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent))

        first = await post_run(app, run_body(thread_id="t1", run_id="r1"))
        interrupt = sole_interrupt(first)

        both = await asyncio.gather(
            post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, "blue"))),
            post_run(app, run_body(thread_id="t1", run_id="r3", text=None, resume=answer(interrupt, "red"))),
        )

        endings = sorted(types_of(events)[-1] for events in both)
        assert endings == ["RUN_ERROR", "RUN_FINISHED"]
        assert asked.answers in (["blue"], ["red"])


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

        events = await dispatch_run(
            AGUIStream(agent),
            run_input(UserMessage(id="m1", content="go")),
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


def asking_twice_at_once(*, timeout: float | None = None) -> tuple[Agent, dict[str, str], dict[str, asyncio.Event]]:
    """An agent whose model calls two asking tools in one response.

    Returns what each tool was told, and an event per question set as its tool
    starts asking: when a parallel call gets to run is up to the scheduler.
    """
    answers: dict[str, str] = {}
    asking = {"A?": asyncio.Event(), "B?": asyncio.Event()}

    agent = Agent(
        "test_agent",
        config=TestConfig(
            [ToolCallEvent(name="ask_a", arguments="{}"), ToolCallEvent(name="ask_b", arguments="{}")],
            "all done",
        ),
    )

    @agent.tool
    async def ask_a(context: Context) -> str:
        """Ask the first question."""
        asking["A?"].set()
        answers["A?"] = await context.input("A?", timeout=timeout)
        return answers["A?"]

    @agent.tool
    async def ask_b(context: Context) -> str:
        """Ask the second question."""
        asking["B?"].set()
        answers["B?"] = await context.input("B?", timeout=timeout)
        return answers["B?"]

    return agent, answers, asking


class TestQuestionsAskedAtOnce:
    """Parallel tool calls each asking: the questions go out one run at a time."""

    async def test_each_is_the_outcome_of_the_run_that_answers_the_one_before(self) -> None:
        agent, answers, _ = asking_twice_at_once()
        app = app_for(AGUIStream(agent))

        first = sole_interrupt(await post_run(app, run_body(thread_id="t1", run_id="r1")))
        second_run = await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(first, "one")))
        second = sole_interrupt(second_run)
        third_run = await post_run(app, run_body(thread_id="t1", run_id="r3", text=None, resume=answer(second, "two")))

        assert {first["message"], second["message"]} == {"A?", "B?"}
        assert "RUN_ERROR" not in types_of(second_run)
        assert answers == {first["message"]: "one", second["message"]: "two"}
        # The first call's result may land either side of the second question;
        # what matters is that each reaches the client, once.
        results = every(second_run, "TOOL_CALL_RESULT") + every(third_run, "TOOL_CALL_RESULT")
        assert sorted(e["content"] for e in results) == ["one", "two"]
        assert outcome_of(third_run) == {"type": "success"}

    async def test_a_queued_question_advertises_the_timeout_it_has_been_spending(self) -> None:
        """`timeout=` runs from the call, so time spent queued comes off the deadline shown."""
        clock = Clock()
        agent, _, asking = asking_twice_at_once(timeout=60.0)
        app = app_for(AGUIStream(agent, now=clock))

        first = sole_interrupt(await post_run(app, run_body(thread_id="t1", run_id="r1")))
        # Both asked before the clock moves: the run ends on the first question,
        # which can be before the other call has been scheduled at all. Once its
        # tool has started asking, give it the moment it takes to reach the queue.
        await asyncio.wait_for(asyncio.gather(*(e.wait() for e in asking.values())), timeout=_NEVER)
        await asyncio.sleep(0.02)
        asked_at_deadline = clock.ahead(60.0)
        clock.advance(30.0)
        second = sole_interrupt(
            await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(first, "one")))
        )

        assert first["expiresAt"] == asked_at_deadline
        assert second["expiresAt"] == asked_at_deadline


async def test_work_finished_while_the_question_waits_reaches_the_client() -> None:
    """A sibling tool call that ends during the pause has its result sent on resume."""
    question_out, sibling_done = asyncio.Event(), asyncio.Event()
    agent = Agent(
        "test_agent",
        config=TestConfig(
            [ToolCallEvent(name="ask_human", arguments="{}"), ToolCallEvent(name="look_up", arguments="{}")],
            "all done",
        ),
    )

    @agent.tool
    async def ask_human(context: Context) -> str:
        """Ask the human."""
        return await context.input(QUESTION)

    @agent.tool
    async def look_up() -> str:
        """Finish only once the question is out."""
        await question_out.wait()
        sibling_done.set()
        return "looked up"

    app = app_for(AGUIStream(agent))

    first_run = await post_run(app, run_body(thread_id="t1", run_id="r1"))
    question_out.set()
    await asyncio.wait_for(sibling_done.wait(), timeout=_NEVER)
    # The body has returned; give its result the moment it takes to be published.
    await asyncio.sleep(0.02)
    second_run = await post_run(
        app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(sole_interrupt(first_run), "blue"))
    )

    assert "TOOL_CALL_RESULT" not in types_of(first_run)
    assert sorted(e["content"] for e in every(second_run, "TOOL_CALL_RESULT")) == ["blue", "looked up"]
    assert outcome_of(second_run) == {"type": "success"}
