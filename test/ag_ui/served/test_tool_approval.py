# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A tool call puts itself to the client before it runs, and the client decides.

The second reason a run pauses, on the mechanism the question path already
built: no new lifecycle, one new occasion for it, and a refusal that stops the
tool without failing the turn — the wording being the in-process approval
middleware's own, not a second vocabulary for the same idea.
"""

from typing import Any

import pytest
from dirty_equals import IsPartialDict, IsStr

from ag2 import Agent, Context
from ag2.ag_ui import NOT_PROVEN, TOOL_CALL_REASON, AGUIStream
from ag2.events import ApprovalChoice, ApprovalChoiceMessage, HumanMessage, ToolApprovalRequest, ToolCallEvent
from ag2.middleware import ToolExecution, ToolResultType, approval_required
from ag2.testing import TestConfig
from test.ag_ui.harness import only, outcome_of, types_of
from test.ag_ui.serving import answer, app_for, ask_once, post_run, resolved, run_body

pytestmark = pytest.mark.asyncio

DENIED = "User denied the tool call request"


def gated_agent(**middleware_kwargs: Any) -> tuple[Agent, list[str]]:
    """An agent whose one tool is gated on approval, and a record of it running."""
    ran: list[str] = []

    agent = Agent(
        "test_agent",
        config=TestConfig(ToolCallEvent(name="delete_everything", arguments='{"path": "/"}'), "all done"),
    )

    @agent.tool(middleware=[approval_required(**middleware_kwargs)])
    async def delete_everything(context: Context, path: str) -> str:
        """Delete a path, once a human has said so."""
        ran.append(path)
        return f"deleted {path}"

    return agent, ran


async def test_the_interrupt_names_the_call_it_is_gating() -> None:
    agent, ran = gated_agent()
    app = app_for(AGUIStream(agent))

    events = await post_run(app, run_body(thread_id="t1", run_id="r1"))
    [interrupt] = outcome_of(events)["interrupts"]

    assert interrupt == IsPartialDict({
        "reason": TOOL_CALL_REASON,
        "toolCallId": only(events, "TOOL_CALL_START")["toolCallId"],
        "message": IsStr(),
        "expiresAt": IsStr(),
    })
    assert "delete_everything" in interrupt["message"]
    assert ran == []


class TestTheAnswerDecides:
    async def test_approval_lets_the_tool_run(self) -> None:
        agent, ran = gated_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)
        events = await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, "y")))

        assert ran == ["/"]
        assert only(events, "TOOL_CALL_RESULT")["content"] == "deleted /"
        assert outcome_of(events) == {"type": "success"}

    async def test_refusal_stops_the_tool_without_failing_the_turn(self) -> None:
        agent, ran = gated_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)
        events = await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, "n")))

        assert ran == []
        # The model is told, in the middleware's own words, and the turn goes on.
        assert only(events, "TOOL_CALL_RESULT")["content"] == DENIED
        assert "RUN_ERROR" not in types_of(events)
        assert outcome_of(events) == {"type": "success"}

    async def test_a_boolean_answers_it_too(self) -> None:
        """A client rendering a yes/no button should not have to know the word."""
        agent, ran = gated_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)
        events = await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, True)))

        assert ran == ["/"]
        assert outcome_of(events) == {"type": "success"}

    async def test_a_false_refuses_it(self) -> None:
        agent, ran = gated_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)
        events = await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, False)))

        assert ran == []
        assert only(events, "TOOL_CALL_RESULT")["content"] == DENIED

    async def test_the_configured_refusal_wording_is_the_one_used(self) -> None:
        agent, ran = gated_agent(denied_message="Not allowed by the security desk")
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)
        events = await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, "n")))

        assert ran == []
        assert only(events, "TOOL_CALL_RESULT")["content"] == "Not allowed by the security desk"


class TestItBehavesLikeAnyOtherInterrupt:
    async def test_an_unproven_approval_is_refused(self) -> None:
        agent, ran = gated_agent()
        app = app_for(AGUIStream(agent, require_resume_proof=True))

        interrupt = await ask_once(app)
        events = await post_run(
            app,
            run_body(thread_id="t1", run_id="r2", text=None, resume=resolved(interrupt["id"], "y")),
        )

        assert only(events, "RUN_ERROR") == IsPartialDict({"code": NOT_PROVEN})
        assert ran == []

    async def test_abandoning_it_ends_the_turn(self) -> None:
        agent, ran = gated_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)
        events = await post_run(
            app,
            run_body(
                thread_id="t1",
                run_id="r2",
                text=None,
                resume=[{"interruptId": interrupt["id"], "status": "cancelled", "metadata": interrupt["metadata"]}],
            ),
        )

        assert outcome_of(events) == {"type": "cancelled"}
        assert ran == []


CHOICES = [ApprovalChoice("opt-b", "Allow for this session"), ApprovalChoice("opt-a", "Allow")]


def choosing_agent() -> tuple[Agent, list[HumanMessage]]:
    """An agent whose one tool asks an approval offering `CHOICES`, and a record of the replies."""
    answers: list[HumanMessage] = []

    async def offer(call_next: ToolExecution, event: ToolCallEvent, context: Context) -> ToolResultType:
        request = ToolApprovalRequest("Run it?", tool_call_id=event.id, choices=CHOICES)
        answers.append(await context.ask_message(request))
        return await call_next(event, context)

    agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="run", arguments="{}"), "all done"))

    @agent.tool(middleware=[offer])
    async def run() -> str:
        """Run something."""
        return "ran"

    return agent, answers


async def test_without_choices_the_schema_is_yes_or_no() -> None:
    agent, _ = gated_agent()

    interrupt = await ask_once(app_for(AGUIStream(agent)))

    assert interrupt["responseSchema"] == IsPartialDict({"type": ["string", "boolean"]})


class TestChoices:
    """An approval offering more than yes or no puts each choice in the schema."""

    async def test_the_schema_offers_each_choice(self) -> None:
        agent, _ = choosing_agent()

        interrupt = await ask_once(app_for(AGUIStream(agent)))

        assert interrupt["responseSchema"] == {
            "title": "Approval",
            "description": "true to let the tool call go ahead, false to refuse it, or one of the choices.",
            "anyOf": [
                {"type": "boolean"},
                {
                    "type": "string",
                    "oneOf": [
                        {"const": "opt-b", "title": "Allow for this session"},
                        {"const": "opt-a", "title": "Allow"},
                    ],
                },
            ],
        }

    @pytest.mark.parametrize(
        ("payload", "received"),
        [
            # A choice arrives as picked, so the asker never reads it as a word.
            pytest.param("opt-b", ApprovalChoiceMessage("opt-b"), id="a choice"),
            # The buttons are a plain yes or no, never widened to the first choice.
            pytest.param(True, HumanMessage("y"), id="approve button"),
            pytest.param(False, HumanMessage("n"), id="refuse button"),
            # A string that is not one of the choices is an ordinary answer.
            pytest.param("always", HumanMessage("always"), id="not a choice"),
        ],
    )
    async def test_the_answer_reaches_the_asker(self, payload: str | bool, received: HumanMessage) -> None:
        agent, answers = choosing_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)
        await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, payload)))

        [reply] = answers
        assert (type(reply), reply.content) == (type(received), received.content)
