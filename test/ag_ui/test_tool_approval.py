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
from ag2.ag_ui import AGUIStream
from ag2.ag_ui.interrupts import NOT_PROVEN, TOOL_APPROVAL_REASON
from ag2.events import ToolCallEvent
from ag2.middleware import approval_required
from ag2.testing import TestConfig

pytest.importorskip("starlette")

from test.ag_ui.driving import (  # noqa: E402
    answer,
    app_for,
    ask_once,
    only,
    outcome_of,
    post_run,
    resolved,
    run_body,
    types_of,
)

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
        "reason": TOOL_APPROVAL_REASON,
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
        app = app_for(AGUIStream(agent))

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

        assert outcome_of(events) == {"type": "success"}
        assert ran == []
