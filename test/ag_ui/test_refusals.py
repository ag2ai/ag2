# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Answers this server will not honour, and the client giving up on the question.

The rule under test throughout: however an exchange goes wrong, the client is
left with an event that ends the run. A stream that simply stops is the one
outcome a client cannot recover from — it waits.
"""

import pytest
from dirty_equals import IsPartialDict

from ag2.ag_ui import AGUIStream
from ag2.ag_ui.interrupts import NOT_OUTSTANDING, NO_HELD_TURN, PAYLOAD_REFUSED, Retention

pytest.importorskip("starlette")

from test.ag_ui.driving import (  # noqa: E402
    QUESTION,
    Clock,
    abandon,
    answer,
    app_for,
    ask_once,
    asking_agent,
    only,
    outcome_of,
    post_run,
    resolved,
    run_body,
    sole_interrupt,
    types_of,
)

pytestmark = pytest.mark.asyncio

TTL = 60.0
SECOND_QUESTION = "And your favourite number?"


async def refusal(app, *, thread_id: str = "t1", run_id: str = "r2", resume: list) -> dict:
    """Drive one exchange expected to be refused, and return its run error."""
    events = await post_run(app, run_body(thread_id=thread_id, run_id=run_id, text=None, resume=resume))
    assert types_of(events)[-1] == "RUN_ERROR", f"the client was left without a terminating event: {types_of(events)}"
    return only(events, "RUN_ERROR")


class TestGivingUp:
    async def test_abandoning_the_question_ends_the_turn(self) -> None:
        """An outcome, not a failure: nobody is answering, and the run is over."""
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)
        events = await post_run(
            app,
            run_body(thread_id="t1", run_id="r2", text=None, resume=abandon(interrupt)),
        )

        assert types_of(events)[-1] == "RUN_FINISHED"
        assert outcome_of(events) == {"type": "success"}
        assert asked.answers == []
        assert await asked.ending_within() == "cancelled"

    async def test_the_abandoned_turn_cannot_be_resumed_afterwards(self) -> None:
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)
        await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=abandon(interrupt)))

        error = await refusal(app, run_id="r3", resume=answer(interrupt, "blue"))

        assert error == IsPartialDict({"code": NO_HELD_TURN})
        assert asked.answers == []


class TestAnswersThatCannotBeHonoured:
    async def test_an_interrupt_nobody_is_holding(self) -> None:
        agent, _ = asking_agent()
        app = app_for(AGUIStream(agent))

        error = await refusal(app, resume=resolved("no-such-interrupt", "blue"))

        assert error == IsPartialDict({"code": NO_HELD_TURN})

    async def test_an_unknown_id_on_a_thread_that_is_holding_one(self) -> None:
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent))

        await ask_once(app)
        error = await refusal(app, resume=resolved("no-such-interrupt", "blue"))

        assert error == IsPartialDict({"code": NOT_OUTSTANDING})
        assert asked.answers == []

    async def test_an_interrupt_that_was_already_answered(self) -> None:
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)
        await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, "blue")))

        error = await refusal(app, run_id="r3", resume=answer(interrupt, "red"))

        assert error == IsPartialDict({"code": NO_HELD_TURN})
        assert asked.answers == ["blue"]

    async def test_an_answer_arriving_after_the_deadline(self) -> None:
        clock = Clock()
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent, retention=Retention(ttl=TTL), now=clock))

        interrupt = await ask_once(app)
        clock.advance(TTL + 1)

        error = await refusal(app, resume=answer(interrupt, "blue"))

        assert error == IsPartialDict({"code": NO_HELD_TURN})
        assert asked.answers == []

    async def test_an_answer_to_an_earlier_round(self) -> None:
        """A late answer from the round before must not be read as this round's."""
        agent, asked = asking_agent(questions=(QUESTION, SECOND_QUESTION))
        app = app_for(AGUIStream(agent))

        first = await ask_once(app)
        second = await post_run(
            app,
            run_body(thread_id="t1", run_id="r2", text=None, resume=answer(first, "blue")),
        )

        error = await refusal(app, run_id="r3", resume=answer(first, "red"))

        assert error == IsPartialDict({"code": NOT_OUTSTANDING})
        assert asked.answers == ["blue"]
        assert sole_interrupt(second) == IsPartialDict({"message": SECOND_QUESTION})

    async def test_a_payload_that_is_not_what_was_asked_for(self) -> None:
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)

        error = await refusal(app, resume=answer(interrupt, {"colour": "blue"}))

        assert error == IsPartialDict({"code": PAYLOAD_REFUSED})
        assert asked.answers == []


class TestWhatARefusalLeavesBehind:
    async def test_a_turn_refused_for_a_bad_payload_is_still_resumable(self) -> None:
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)
        await refusal(app, resume=answer(interrupt, 42))

        events = await post_run(
            app,
            run_body(thread_id="t1", run_id="r3", text=None, resume=answer(interrupt, "blue")),
        )

        assert asked.answers == ["blue"]
        assert outcome_of(events) == {"type": "success"}

    async def test_a_turn_refused_for_a_wrong_id_is_still_resumable(self) -> None:
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)
        await refusal(app, resume=resolved("no-such-interrupt", "blue"))

        events = await post_run(
            app,
            run_body(thread_id="t1", run_id="r3", text=None, resume=answer(interrupt, "blue")),
        )

        assert asked.answers == ["blue"]
        assert outcome_of(events) == {"type": "success"}

    async def test_refusals_do_not_extend_the_deadline(self) -> None:
        """A stream of bad answers cannot keep a turn alive past what its client was shown."""
        clock = Clock()
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent, retention=Retention(ttl=TTL), now=clock))

        interrupt = await ask_once(app)
        clock.advance(TTL - 1)
        await refusal(app, resume=answer(interrupt, 42))
        clock.advance(2)

        error = await refusal(app, run_id="r3", resume=answer(interrupt, "blue"))

        assert error == IsPartialDict({"code": NO_HELD_TURN})
        assert asked.answers == []
