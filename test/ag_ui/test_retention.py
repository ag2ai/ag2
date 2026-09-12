# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""What a held turn costs the server, and what the client is told about it.

A held turn is a suspended coroutine, not a record, so every way one can end
is asserted on the agent's own code rather than on the registry: the proof that
a bound was enforced is that the function stopped running.
"""

import pytest

from ag2.ag_ui import AGUIStream
from ag2.ag_ui.interrupts import DEFAULT_RETENTION, Retention

pytest.importorskip("starlette")

from test.ag_ui.driving import (  # noqa: E402
    QUESTION,
    Clock,
    answer,
    app_for,
    ask_once,
    asking_agent,
    post_run,
    run_body,
    sole_interrupt,
    types_of,
)

pytestmark = pytest.mark.asyncio

TTL = 60.0


class TestTheAdvertisedDeadline:
    async def test_it_is_the_configured_time_bound(self) -> None:
        clock = Clock()
        agent, _ = asking_agent()
        stream = AGUIStream(agent, retention=Retention(ttl=TTL), now=clock)

        interrupt = await ask_once(app_for(stream))

        assert interrupt["expiresAt"] == clock.ahead(TTL)

    async def test_it_falls_back_to_the_documented_default(self) -> None:
        clock = Clock()
        agent, _ = asking_agent()
        stream = AGUIStream(agent, now=clock)

        interrupt = await ask_once(app_for(stream))

        assert interrupt["expiresAt"] == clock.ahead(DEFAULT_RETENTION.ttl)

    async def test_a_shorter_caller_timeout_wins(self) -> None:
        """The client is shown the bound that will in fact apply, whichever it is."""
        clock = Clock()
        agent, _ = asking_agent(timeout=TTL / 2)
        stream = AGUIStream(agent, retention=Retention(ttl=TTL), now=clock)

        interrupt = await ask_once(app_for(stream))

        assert interrupt["expiresAt"] == clock.ahead(TTL / 2)

    async def test_a_longer_caller_timeout_does_not(self) -> None:
        clock = Clock()
        agent, _ = asking_agent(timeout=TTL * 10)
        stream = AGUIStream(agent, retention=Retention(ttl=TTL), now=clock)

        interrupt = await ask_once(app_for(stream))

        assert interrupt["expiresAt"] == clock.ahead(TTL)

    async def test_it_is_measured_from_the_most_recent_question(self) -> None:
        """Not from when the turn was created: answering one question buys the next a full bound."""
        clock = Clock()
        agent, _ = asking_agent(questions=(QUESTION, "And your favourite number?"))
        app = app_for(AGUIStream(agent, retention=Retention(ttl=TTL), now=clock))

        first = await ask_once(app)
        clock.advance(TTL - 1)
        second = await post_run(
            app,
            run_body(thread_id="t1", run_id="r2", text=None, resume=answer(first, "blue")),
        )

        assert sole_interrupt(second)["expiresAt"] == clock.ahead(TTL)


class TestBoundsOnWhatIsHeld:
    async def test_a_turn_past_its_time_bound_is_cancelled(self) -> None:
        clock = Clock()
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent, retention=Retention(ttl=TTL), now=clock))

        await ask_once(app)
        clock.advance(TTL + 1)
        # Any later traffic reaches the registry; nothing sweeps on a timer.
        await post_run(app, run_body(thread_id="t2", run_id="r2"))

        assert await asked.ending_within() == "cancelled"

    async def test_a_turn_past_its_time_bound_is_unreachable(self) -> None:
        clock = Clock()
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent, retention=Retention(ttl=TTL), now=clock))

        interrupt = await ask_once(app)
        clock.advance(TTL + 1)
        events = await post_run(
            app,
            run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, "blue")),
        )

        assert types_of(events)[-1] == "RUN_ERROR"
        assert asked.answers == []

    async def test_holding_past_the_maximum_evicts_the_oldest(self) -> None:
        clock = Clock()
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent, retention=Retention(ttl=TTL, max_held=1), now=clock))

        first = await ask_once(app, thread_id="t1")
        await ask_once(app, thread_id="t2", run_id="r2")

        assert await asked.ending_within() == "cancelled"
        refused = await post_run(
            app,
            run_body(thread_id="t1", run_id="r3", text=None, resume=answer(first, "blue")),
        )
        assert types_of(refused)[-1] == "RUN_ERROR"

    async def test_shutdown_cancels_a_held_turn(self) -> None:
        agent, asked = asking_agent()
        stream = AGUIStream(agent)

        await ask_once(app_for(stream))
        await stream.aclose()

        assert await asked.ending_within() == "cancelled"


class TestTheCallersOwnTimeout:
    async def test_it_still_ends_the_turn_when_it_falls_first(self) -> None:
        """The two bounds stay independent: whichever elapses first ends the turn."""
        agent, asked = asking_agent(timeout=0.05)
        app = app_for(AGUIStream(agent, retention=Retention(ttl=TTL)))

        interrupt = await ask_once(app)

        assert await asked.ending_within() == "no answer"
        events = await post_run(
            app,
            run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, "blue")),
        )
        assert types_of(events)[-1] == "RUN_ERROR"
