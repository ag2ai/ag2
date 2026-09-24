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
from ag2.ag_ui.interrupts import AG2_METADATA_KEY, DEFAULT_RETENTION, NOT_PROVEN, PROOF_KEY, Retention
from ag2.exceptions import HumanInputTimeoutError
from test.ag_ui.harness import only, sole_interrupt, types_of
from test.ag_ui.serving import (
    QUESTION,
    Clock,
    answer,
    app_for,
    ask_once,
    asking_agent,
    post_run,
    resolved,
    run_body,
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
        # Traffic reaches the registry, which is the fast path to eviction; the
        # timer below is what covers a server nothing is talking to.
        await post_run(app, run_body(thread_id="t2", run_id="r2"))

        assert await asked.ending_within() == "cancelled"

    async def test_a_turn_past_its_time_bound_ends_with_no_traffic_at_all(self) -> None:
        """The deadline is kept by the suspended call, not by the next request to arrive.

        Driven on the real clock rather than the hand-advanced one: the point of
        the test is that nothing had to touch the registry for the bound to
        apply, and a hand-advanced clock can only be advanced by the test itself.
        """
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent, retention=Retention(ttl=0.05)))

        await ask_once(app)

        assert await asked.ending_within() == "no answer"
        # The same exception the caller's own `timeout=` raises: one deadline was
        # advertised, so there is one way for it to elapse.
        assert asked.raised is HumanInputTimeoutError

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

    async def test_a_refused_resume_does_not_move_a_turn_ahead_of_one_held_after_it(self) -> None:
        """Putting a turn back is not holding it again.

        Otherwise a client with nothing but a wrong proof could keep its own turn
        at the front of the queue and have another tenant's evicted in its place.
        """
        clock = Clock()
        agent, _ = asking_agent()
        app = app_for(AGUIStream(agent, retention=Retention(ttl=TTL, max_held=2), now=clock))

        first = await ask_once(app, thread_id="t1", run_id="r1")
        # Held a second apart, so the two deadlines are ordered rather than tied:
        # turns expiring at the same instant are evicted in no particular order,
        # which is fair enough — they were about to go together anyway.
        clock.advance(1)
        second = await ask_once(app, thread_id="t2", run_id="r2")
        refused = await post_run(
            app,
            run_body(
                thread_id="t1",
                run_id="r3",
                text=None,
                resume=resolved(first["id"], "blue", metadata={AG2_METADATA_KEY: {PROOF_KEY: "not the one issued"}}),
            ),
        )
        assert only(refused, "RUN_ERROR")["code"] == NOT_PROVEN

        await ask_once(app, thread_id="t3", run_id="r4")

        # The turn held first is the turn evicted first, refusals notwithstanding.
        gone = await post_run(app, run_body(thread_id="t1", run_id="r5", text=None, resume=answer(first, "blue")))
        assert types_of(gone)[-1] == "RUN_ERROR"
        kept = await post_run(app, run_body(thread_id="t2", run_id="r6", text=None, resume=answer(second, "green")))
        assert types_of(kept)[-1] == "RUN_FINISHED"

    async def test_shutdown_cancels_a_held_turn(self) -> None:
        agent, asked = asking_agent()
        stream = AGUIStream(agent)

        await ask_once(app_for(stream))
        await stream.aclose()

        assert await asked.ending_within() == "cancelled"

    async def test_leaving_the_streams_context_cancels_a_held_turn(self) -> None:
        """`build_asgi` returns an endpoint, so closing the stream is the caller's to do."""
        agent, asked = asking_agent()

        async with AGUIStream(agent) as stream:
            await ask_once(app_for(stream))

        assert asked.ending == "cancelled"

    async def test_shutdown_waits_for_the_turns_it_cancels(self) -> None:
        """``aclose`` returning means the cleanup ran, not that it was scheduled.

        Asserted without ``ending_within``: a cancellation merely requested is
        indistinguishable from one that completed if the assertion may wait.
        """
        agent, asked = asking_agent()
        stream = AGUIStream(agent)

        await ask_once(app_for(stream))
        await stream.aclose()

        assert asked.ending == "cancelled"


class TestARetentionThatCannotBeHonoured:
    """Refused where it is written, not one held turn later."""

    @pytest.mark.parametrize("ttl", [0.0, -1.0])
    async def test_a_deadline_that_has_already_passed(self, ttl: float) -> None:
        with pytest.raises(ValueError, match="ttl must be positive"):
            Retention(ttl=ttl)

    async def test_a_capacity_that_holds_nothing(self) -> None:
        with pytest.raises(ValueError, match="max_held must be at least 1"):
            Retention(max_held=0)


async def test_the_callers_own_timeout_still_ends_the_turn_when_it_falls_first() -> None:
    """The two bounds stay independent: whichever elapses first ends the turn."""
    agent, asked = asking_agent(timeout=0.05)
    app = app_for(AGUIStream(agent, retention=Retention(ttl=TTL)))

    interrupt = await ask_once(app)

    assert await asked.ending_within() == "no answer"
    # The caller's own deadline, and not some other way of going unanswered.
    assert asked.raised is HumanInputTimeoutError
    events = await post_run(
        app,
        run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, "blue")),
    )
    assert types_of(events)[-1] == "RUN_ERROR"
