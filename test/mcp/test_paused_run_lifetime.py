# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""What bounds a paused run, and what happens at each bound.

Two bounds apply, deliberately unequal in kind: the state's lifetime bounds how
long the pause can be resumed at all, and ``context.input(timeout=)`` bounds how
long the asking tool waits. Whichever elapses first ends the turn; the other
must not then report a second, contradictory failure.
"""

import asyncio

import pytest
from mcp.server.request_state import RequestStateSecurity
from mcp.shared.exceptions import MCPError
from mcp.types import InputRequiredResult

from ag2.mcp import MCPServer
from ag2.mcp.pause import PausedRuns, SuspendedTurn
from ag2.mcp.sessions import SessionConfig
from ag2.mcp.testing import connect_modern
from ag2.stream import MemoryStream

from ._helpers import (
    Asked,
    Clock,
    accepting,
    answer,
    ask,
    asking_agent,
    declares_elicitation,
    first_text,
    parks_until_cancelled,
    settle,
)

# Short enough that a test can outlast it without a meaningful pause, long enough
# that no plausible scheduling delay makes the *first* round arrive late.
BRIEF = 0.05
PAST_BRIEF = 0.2


@pytest.mark.asyncio
class TestTheStatesLifetimeBoundsTheRun:
    async def test_a_retry_after_the_state_expired_is_a_protocol_error(self) -> None:
        """Not a resume and not a hang: the round is refused where the token is checked."""
        server = MCPServer(asking_agent(), request_state_security=RequestStateSecurity.ephemeral(ttl=BRIEF))

        async with connect_modern(server, raise_exceptions=False, elicitation_callback=declares_elicitation) as session:
            first = await ask(session)
            assert isinstance(first, InputRequiredResult)
            await asyncio.sleep(PAST_BRIEF)

            with pytest.raises(MCPError) as raised:
                await answer(session, first, accepting("blue"))

        assert "requestState" in str(raised.value)

    async def test_the_state_expiring_reclaims_the_run_it_named(self) -> None:
        """The clock is injected rather than waited on: this is the registry's own bound."""
        clock = Clock()
        runs = PausedRuns(ttl=10.0, clock=clock)
        closed: list[str] = []
        abandoned = SuspendedTurn(conversation=None, stream=MemoryStream(), created=runs.now())
        abandoned.start(parks_until_cancelled(closed))
        runs.register(abandoned)
        await settle()

        clock.advance(11.0)
        runs.register(SuspendedTurn(conversation=None, stream=MemoryStream(), created=runs.now()))
        await settle()

        assert closed == ["closed"], "the expired run's turn scope was left open"
        assert runs.take(abandoned.id) is None

    async def test_a_run_that_pauses_again_is_held_for_its_newest_state(self) -> None:
        """Retention runs from the state a client actually holds, not from the first one.

        Every round mints a fresh ``requestState``, so measuring from the first pause
        would reclaim a run under a token the boundary still accepts.
        """
        clock = Clock()
        runs = PausedRuns(ttl=10.0, clock=clock)
        closed: list[str] = []
        turn = SuspendedTurn(conversation=None, stream=MemoryStream(), created=runs.now())
        turn.start(parks_until_cancelled(closed))
        runs.register(turn)
        await settle()

        clock.advance(8.0)
        # A round arrives, does not finish the turn, and the run pauses again
        # under freshly minted state.
        assert runs.take(turn.id) is turn
        runs.register(turn)

        # Sixteen seconds since the first pause, eight since the last one.
        clock.advance(8.0)
        runs.register(SuspendedTurn(conversation=None, stream=MemoryStream(), created=runs.now()))
        await settle()

        assert closed == [], "a run resumable by the state its client holds was reclaimed"
        assert runs.take(turn.id) is turn

    async def test_a_run_is_reclaimed_when_its_conversation_is_evicted(self) -> None:
        """Neither bound elapsed — the registry that names the conversation dropped it."""
        asked = Asked()
        server = MCPServer(asking_agent(asked), sessions=SessionConfig(max_sessions=1))

        async with connect_modern(server, raise_exceptions=False, elicitation_callback=declares_elicitation) as session:
            abandoned = await ask(session)
            assert isinstance(abandoned, InputRequiredResult)
            # A second call names no conversation, so it mints one — and the
            # registry holds one, so the abandoned run's conversation goes.
            await ask(session)
            await settle()

            assert asked.outcomes == ["CancelledError"], "the abandoned run outlived its conversation"

            with pytest.raises(MCPError):
                await ask(session, request_state=abandoned.request_state)


@pytest.mark.asyncio
class TestTheTwoBoundsDoNotFight:
    async def test_the_input_timeout_spans_the_clients_side_of_the_round_trip(self) -> None:
        """``context.input(timeout=)`` keeps its meaning, now measured across the wire."""
        asked = Asked()
        server = MCPServer(asking_agent(asked, timeout=BRIEF))

        async with connect_modern(server, raise_exceptions=False, elicitation_callback=declares_elicitation) as session:
            first = await ask(session)
            assert isinstance(first, InputRequiredResult)
            await asyncio.sleep(PAST_BRIEF)
            late = await answer(session, first, accepting("blue"))

        assert asked.outcomes == ["HumanInputTimeoutError"]
        assert not isinstance(late, InputRequiredResult)
        assert late.is_error is True
        assert "Nobody answered the human-input request" in first_text(late)

    async def test_a_timed_out_turn_reports_the_timeout_and_not_an_expired_state(self) -> None:
        """The state is still valid, so the answer arrives — and finds the turn already over."""
        server = MCPServer(
            asking_agent(timeout=BRIEF),
            request_state_security=RequestStateSecurity.ephemeral(ttl=30.0),
        )

        async with connect_modern(server, raise_exceptions=False, elicitation_callback=declares_elicitation) as session:
            first = await ask(session)
            assert isinstance(first, InputRequiredResult)
            await asyncio.sleep(PAST_BRIEF)
            late = await answer(session, first, accepting("blue"))

        assert late.is_error is True
        assert "requestState" not in first_text(late)

    async def test_an_expired_state_reclaims_the_turn_without_a_second_failure(self) -> None:
        """The state bound elapsed first, so the timeout must not also fire."""
        asked = Asked()
        server = MCPServer(
            asking_agent(asked, timeout=30.0),
            request_state_security=RequestStateSecurity.ephemeral(ttl=BRIEF),
        )

        async with connect_modern(server, raise_exceptions=False, elicitation_callback=declares_elicitation) as session:
            first = await ask(session)
            assert isinstance(first, InputRequiredResult)
            await asyncio.sleep(PAST_BRIEF)
            # A fresh call sweeps the registry, which is where the unreachable
            # run is reclaimed.
            await ask(session)
            await settle()

        assert asked.outcomes == ["CancelledError"], "the reclaimed run reported a timeout as well"
