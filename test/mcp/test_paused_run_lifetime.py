# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""What bounds a paused run, and what happens at each bound.

A modern-era client that asks a question and never comes back must not hold an
agent's turn open forever. Two bounds apply, and they are deliberately unequal in
kind: the *state's* lifetime bounds how long the pause can be resumed at all, and
``context.input(timeout=)`` bounds how long the asking tool waits. Whichever
elapses first ends the turn; the other must not then report a second,
contradictory failure.

The retention bound is the state token's TTL and nothing else — once no client
can present a resumable token the run is unreachable, so it is reclaimed. These
tests read that from the client's side wherever they can, and from the registry
directly only where the clock has to be controlled rather than waited on.
"""

import asyncio
from typing import Any

import pytest
from mcp.client.session import ClientRequestContext
from mcp.server.request_state import RequestStateSecurity
from mcp.shared.exceptions import MCPError
from mcp.types import ElicitRequestParams, ElicitResult, InputRequiredResult, TextContent

from ag2 import Agent, Context
from ag2.events import ToolCallEvent
from ag2.mcp import MCPServer
from ag2.mcp.elicitation import ANSWER_FIELD
from ag2.mcp.pause import PausedRuns, SuspendedTurn
from ag2.mcp.sessions import SessionConfig
from ag2.mcp.testing import connect_modern
from ag2.stream import MemoryStream
from ag2.testing import TestConfig

# Short enough that a test can outlast it without a meaningful pause, long enough
# that no plausible scheduling delay makes the *first* round arrive late.
BRIEF = 0.05
PAST_BRIEF = 0.2


def asking_agent(
    *,
    outcomes: list[str] | None = None,
    timeout: float | None = None,
) -> Agent:
    """An agent whose one tool asks a question and records how the ask ended.

    ``outcomes`` gets the name of whatever ended the ask — ``"answered"``, or the
    exception type — which is how these tests tell a reclaimed run from a timed
    out one without reaching into the executor.
    """

    async def ask_human(ctx: Context) -> str:
        try:
            answer = await ctx.input("What colour?", timeout=timeout)
        except BaseException as exc:
            if outcomes is not None:
                outcomes.append(type(exc).__name__)
            raise
        if outcomes is not None:
            outcomes.append("answered")
        return f"human said: {answer}"

    return Agent(
        "asker",
        config=TestConfig(ToolCallEvent(name="ask_human"), "done", raise_tool_errors=False),
        tools=[ask_human],
    )


async def declares_elicitation(context: ClientRequestContext, params: ElicitRequestParams) -> ElicitResult:
    """Supplying a callback is what makes the client declare it can answer."""
    raise AssertionError("these tests answer by retrying, not through the callback")


async def _call(session: Any, **kwargs: Any) -> Any:
    return await session.call_tool("ask", {"message": "go"}, allow_input_required=True, **kwargs)


async def _settle() -> None:
    """Give the loop enough turns to deliver a cancellation into a held run."""
    for _ in range(10):
        await asyncio.sleep(0)


def _first_text(result: Any) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


@pytest.mark.asyncio
class TestTheStatesLifetimeBoundsTheRun:
    async def test_a_retry_after_the_state_expired_is_a_protocol_error(self) -> None:
        """Not a resume and not a hang: the round is refused where the token is checked."""
        server = MCPServer(asking_agent(), request_state_security=RequestStateSecurity.ephemeral(ttl=BRIEF))

        async with connect_modern(server, raise_exceptions=False, elicitation_callback=declares_elicitation) as session:
            first = await _call(session)
            assert isinstance(first, InputRequiredResult)
            (key,) = (first.input_requests or {}).keys()
            await asyncio.sleep(PAST_BRIEF)

            with pytest.raises(MCPError) as raised:
                await _call(
                    session,
                    input_responses={key: ElicitResult(action="accept", content={ANSWER_FIELD: "blue"})},
                    request_state=first.request_state,
                )

        assert "requestState" in str(raised.value)

    async def test_the_state_expiring_reclaims_the_run_it_named(self) -> None:
        """The clock is injected rather than waited on: this is the registry's own bound.

        A run whose state no client can present any more is unreachable, so the
        next registry operation reclaims it and its turn scope closes.
        """
        clock = _Clock()
        runs = PausedRuns(ttl=10.0, clock=clock)
        closed: list[str] = []
        abandoned = SuspendedTurn(conversation=None, stream=MemoryStream(), created=runs.now())
        abandoned.start(_records_when_cancelled(closed))
        runs.register(abandoned)
        await _settle()

        clock.advance(11.0)
        runs.register(SuspendedTurn(conversation=None, stream=MemoryStream(), created=runs.now()))
        await _settle()

        assert closed == ["closed"], "the expired run's turn scope was left open"
        assert runs.take(abandoned.id) is None

    async def test_a_run_that_pauses_again_is_held_for_its_newest_state(self) -> None:
        """Retention runs from the state a client actually holds, not from the first one.

        Every round mints a fresh ``requestState``, so a conversation that pauses
        and resumes several times can outlive the TTL in total while the token
        its client holds is always young. Measuring retention from the run's
        first pause would reclaim it under a token the boundary still accepts —
        the client presents valid state and is told the run is gone.
        """
        clock = _Clock()
        runs = PausedRuns(ttl=10.0, clock=clock)
        closed: list[str] = []
        turn = SuspendedTurn(conversation=None, stream=MemoryStream(), created=runs.now())
        turn.start(_records_when_cancelled(closed))
        runs.register(turn)
        await _settle()

        clock.advance(8.0)
        # A round arrives, does not finish the turn, and the run pauses again
        # under freshly minted state.
        assert runs.take(turn.id) is turn
        runs.register(turn)

        # Sixteen seconds since the first pause, eight since the last one.
        clock.advance(8.0)
        runs.register(SuspendedTurn(conversation=None, stream=MemoryStream(), created=runs.now()))
        await _settle()

        assert closed == [], "a run resumable by the state its client holds was reclaimed"
        assert runs.take(turn.id) is turn

    async def test_a_run_is_reclaimed_when_its_conversation_is_evicted(self) -> None:
        """Neither bound elapsed — the registry that names the conversation dropped it."""
        outcomes: list[str] = []
        server = MCPServer(asking_agent(outcomes=outcomes), sessions=SessionConfig(max_sessions=1))

        async with connect_modern(server, raise_exceptions=False, elicitation_callback=declares_elicitation) as session:
            abandoned = await _call(session)
            assert isinstance(abandoned, InputRequiredResult)
            # A second call names no conversation, so it mints one — and the
            # registry holds one, so the abandoned run's conversation goes.
            await _call(session)
            await _settle()

            assert outcomes == ["CancelledError"], "the abandoned run outlived its conversation"

            with pytest.raises(MCPError):
                await _call(session, request_state=abandoned.request_state)


@pytest.mark.asyncio
class TestTheTwoBoundsDoNotFight:
    async def test_the_input_timeout_spans_the_clients_side_of_the_round_trip(self) -> None:
        """``context.input(timeout=)`` keeps its meaning, now measured across the wire."""
        outcomes: list[str] = []
        server = MCPServer(asking_agent(outcomes=outcomes, timeout=BRIEF))

        async with connect_modern(server, raise_exceptions=False, elicitation_callback=declares_elicitation) as session:
            first = await _call(session)
            assert isinstance(first, InputRequiredResult)
            (key,) = (first.input_requests or {}).keys()
            await asyncio.sleep(PAST_BRIEF)
            late = await _call(
                session,
                input_responses={key: ElicitResult(action="accept", content={ANSWER_FIELD: "blue"})},
                request_state=first.request_state,
            )

        assert outcomes == ["HumanInputTimeoutError"]
        assert not isinstance(late, InputRequiredResult)
        assert late.is_error is True
        assert "Nobody answered the human-input request" in _first_text(late)

    async def test_a_timed_out_turn_reports_the_timeout_and_not_an_expired_state(self) -> None:
        """The state is still valid, so the answer arrives — and finds the turn already over."""
        server = MCPServer(
            asking_agent(timeout=BRIEF),
            request_state_security=RequestStateSecurity.ephemeral(ttl=30.0),
        )

        async with connect_modern(server, raise_exceptions=False, elicitation_callback=declares_elicitation) as session:
            first = await _call(session)
            assert isinstance(first, InputRequiredResult)
            (key,) = (first.input_requests or {}).keys()
            await asyncio.sleep(PAST_BRIEF)
            late = await _call(
                session,
                input_responses={key: ElicitResult(action="accept", content={ANSWER_FIELD: "blue"})},
                request_state=first.request_state,
            )

        assert late.is_error is True
        assert "requestState" not in _first_text(late)

    async def test_an_expired_state_reclaims_the_turn_without_a_second_failure(self) -> None:
        """The state bound elapsed first, so the timeout must not also fire.

        The turn ends by reclamation — a cancellation — and the client has
        already been told the one thing there is to tell it.
        """
        outcomes: list[str] = []
        server = MCPServer(
            asking_agent(outcomes=outcomes, timeout=30.0),
            request_state_security=RequestStateSecurity.ephemeral(ttl=BRIEF),
        )

        async with connect_modern(server, raise_exceptions=False, elicitation_callback=declares_elicitation) as session:
            first = await _call(session)
            assert isinstance(first, InputRequiredResult)
            await asyncio.sleep(PAST_BRIEF)
            # A fresh call sweeps the registry, which is where the unreachable
            # run is reclaimed.
            await _call(session)
            await _settle()

        assert outcomes == ["CancelledError"], "the reclaimed run reported a timeout as well"


class _Clock:
    """A monotonic clock a test advances by hand."""

    __slots__ = ("_now",)

    def __init__(self) -> None:
        self._now = 1000.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


async def _records_when_cancelled(closed: list[str]) -> Any:
    """Stand in for a held turn: parks forever, and records that it was closed."""
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        closed.append("closed")
        raise
