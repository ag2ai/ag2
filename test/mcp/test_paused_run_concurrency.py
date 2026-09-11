# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""What else may happen to a conversation, and to a process, while a run is paused.

A modern-era pause lets go of the conversation's turn lock, so the retry that
resumes it is not blocked by the run it is resuming. One class each: another
call naming that conversation, a round that dies between the registry and the
client, and the process going down with runs still held.
"""

import asyncio
from typing import Any, cast

import pytest
from mcp.types import ClientCapabilities, ElicitationCapability, InputRequiredResult

from ag2.mcp import MCPServer
from ag2.mcp.executor import AgentExecutor
from ag2.mcp.pause import PauseState, PausedRuns, SuspendedTurn
from ag2.mcp.sessions import Conversation, SessionStore
from ag2.mcp.testing import connect_modern
from ag2.stream import MemoryStream

from ._helpers import (
    Asked,
    Clock,
    Gate,
    accepting,
    answer,
    ask,
    asking_agent,
    asks_then_works,
    asks_twice,
    declares_elicitation,
    first_text,
    handle_of,
    parks_until_cancelled,
    settle,
)

# Only ever reached on a regression, and then it is the difference between a
# failing test and a suite that never returns.
NEVER_ON_A_PASSING_RUN = 5.0


@pytest.mark.asyncio
class TestAConversationHoldingAPausedRun:
    """A second call naming it must be refused, and refused *promptly*.

    The paused run is still inside ``Agent.ask`` holding the lock keyed on the
    conversation's stream id, so letting the second through would park it with no
    timeout rather than interleave two turns.
    """

    async def test_a_second_call_on_it_is_refused_rather_than_hung(self) -> None:
        server = MCPServer(asking_agent())

        async with connect_modern(server, raise_exceptions=False, elicitation_callback=declares_elicitation) as session:
            opened = await ask(session)
            assert isinstance(opened, InputRequiredResult)
            handle = handle_of(await answer(session, opened, accepting("blue")))

            paused = await ask(session, "again", conversation=handle)
            assert isinstance(paused, InputRequiredResult)

            second = await asyncio.wait_for(
                ask(session, "meanwhile", conversation=handle),
                timeout=NEVER_ON_A_PASSING_RUN,
            )

        assert second.is_error is True
        assert "waiting on an answer" in first_text(second)

    async def test_answering_the_paused_call_frees_the_conversation(self) -> None:
        """The refusal lasts exactly as long as the pause does."""
        server = MCPServer(asking_agent())

        async with connect_modern(server, raise_exceptions=False, elicitation_callback=declares_elicitation) as session:
            opened = await ask(session)
            assert isinstance(opened, InputRequiredResult)
            handle = handle_of(await answer(session, opened, accepting("blue")))

            paused = await ask(session, "again", conversation=handle)
            assert isinstance(paused, InputRequiredResult)
            await answer(session, paused, accepting("green"), message="again", conversation=handle)

            after = await asyncio.wait_for(
                ask(session, "meanwhile", conversation=handle),
                timeout=NEVER_ON_A_PASSING_RUN,
            )

        assert isinstance(after, InputRequiredResult), "the conversation was still refusing calls"


@pytest.mark.asyncio
class TestARoundThatDiesLeavesNothingBehind:
    """A cancelled round must not strand the run it was driving.

    A held run in no registry can be reached by nothing — no retry, no sweep, no
    eviction — so it would hold its conversation's lock for the life of the process.
    """

    async def test_a_cancelled_first_round_reclaims_the_run_it_started(self) -> None:
        gate = Gate()
        asked = Asked()
        runs = PausedRuns(ttl=1000.0)
        executor = AgentExecutor(
            asking_agent(asked, gate=gate),
            stream_progress=False,
            paused_runs=runs,
        )
        convo = Conversation(stream=MemoryStream())

        round_one = asyncio.ensure_future(
            executor._start_suspendable(convo, "go", None, cast(Any, _Peer())),
        )
        await asyncio.wait_for(gate.entered.wait(), timeout=NEVER_ON_A_PASSING_RUN)
        round_one.cancel()
        with pytest.raises(asyncio.CancelledError):
            await round_one
        await settle()

        assert asked.outcomes == ["CancelledError"], "the run outlived the round that started it"
        assert runs.holds_conversation(convo.handle) is False

    async def test_a_cancelled_resume_puts_the_run_back(self) -> None:
        """The client's state still names this run, so a later retry must find it."""
        gate = asyncio.Event()
        runs = PausedRuns(ttl=1000.0)
        turn = SuspendedTurn(conversation=None, stream=MemoryStream(), created=runs.now())
        turn.start(asks_then_works(turn, gate))
        await settle()
        assert turn.outstanding is not None
        (key, _request) = turn.outstanding
        runs.register(turn)
        executor = AgentExecutor(asking_agent(), stream_progress=False, paused_runs=runs)
        state = PauseState.mint(run_id=turn.id, request_key=key).encode()

        # The answer un-parks the run, which then blocks on the gate rather than
        # on a question — so this round is mid-flight, not parked, when it dies.
        resume = asyncio.ensure_future(executor._resume(state, {key: accepting("blue")}, cast(Any, None)))
        await settle()
        resume.cancel()
        with pytest.raises(asyncio.CancelledError):
            await resume

        assert runs.take(turn.id) is turn, "the state a client holds named a run no registry had"
        turn.reclaim()

    async def test_shutdown_reclaims_every_held_run(self) -> None:
        """Nothing else does: retention is swept on the next call, and there is none."""
        runs = PausedRuns(ttl=1000.0)
        closed: list[str] = []
        for _ in range(3):
            turn = SuspendedTurn(conversation=None, stream=MemoryStream(), created=runs.now())
            turn.start(parks_until_cancelled(closed))
            runs.register(turn)
        await settle()

        runs.reclaim_all()
        await settle()

        assert closed == ["closed", "closed", "closed"]

    async def test_the_http_app_reclaims_them_from_its_lifespan(self) -> None:
        """The wiring, not the registry: a server torn down must take its runs with it."""
        server = MCPServer(asking_agent())
        closed: list[str] = []
        turn = SuspendedTurn(conversation=None, stream=MemoryStream(), created=server._paused_runs.now())
        turn.start(parks_until_cancelled(closed))
        server._paused_runs.register(turn)
        await settle()

        await _drive_asgi_lifespan(server)
        await settle()

        assert closed == ["closed"], "shutting the app down left a held run parked"


@pytest.mark.asyncio
async def test_a_paused_turn_is_not_idle_evicted_between_its_own_rounds() -> None:
    """Resuming keeps the conversation alive; uncounted, a turn asking several questions ages out mid-question."""
    clock = Clock()
    store = SessionStore(ttl=10.0, clock=clock)
    async with store.fresh() as convo:
        handle = convo.handle
    assert handle is not None

    runs = PausedRuns(ttl=1000.0)
    turn = SuspendedTurn(conversation=handle, stream=MemoryStream(), created=runs.now())
    turn.start(asks_twice(turn))
    await settle()
    assert turn.outstanding is not None
    (key, _request) = turn.outstanding
    runs.register(turn)
    executor = AgentExecutor(asking_agent(), stream_progress=False, session_store=store, paused_runs=runs)

    clock.advance(8.0)
    await executor._resume(
        PauseState.mint(run_id=turn.id, request_key=key).encode(),
        {key: accepting("blue")},
        cast(Any, None),
    )
    clock.advance(8.0)

    # Sixteen seconds since the conversation was created, eight since it was
    # last used. A fresh conversation is what sweeps the registry.
    async with store.fresh():
        pass
    async with store.by_handle(handle) as still_there:
        assert still_there.handle == handle

    turn.reclaim()


class _Peer:
    """Just enough request context for the paths under test."""

    def __init__(self) -> None:
        self.session = _PeerSession()
        self.meta: dict[str, Any] | None = None
        self.request_id = 1


class _PeerSession:
    client_capabilities = ClientCapabilities(elicitation=ElicitationCapability())
    can_send_request = False


async def _drive_asgi_lifespan(server: MCPServer) -> None:
    """Start and then shut down ``server``'s ASGI app, as a host would."""
    events: list[dict[str, Any]] = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return events.pop(0) if events else {"type": "lifespan.shutdown"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await server({"type": "lifespan", "asgi": {"version": "3.0"}}, receive, send)
    assert [m["type"] for m in sent] == ["lifespan.startup.complete", "lifespan.shutdown.complete"], sent
