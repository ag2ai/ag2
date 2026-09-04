# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""What else may happen to a conversation, and to a process, while a run is paused.

A modern-era pause lets go of the conversation's turn lock so the retry that
resumes it is not blocked by the run it is resuming. Three things follow, one per
class here: another call naming that conversation, a round that dies between the
registry and the client, and the process going down with runs still held.

``test_pause_and_resume.py`` covers the happy round trip;
``test_paused_run_lifetime.py`` the two bounds on how long a run may be held.
"""

import asyncio
from typing import Any, cast

import pytest
from mcp.client.session import ClientRequestContext
from mcp.types import (
    CallToolResult,
    ClientCapabilities,
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitRequestParams,
    ElicitResult,
    ElicitationCapability,
    InputRequiredResult,
    TextContent,
)

from ag2 import Agent, Context
from ag2.events import ToolCallEvent
from ag2.mcp import MCPServer
from ag2.mcp.elicitation import ANSWER_FIELD
from ag2.mcp.executor import AgentExecutor
from ag2.mcp.pause import PauseState, PausedRuns, SuspendedTurn
from ag2.mcp.sessions import CONVERSATION_META_KEY, Conversation, SessionStore
from ag2.mcp.testing import connect_modern
from ag2.stream import MemoryStream
from ag2.testing import TestConfig

# Only ever reached on a regression, and then it is the difference between a
# failing test and a suite that never returns.
NEVER_ON_A_PASSING_RUN = 5.0


def asking_agent(
    *,
    gate: "asyncio.Event | None" = None,
    entered: "asyncio.Event | None" = None,
    outcomes: list[str] | None = None,
) -> Agent:
    """An agent whose one tool asks the caller a question.

    ``gate`` is held *before* the question, giving a deterministic point at which
    a round is mid-flight rather than parked; ``entered`` signals reaching it.
    """

    async def ask_human(ctx: Context) -> str:
        try:
            if entered is not None:
                entered.set()
            if gate is not None:
                await gate.wait()
            answer = await ctx.input("What colour?")
        except BaseException as exc:
            if outcomes is not None:
                outcomes.append(type(exc).__name__)
            raise
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


async def _call_in(session: Any, handle: str, message: str, **kwargs: Any) -> Any:
    """A call naming a conversation.

    The boundary binds a token to its call's arguments, so every round of one
    turn goes through here with the same two.
    """
    return await session.call_tool(
        "ask", {"message": message, "conversation": handle}, allow_input_required=True, **kwargs
    )


def _accepting(answer: str) -> ElicitResult:
    return ElicitResult(action="accept", content={ANSWER_FIELD: answer})


def _handle(result: Any) -> str:
    assert result.meta is not None
    return str(result.meta[CONVERSATION_META_KEY])


def _first_text(result: Any) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


async def _settle() -> None:
    """Give the loop enough turns for a held run to reach its next await."""
    for _ in range(10):
        await asyncio.sleep(0)


@pytest.mark.asyncio
class TestAConversationHoldingAPausedRun:
    """A second call naming it must be refused, and refused *promptly*.

    The paused run is still inside ``Agent.ask`` holding the lock the agent keys
    on the conversation's stream id, which every call on that conversation shares
    however fresh its ``MemoryStream`` object. Letting the second through would
    not interleave two turns; it would park one with no timeout.
    """

    async def test_a_second_call_on_it_is_refused_rather_than_hung(self) -> None:
        server = MCPServer(asking_agent())

        async with connect_modern(server, raise_exceptions=False, elicitation_callback=declares_elicitation) as session:
            opened = await _call(session)
            assert isinstance(opened, InputRequiredResult)
            (key,) = (opened.input_requests or {}).keys()
            finished = await _call(
                session, input_responses={key: _accepting("blue")}, request_state=opened.request_state
            )
            handle = _handle(finished)

            paused = await _call_in(session, handle, "again")
            assert isinstance(paused, InputRequiredResult)

            second = await asyncio.wait_for(
                session.call_tool("ask", {"message": "meanwhile", "conversation": handle}),
                timeout=NEVER_ON_A_PASSING_RUN,
            )

        assert second.is_error is True
        assert "waiting on an answer" in _first_text(second)

    async def test_answering_the_paused_call_frees_the_conversation(self) -> None:
        """The refusal lasts exactly as long as the pause does."""
        server = MCPServer(asking_agent())

        async with connect_modern(server, raise_exceptions=False, elicitation_callback=declares_elicitation) as session:
            opened = await _call(session)
            assert isinstance(opened, InputRequiredResult)
            (first_key,) = (opened.input_requests or {}).keys()
            finished = await _call(
                session, input_responses={first_key: _accepting("blue")}, request_state=opened.request_state
            )
            handle = _handle(finished)

            paused = await _call_in(session, handle, "again")
            assert isinstance(paused, InputRequiredResult)
            (key,) = (paused.input_requests or {}).keys()
            await _call_in(
                session,
                handle,
                "again",
                input_responses={key: _accepting("green")},
                request_state=paused.request_state,
            )

            after = await asyncio.wait_for(
                _call_in(session, handle, "meanwhile"),
                timeout=NEVER_ON_A_PASSING_RUN,
            )

        assert isinstance(after, InputRequiredResult), "the conversation was still refusing calls"


@pytest.mark.asyncio
class TestARoundThatDiesLeavesNothingBehind:
    """A cancelled round must not strand the run it was driving.

    A held run is a live task holding its conversation's stream lock, and one in
    no registry can be reached by nothing — no retry, no sweep, no eviction — so
    it would hold that lock for the life of the process.
    """

    async def test_a_cancelled_first_round_reclaims_the_run_it_started(self) -> None:
        gate = asyncio.Event()
        entered = asyncio.Event()
        outcomes: list[str] = []
        runs = PausedRuns(ttl=1000.0)
        executor = AgentExecutor(
            asking_agent(gate=gate, entered=entered, outcomes=outcomes),
            stream_progress=False,
            paused_runs=runs,
        )
        convo = Conversation(stream=MemoryStream())

        round_one = asyncio.ensure_future(
            executor._start_suspendable(convo, "go", None, cast(Any, _Peer())),
        )
        await asyncio.wait_for(entered.wait(), timeout=NEVER_ON_A_PASSING_RUN)
        round_one.cancel()
        with pytest.raises(asyncio.CancelledError):
            await round_one
        await _settle()

        assert outcomes == ["CancelledError"], "the run outlived the round that started it"
        assert runs.holds_conversation(convo.handle) is False

    async def test_a_cancelled_resume_puts_the_run_back(self) -> None:
        """The client's state still names this run, so a later retry must find it."""
        gate = asyncio.Event()
        runs = PausedRuns(ttl=1000.0)
        turn = SuspendedTurn(conversation=None, stream=MemoryStream(), created=runs.now())
        turn.start(_asks_twice(turn, gate))
        await _settle()
        assert turn.outstanding is not None
        (key, _request) = turn.outstanding
        runs.register(turn)
        executor = AgentExecutor(asking_agent(), stream_progress=False, paused_runs=runs)
        state = PauseState.mint(run_id=turn.id, request_key=key).encode()

        # The answer un-parks the run, which then blocks on the gate rather than
        # on a question — so this round is mid-flight, not parked, when it dies.
        resume = asyncio.ensure_future(executor._resume(state, {key: _accepting("blue")}, cast(Any, None)))
        await _settle()
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
            turn.start(_records_when_cancelled(closed))
            runs.register(turn)
        await _settle()

        runs.reclaim_all()
        await _settle()

        assert closed == ["closed", "closed", "closed"]

    async def test_the_http_app_reclaims_them_from_its_lifespan(self) -> None:
        """The wiring, not the registry: a server torn down must take its runs with it."""
        server = MCPServer(asking_agent())
        closed: list[str] = []
        turn = SuspendedTurn(conversation=None, stream=MemoryStream(), created=server._paused_runs.now())
        turn.start(_records_when_cancelled(closed))
        server._paused_runs.register(turn)
        await _settle()

        await _drive_asgi_lifespan(server)
        await _settle()

        assert closed == ["closed"], "shutting the app down left a held run parked"


@pytest.mark.asyncio
class TestResumingKeepsTheConversationAlive:
    async def test_a_paused_turn_is_not_idle_evicted_between_its_own_rounds(self) -> None:
        """A resume goes nowhere near the registry — it continues a turn already
        inside a conversation. Left uncounted, one whose turn asks several
        questions ages out mid-question, and the eviction reclaims that run."""
        clock = _Clock()
        store = SessionStore(ttl=10.0, clock=clock)
        async with store.fresh() as convo:
            handle = convo.handle
        assert handle is not None

        runs = PausedRuns(ttl=1000.0)
        turn = SuspendedTurn(conversation=handle, stream=MemoryStream(), created=runs.now())
        turn.start(_asks_twice(turn, asyncio.Event(), keep_asking=True))
        await _settle()
        assert turn.outstanding is not None
        (key, _request) = turn.outstanding
        runs.register(turn)
        executor = AgentExecutor(asking_agent(), stream_progress=False, session_store=store, paused_runs=runs)

        clock.advance(8.0)
        await executor._resume(
            PauseState.mint(run_id=turn.id, request_key=key).encode(),
            {key: _accepting("blue")},
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
    """Just enough request context for the paths under test.

    With ``stream_progress=False`` and no context provider or client model, the
    only thing read off one is whether the caller can answer a question.
    """

    def __init__(self) -> None:
        self.session = _PeerSession()
        self.meta: dict[str, Any] | None = None
        self.request_id = 1


class _PeerSession:
    client_capabilities = ClientCapabilities(elicitation=ElicitationCapability())
    can_send_request = False


class _Clock:
    """A monotonic clock a test advances by hand."""

    __slots__ = ("_now",)

    def __init__(self) -> None:
        self._now = 1000.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


async def _asks_twice(turn: SuspendedTurn, gate: asyncio.Event, *, keep_asking: bool = False) -> CallToolResult:
    """Stand in for a held turn that has more to do after its first answer."""
    await turn.ask(_question("First?"))
    if keep_asking:
        await turn.ask(_question("Second?"))
    else:
        await gate.wait()
    return CallToolResult(content=[TextContent(type="text", text="done")])


async def _records_when_cancelled(closed: list[str]) -> Any:
    """Stand in for a held turn: parks forever, and records that it was closed."""
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        closed.append("closed")
        raise


def _question(message: str) -> ElicitRequest:
    return ElicitRequest(
        params=ElicitRequestFormParams(
            message=message,
            requested_schema={"type": "object", "properties": {ANSWER_FIELD: {"type": "string"}}},
        )
    )


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
