# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Sequence
from typing import Any

import acp
import pytest
from acp import schema
from acp.core import ClientSideConnection
from typing_extensions import Self

from ag2 import Agent, Context
from ag2.acp import ACPAgent, SessionConfig
from ag2.acp.guard import serve
from ag2.acp.testing import RecordingClient, connect, duplex
from ag2.config import LLMClient, ModelConfig
from ag2.events import BaseEvent, ModelResponse, ToolCallEvent
from ag2.history import MemoryStorage
from ag2.testing import TestConfig


class _HeldClient(LLMClient):
    """Parks inside the LLM call and records whether the turn was cancelled there."""

    def __init__(self, entered: asyncio.Event, cancelled: asyncio.Event) -> None:
        self.entered = entered
        self.cancelled = cancelled

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("the held turn was never meant to finish")  # pragma: no cover


class _HeldConfig(ModelConfig):
    """A config whose turns hang until something cancels them."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()

    def copy(self) -> Self:
        return self

    def create(self) -> _HeldClient:
        return _HeldClient(self.entered, self.cancelled)

    def create_files_client(self) -> None:
        raise NotImplementedError


class _StaticAgent:
    """An ACP agent object rather than a per-connection factory — ``serve`` takes both.

    Records whether teardown reached it: an object the caller built is the
    caller's to close, so it should not.
    """

    def __init__(self) -> None:
        self.initialized = False
        self.closed = False

    async def initialize(self, protocol_version: int, **kwargs: Any) -> schema.InitializeResponse:
        self.initialized = True
        return schema.InitializeResponse(protocol_version=min(protocol_version, acp.PROTOCOL_VERSION))

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
class TestSessionsDoNotOutliveTheirConnection:
    async def test_the_registry_is_empty_after_the_client_disconnects(self) -> None:
        server = ACPAgent(Agent("workie", config=TestConfig("ok")))

        async with connect(server) as (conn, _):
            await conn.new_session(cwd="/tmp")
            assert len(server.sessions) == 1

        assert len(server.sessions) == 0

    async def test_every_session_of_a_connection_goes_at_once(self) -> None:
        server = ACPAgent(Agent("workie", config=TestConfig("ok")))

        async with connect(server) as (conn, _):
            for _ in range(3):
                await conn.new_session(cwd="/tmp")
            assert len(server.sessions) == 3

        assert len(server.sessions) == 0

    async def test_a_reconnecting_client_cannot_name_the_old_connection_s_session(self) -> None:
        server = ACPAgent(Agent("workie", config=TestConfig("ok")))

        async with connect(server) as (first, _):
            stale = (await first.new_session(cwd="/tmp")).session_id

        async with connect(server) as (second, _):
            with pytest.raises(acp.RequestError):
                await second.prompt(session_id=stale, prompt=[acp.text_block("still there?")])

    async def test_one_connection_s_teardown_leaves_another_s_sessions_alone(self) -> None:
        """Scopes are per-connection, so teardown has to be too."""
        server = ACPAgent(Agent("workie", config=TestConfig("ok")))

        async with connect(server) as (first, _):
            kept = (await first.new_session(cwd="/tmp")).session_id
            survivor = server.sessions

            async with connect(server) as (second, _):
                await second.new_session(cwd="/tmp")
                doomed = server.sessions

            assert len(doomed) == 0
            assert len(survivor) == 1
            assert (await survivor.get(kept)).session_id == kept


@pytest.mark.asyncio
class TestTeardownStopsLiveWork:
    async def test_a_turn_still_running_at_disconnect_is_cancelled(self) -> None:
        config = _HeldConfig()
        server = ACPAgent(Agent("workie", config=config))

        async with connect(server) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            turn = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("slow")]))
            await asyncio.wait_for(config.entered.wait(), timeout=5)

        assert config.cancelled.is_set()
        # The request dies with the connection instead of being answered.
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(turn, timeout=5)

    async def test_a_busy_session_is_still_dropped_from_the_registry(self) -> None:
        """A turn in flight must not be able to hold its connection's scope open."""
        config = _HeldConfig()
        server = ACPAgent(Agent("workie", config=config))

        async with connect(server) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            turn = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("slow")]))
            await asyncio.wait_for(config.entered.wait(), timeout=5)

        assert len(server.sessions) == 0
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(turn, timeout=5)


@pytest.mark.asyncio
class TestDurableHistoryIsDropped:
    async def test_a_session_s_history_does_not_outlive_its_connection(self) -> None:
        """The default storage is in-memory; an injected one keeps orphans visible."""
        storage = MemoryStorage()
        server = ACPAgent(Agent("workie", config=TestConfig("answered")), sessions=SessionConfig(storage=storage))

        async with connect(server) as (conn, _):
            session_id = (await conn.new_session(cwd="/tmp")).session_id
            await conn.prompt(session_id=session_id, prompt=[acp.text_block("hello")])
            store = server.sessions
            session = await store.get(session_id)
            assert await store.stream(session).history.get_events() != []

        assert list(await store.stream(session).history.get_events()) == []

    async def test_only_the_closing_connection_s_history_is_dropped(self) -> None:
        storage = MemoryStorage()
        agent = Agent("workie", config=TestConfig("answered", "answered"))
        server = ACPAgent(agent, sessions=SessionConfig(storage=storage))

        async with connect(server) as (first, _):
            kept_id = (await first.new_session(cwd="/tmp")).session_id
            await first.prompt(session_id=kept_id, prompt=[acp.text_block("hello")])
            survivor = server.sessions
            kept = await survivor.get(kept_id)

            async with connect(server) as (second, _):
                doomed_id = (await second.new_session(cwd="/tmp")).session_id
                await second.prompt(session_id=doomed_id, prompt=[acp.text_block("hello")])
                doomed_store = server.sessions
                doomed = await doomed_store.get(doomed_id)

            assert list(await doomed_store.stream(doomed).history.get_events()) == []
            assert await survivor.stream(kept).history.get_events() != []


@pytest.mark.asyncio
class TestServeOverStreams:
    """``run_stdio``'s own path: the Client goes away, nothing cancels the server."""

    async def test_eof_from_the_client_releases_the_scope(self) -> None:
        server = ACPAgent(Agent("workie", config=TestConfig("ok")))
        (agent_r, agent_w), (client_r, client_w) = await duplex()
        served = asyncio.create_task(serve(server.bind, agent_r, agent_w))
        conn = ClientSideConnection(lambda _agent: RecordingClient(), client_w, client_r)

        try:
            await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
            await conn.new_session(cwd="/tmp")
            assert len(server.sessions) == 1

            client_w.close()  # EOF on the agent's input: the Client hung up
            await asyncio.wait_for(served, timeout=5)

            assert len(server.sessions) == 0
        finally:
            await conn.close()
            agent_w.close()

    async def test_an_agent_object_the_caller_built_is_not_closed(self) -> None:
        """``serve`` only releases the scope it created; a passed-in agent is not one."""
        agent = _StaticAgent()
        (agent_r, agent_w), (client_r, client_w) = await duplex()
        served = asyncio.create_task(serve(agent, agent_r, agent_w))
        conn = ClientSideConnection(lambda _agent: RecordingClient(), client_w, client_r)

        try:
            await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
            client_w.close()
            await asyncio.wait_for(served, timeout=5)

            assert agent.initialized is True
            assert agent.closed is False
        finally:
            await conn.close()
            agent_w.close()


@pytest.mark.asyncio
class TestBackgroundWorkBelongsToTheSession:
    """A stream carries an inbox and background tasks, not only history.

    A fresh stream per turn would strand work that finished after its own turn
    and hide it from teardown.
    """

    @staticmethod
    def _spawning_agent(ran: list[str]) -> Agent:
        agent = Agent(
            "workie",
            config=TestConfig(ToolCallEvent(name="spawn", arguments="{}"), "ok", "ok"),
        )

        @agent.tool
        async def spawn(ctx: Context) -> str:
            """Start work that deliberately outlives this turn."""

            async def later() -> None:
                await asyncio.sleep(0.05)
                ran.append("done")

            ctx.spawn_background(later())
            return "spawned"

        return agent

    async def test_the_session_keeps_one_stream_across_turns(self) -> None:
        server = ACPAgent(Agent("workie", config=TestConfig("ok", "ok")))

        async with connect(server) as (conn, _):
            created = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=created.session_id, prompt=[acp.text_block("one")])
            session = await server.sessions.get(created.session_id)
            first = server.sessions.stream(session)
            await conn.prompt(session_id=created.session_id, prompt=[acp.text_block("two")])

            assert server.sessions.stream(session) is first

    async def test_background_work_is_tracked_where_teardown_can_see_it(self) -> None:
        ran: list[str] = []
        server = ACPAgent(self._spawning_agent(ran))

        async with connect(server) as (conn, _):
            created = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=created.session_id, prompt=[acp.text_block("go")])
            session = await server.sessions.get(created.session_id)

            assert server.sessions.stream(session)._background_tasks

    async def test_closing_a_session_stops_its_background_work(self) -> None:
        """Otherwise a closed conversation keeps reaching the outside world."""
        ran: list[str] = []
        server = ACPAgent(self._spawning_agent(ran))

        async with connect(server) as (conn, _):
            created = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=created.session_id, prompt=[acp.text_block("go")])
            await server.sessions.close(created.session_id)
            await asyncio.sleep(0.2)

        assert ran == []

    async def test_a_turn_does_not_leave_its_forwarder_behind(self) -> None:
        """The stream outlives the turn now, so the subscriber must be released.

        Left attached, the second turn would deliver every update twice.
        """
        server = ACPAgent(Agent("workie", config=TestConfig("first", "second")))

        async with connect(server) as (conn, recorder):
            created = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=created.session_id, prompt=[acp.text_block("one")])
            await conn.prompt(session_id=created.session_id, prompt=[acp.text_block("two")])

        # Two turns, so two message updates. A leaked forwarder would make the
        # second turn deliver its update once per prompt the session had run.
        texts = [
            u.content.text for u in recorder.updates_for(created.session_id) if isinstance(u, schema.AgentMessageChunk)
        ]
        assert len(texts) == 2
