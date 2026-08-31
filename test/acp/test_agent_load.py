# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""``session/load``: reattaching to a conversation by id and replaying it.

The consumer this exists for spawns a fresh agent process per message and loads
the id it stored, so the cases that matter most are the ones that cross a
connection: history found by id alone, replayed in full before the response,
and nothing sent after it.
"""

import asyncio
from collections.abc import Sequence
from typing import Any
from uuid import UUID, uuid4

import acp
import pytest
from acp import schema
from acp.exceptions import RequestError
from typing_extensions import Self

from ag2 import Agent, Context
from ag2.acp import ACPAgent, SessionConfig, StaticTokenAuth
from ag2.acp.executor import CANCELLED_TOOL_RESULT, unanswered_tool_calls
from ag2.acp.testing import RecordingClient, connect
from ag2.config import LLMClient, ModelConfig
from ag2.events import (
    BaseEvent,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from ag2.events.tool_events import ToolResult
from ag2.events.types import ModelMessage
from ag2.history import MemoryStorage
from ag2.stream import MemoryStream
from ag2.testing import TestConfig

NOT_FOUND = RequestError.resource_not_found().code
INVALID_REQUEST = RequestError.invalid_request().code


def _agent(*turns: Any) -> Agent:
    return Agent("workie", config=TestConfig(*(turns or ("ok",))))


def _adder(*turns: Any) -> Agent:
    """An agent whose first turn calls ``add`` and then answers."""
    agent = Agent("workie", config=TestConfig(ToolCallEvent(name="add", arguments='{"a": 2, "b": 2}'), *turns))

    @agent.tool
    async def add(a: int, b: int) -> str:
        """Add two integers."""
        return str(a + b)

    return agent


def _retaining(storage: MemoryStorage | None = None, **overrides: Any) -> SessionConfig:
    return SessionConfig(storage=storage or MemoryStorage(), retain_history=True, **overrides)


def _kinds(updates: Sequence[Any]) -> list[type]:
    return [type(u) for u in updates]


class _HeldClient(LLMClient):
    def __init__(self, entered: asyncio.Event) -> None:
        self.entered = entered

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError("never finishes")  # pragma: no cover


class _HeldConfig(ModelConfig):
    """A config whose turn parks inside the LLM call until cancelled."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    def copy(self) -> Self:
        return self

    def create(self) -> _HeldClient:
        return _HeldClient(self.entered)

    def create_files_client(self) -> None:
        raise NotImplementedError


@pytest.mark.asyncio
class TestReplay:
    async def test_load_replays_the_conversation_the_client_already_had(self) -> None:
        server = ACPAgent(_adder("4"))

        async with connect(server) as (conn, recorder):
            sid = (await conn.new_session(cwd="/tmp")).session_id
            await conn.prompt(session_id=sid, prompt=[acp.text_block("what is 2 + 2")])
            live = len(recorder.updates_for(sid))

            await conn.load_session(session_id=sid, cwd="/tmp")
            await asyncio.sleep(0.05)

        replay = recorder.updates_for(sid)[live:]
        assert _kinds(replay) == [
            schema.UserMessageChunk,
            schema.ToolCallStart,
            schema.ToolCallProgress,
            schema.AgentMessageChunk,
        ]
        user, start, progress, answer = replay
        assert user.content.text == "what is 2 + 2"
        assert user.message_id
        assert start.title == "add"
        assert (progress.status, progress.content[0].content.text) == ("completed", "4")
        assert answer.content.text == "4"

    async def test_nothing_is_sent_after_load_returns(self) -> None:
        """The response is the line between history and whatever comes next."""
        server = ACPAgent(_adder("4"))

        async with connect(server) as (conn, recorder):
            sid = (await conn.new_session(cwd="/tmp")).session_id
            await conn.prompt(session_id=sid, prompt=[acp.text_block("go")])

            await conn.load_session(session_id=sid, cwd="/tmp")
            await asyncio.sleep(0.05)
            settled = len(recorder.updates_for(sid))
            await asyncio.sleep(0.05)

        assert len(recorder.updates_for(sid)) == settled
        assert settled == 3 + 4  # the live turn, then its replay

    async def test_a_prompt_after_load_continues_the_history(self) -> None:
        """The loaded session is the same stream, so the next turn reads the earlier one."""
        server = ACPAgent(_agent("one", "two"), sessions=_retaining())

        async with connect(server) as (first, _):
            sid = (await first.new_session(cwd="/tmp")).session_id
            await first.prompt(session_id=sid, prompt=[acp.text_block("first")])

        async with connect(server) as (second, _):
            await second.load_session(session_id=sid, cwd="/tmp")
            await second.prompt(session_id=sid, prompt=[acp.text_block("second")])
            session = await server.sessions.get(sid)
            events = list(await server.sessions.stream(session).history.get_events())

        turns = [e.parts for e in events if isinstance(e, ModelRequest)]
        assert turns == [[TextInput("first")], [TextInput("second")]]

    async def test_an_unprompted_live_session_replays_nothing(self) -> None:
        async with connect(ACPAgent(_agent())) as (conn, recorder):
            sid = (await conn.new_session(cwd="/tmp")).session_id

            await conn.load_session(session_id=sid, cwd="/tmp")
            await asyncio.sleep(0.05)

        assert recorder.updates_for(sid) == []


@pytest.mark.asyncio
class TestAcrossConnections:
    """The case the feature exists for: the process that minted the id is gone."""

    async def test_a_retained_session_loads_from_a_later_connection(self) -> None:
        storage = MemoryStorage()
        server = ACPAgent(_agent("one", "two"), sessions=_retaining(storage))

        async with connect(server) as (first, _):
            sid = (await first.new_session(cwd="/tmp")).session_id
            await first.prompt(session_id=sid, prompt=[acp.text_block("hello")])

        async with connect(server) as (second, recorder):
            await second.load_session(session_id=sid, cwd="/tmp")
            await asyncio.sleep(0.05)
            replay = list(recorder.updates_for(sid))
            response = await second.prompt(session_id=sid, prompt=[acp.text_block("still there?")])

        assert _kinds(replay) == [schema.UserMessageChunk, schema.AgentMessageChunk]
        assert [u.content.text for u in replay] == ["hello", "one"]
        assert response.stop_reason == "end_turn"

    async def test_without_retention_a_disconnected_session_is_not_found(self) -> None:
        server = ACPAgent(_agent("one"), sessions=SessionConfig(storage=MemoryStorage()))

        async with connect(server) as (first, _):
            sid = (await first.new_session(cwd="/tmp")).session_id
            await first.prompt(session_id=sid, prompt=[acp.text_block("hello")])

        async with connect(server) as (second, _):
            with pytest.raises(RequestError) as caught:
                await second.load_session(session_id=sid, cwd="/tmp")

        assert caught.value.code == NOT_FOUND

    async def test_a_session_never_prompted_cannot_be_loaded_from_elsewhere(self) -> None:
        """No history means nothing distinguishes it from an id never issued — and nothing to resume."""
        server = ACPAgent(_agent(), sessions=_retaining())

        async with connect(server) as (first, _):
            sid = (await first.new_session(cwd="/tmp")).session_id

        async with connect(server) as (second, _):
            with pytest.raises(RequestError) as caught:
                await second.load_session(session_id=sid, cwd="/tmp")

        assert caught.value.code == NOT_FOUND

    @pytest.mark.parametrize("retain", [True, False])
    async def test_an_evicted_session_loads_only_when_retained(self, retain: bool) -> None:
        """Tier 1 without Tier 2 would resume right up until the first eviction, then stop."""
        server = ACPAgent(
            _agent("one", "two"),
            sessions=SessionConfig(max_sessions=1, storage=MemoryStorage(), retain_history=retain),
        )

        async with connect(server) as (conn, recorder):
            first = (await conn.new_session(cwd="/tmp")).session_id
            await conn.prompt(session_id=first, prompt=[acp.text_block("hello")])
            await conn.new_session(cwd="/tmp")  # evicts ``first``

            if retain:
                await conn.load_session(session_id=first, cwd="/tmp")
                await asyncio.sleep(0.05)
                assert _kinds(recorder.updates_for(first))[-2:] == [schema.UserMessageChunk, schema.AgentMessageChunk]
            else:
                with pytest.raises(RequestError) as caught:
                    await conn.load_session(session_id=first, cwd="/tmp")
                assert caught.value.code == NOT_FOUND

    async def test_the_id_is_accepted_in_any_uuid_spelling(self) -> None:
        storage = MemoryStorage()
        server = ACPAgent(_agent("one"), sessions=_retaining(storage))

        async with connect(server) as (first, _):
            sid = (await first.new_session(cwd="/tmp")).session_id
            await first.prompt(session_id=sid, prompt=[acp.text_block("hello")])

        async with connect(server) as (second, _):
            await second.load_session(session_id=str(UUID(sid)), cwd="/tmp")
            session = await server.sessions.get(sid)

        assert session.session_id == sid


@pytest.mark.asyncio
class TestSessionState:
    @staticmethod
    def _spender(config: ModelConfig) -> Agent:
        agent = Agent("workie", config=config, variables={"budget": [1]})

        @agent.tool
        async def spend(ctx: Context) -> str:
            """Record a spend in the session's variables."""
            ctx.variables["budget"].append(2)
            return "spent"

        return agent

    async def test_a_loaded_session_reseeds_variables_from_the_agent_defaults(self) -> None:
        """Variables are not history; a session loaded elsewhere starts from the defaults."""
        config = TestConfig(ToolCallEvent(name="spend", arguments="{}"), "ok")
        server = ACPAgent(self._spender(config), sessions=_retaining())

        async with connect(server) as (first, _):
            sid = (await first.new_session(cwd="/tmp")).session_id
            await first.prompt(session_id=sid, prompt=[acp.text_block("spend")])
            assert (await server.sessions.get(sid)).variables["budget"] == [1, 2]

        async with connect(server) as (second, _):
            await second.load_session(session_id=sid, cwd="/tmp")
            reseeded = (await server.sessions.get(sid)).variables

        assert reseeded["budget"] == [1]

    async def test_a_live_session_keeps_its_variables_across_a_load(self) -> None:
        config = TestConfig(ToolCallEvent(name="spend", arguments="{}"), "ok")
        server = ACPAgent(self._spender(config))

        async with connect(server) as (conn, _):
            sid = (await conn.new_session(cwd="/tmp")).session_id
            await conn.prompt(session_id=sid, prompt=[acp.text_block("spend")])
            await conn.load_session(session_id=sid, cwd="/tmp")
            kept = (await server.sessions.get(sid)).variables

        assert kept["budget"] == [1, 2]

    async def test_a_load_refreshes_the_request_context_but_not_absent_meta(self) -> None:
        server = ACPAgent(_agent())

        async with connect(server) as (conn, _):
            raw = conn._conn if hasattr(conn, "_conn") else conn._connection
            created = await raw.send_request(
                "session/new",
                {"cwd": "/a", "mcpServers": [], "_meta": {"ag2.space": {"room": "!r"}}},
            )
            sid = created["sessionId"]

            await conn.load_session(session_id=sid, cwd="/b", additional_directories=["/extra"])
            session = await server.sessions.get(sid)

        assert (session.cwd, session.additional_directories) == ("/b", ["/extra"])
        assert session.meta == {"ag2.space": {"room": "!r"}}

    async def test_meta_on_a_load_replaces_the_session_meta(self) -> None:
        server = ACPAgent(_agent())

        async with connect(server) as (conn, _):
            sid = (await conn.new_session(cwd="/tmp")).session_id
            raw = conn._conn if hasattr(conn, "_conn") else conn._connection
            await raw.send_request(
                "session/load",
                {"sessionId": sid, "cwd": "/tmp", "mcpServers": [], "_meta": {"ag2.space": {"room": "!r"}}},
            )
            session = await server.sessions.get(sid)

        assert session.meta == {"ag2.space": {"room": "!r"}}


@pytest.mark.asyncio
class TestRepair:
    @staticmethod
    def _agent_with_a_hanging_tool(entered: asyncio.Event) -> Agent:
        agent = Agent("workie", config=TestConfig(ToolCallEvent(name="slow", arguments="{}"), "done"))

        @agent.tool
        async def slow() -> str:
            """Never finishes on its own."""
            entered.set()
            await asyncio.Event().wait()
            return "unreachable"  # pragma: no cover

        return agent

    async def test_a_turn_cut_short_by_disconnect_is_healed_on_load(self) -> None:
        """The process died mid-tool; the next one must not send a dangling call to the provider."""
        entered = asyncio.Event()
        storage = MemoryStorage()
        server = ACPAgent(self._agent_with_a_hanging_tool(entered), sessions=_retaining(storage))

        async with connect(server) as (first, _):
            sid = (await first.new_session(cwd="/tmp")).session_id
            turn = asyncio.create_task(first.prompt(session_id=sid, prompt=[acp.text_block("go")]))
            await asyncio.wait_for(entered.wait(), timeout=5)
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(turn, timeout=5)

        async with connect(server) as (second, recorder):
            await second.load_session(session_id=sid, cwd="/tmp")
            await asyncio.sleep(0.05)
            session = await server.sessions.get(sid)
            events = list(await server.sessions.stream(session).history.get_events())

        replay = recorder.updates_for(sid)
        assert _kinds(replay) == [schema.UserMessageChunk, schema.ToolCallStart, schema.ToolCallProgress]
        assert replay[2].status == "completed"
        assert replay[2].content[0].content.text == CANCELLED_TOOL_RESULT
        assert len([e for e in events if isinstance(e, ToolResultsEvent)]) == 1
        assert unanswered_tool_calls(events) == set()

    async def test_a_cleanly_finished_history_is_not_rewritten_by_a_load(self) -> None:
        """A loose result without its wrapper is what a ``final`` tool leaves; it is answered, so leave it."""
        server = ACPAgent(_agent())
        call = ToolCallEvent(id="c1", name="finish", arguments="{}")
        history = [
            ModelRequest([TextInput("go")]),
            ModelResponse(tool_calls=ToolCallsEvent([call])),
            ToolCallsEvent([call]),
            call,
            ToolResultEvent(parent_id="c1", name="finish", result=ToolResult("done")),
            ModelResponse(ModelMessage("done"), response_force=True),
        ]

        async with connect(server) as (conn, _):
            sid = (await conn.new_session(cwd="/tmp")).session_id
            session = await server.sessions.get(sid)
            await server.sessions.stream(session).history.replace(history)

            await conn.load_session(session_id=sid, cwd="/tmp")

            after = list(await server.sessions.stream(session).history.get_events())

        assert len(after) == len(history)


class TestHealGate:
    """``unanswered_tool_calls`` asks the narrower question ``heal`` cannot."""

    def test_a_call_with_no_result_anywhere_is_unanswered(self) -> None:
        call = ToolCallEvent(id="c1", name="slow", arguments="{}")

        assert unanswered_tool_calls([ModelResponse(tool_calls=ToolCallsEvent([call])), call]) == {"c1"}

    def test_a_loose_result_answers_its_call(self) -> None:
        call = ToolCallEvent(id="c1", name="finish", arguments="{}")
        result = ToolResultEvent(parent_id="c1", name="finish", result=ToolResult("ok"))

        assert unanswered_tool_calls([ToolCallsEvent([call]), result]) == set()

    def test_a_wrapped_result_answers_its_call(self) -> None:
        call = ToolCallEvent(id="c1", name="finish", arguments="{}")
        result = ToolResultEvent(parent_id="c1", name="finish", result=ToolResult("ok"))

        assert unanswered_tool_calls([ToolCallsEvent([call]), ToolResultsEvent([result])]) == set()


@pytest.mark.asyncio
class TestRefusals:
    async def test_a_session_mid_turn_is_not_loaded_underneath_it(self) -> None:
        config = _HeldConfig()
        server = ACPAgent(Agent("workie", config=config))

        async with connect(server) as (conn, _):
            sid = (await conn.new_session(cwd="/tmp")).session_id
            turn = asyncio.create_task(conn.prompt(session_id=sid, prompt=[acp.text_block("slow")]))
            await asyncio.wait_for(config.entered.wait(), timeout=5)

            with pytest.raises(RequestError) as caught:
                await conn.load_session(session_id=sid, cwd="/tmp")

            await conn.cancel(session_id=sid)
            assert (await turn).stop_reason == "cancelled"

        assert caught.value.code == INVALID_REQUEST

    async def test_load_is_gated_until_authenticated(self) -> None:
        server = ACPAgent(_agent(), auth=StaticTokenAuth("s3cret"))

        async with connect(server) as (conn, _):
            with pytest.raises(RequestError) as caught:
                await conn.load_session(session_id=uuid4().hex, cwd="/tmp")

        assert caught.value.code == RequestError.auth_required().code

    async def test_load_is_refused_before_initialize(self) -> None:
        async with connect(ACPAgent(_agent()), initialize=False) as (conn, _):
            with pytest.raises(RequestError):
                await conn.load_session(session_id=uuid4().hex, cwd="/tmp")


@pytest.mark.asyncio
class TestOnSession:
    """The host's seam: told each time a session comes to exist on a connection."""

    async def test_created_on_new_and_loaded_on_load(self) -> None:
        seen: list[tuple[str, str]] = []
        server = ACPAgent(_agent("ok"), on_session=lambda s, origin: seen.append((s.session_id, origin)))

        async with connect(server) as (conn, _):
            sid = (await conn.new_session(cwd="/tmp")).session_id
            await conn.load_session(session_id=sid, cwd="/tmp")

        assert seen == [(sid, "created"), (sid, "loaded")]

    async def test_a_session_rehydrated_from_storage_is_reported_as_loaded(self) -> None:
        seen: list[tuple[str, str, str]] = []

        async def observe(session: Any, origin: str) -> None:
            seen.append((session.session_id, session.stream_id.hex, origin))

        server = ACPAgent(_agent("ok"), sessions=_retaining(), on_session=observe)

        async with connect(server) as (first, _):
            sid = (await first.new_session(cwd="/tmp")).session_id
            await first.prompt(session_id=sid, prompt=[acp.text_block("hello")])

        async with connect(server) as (second, _):
            await second.load_session(session_id=sid, cwd="/tmp")

        assert seen == [(sid, sid, "created"), (sid, sid, "loaded")]

    async def test_the_hook_is_told_before_the_replay(self) -> None:
        order: list[str] = []

        class _OrderingRecorder(RecordingClient):
            async def session_update(self, *, session_id: str, update: Any, **kwargs: Any) -> None:
                order.append("update")
                await super().session_update(session_id=session_id, update=update, **kwargs)

        server = ACPAgent(_agent("ok"), on_session=lambda s, origin: order.append(origin))

        async with connect(server, client=_OrderingRecorder()) as (conn, _):
            sid = (await conn.new_session(cwd="/tmp")).session_id
            await conn.prompt(session_id=sid, prompt=[acp.text_block("hello")])
            order.clear()
            await conn.load_session(session_id=sid, cwd="/tmp")
            await asyncio.sleep(0.05)

        assert order[0] == "loaded"
        assert "update" in order[1:]

    async def test_a_failing_hook_fails_the_request(self) -> None:
        def refuse(session: Any, origin: str) -> None:
            raise RuntimeError("cannot correlate")

        server = ACPAgent(_agent(), on_session=refuse)

        async with connect(server) as (conn, _):
            with pytest.raises(RequestError):
                await conn.new_session(cwd="/tmp")

            assert len(server.sessions) == 0

    async def test_a_refused_load_leaves_nothing_behind_to_prompt(self) -> None:
        """The hook is an authorization seam only if a refusal really refuses.

        A session adopted for the hook's benefit must be unregistered again when
        the hook says no — otherwise ``session/prompt`` finds it and runs, and
        the refusal was cosmetic.
        """

        def refuse_loads(session: Any, origin: str) -> None:
            if origin == "loaded":
                raise PermissionError("not this caller's conversation")

        storage = MemoryStorage()
        server = ACPAgent(_agent("one", "two"), sessions=_retaining(storage), on_session=refuse_loads)

        async with connect(server) as (first, _):
            sid = (await first.new_session(cwd="/tmp")).session_id
            await first.prompt(session_id=sid, prompt=[acp.text_block("private")])

        async with connect(server) as (second, _):
            with pytest.raises(RequestError):
                await second.load_session(session_id=sid, cwd="/tmp")

            assert len(server.sessions) == 0
            with pytest.raises(RequestError) as caught:
                await second.prompt(session_id=sid, prompt=[acp.text_block("mine now?")])

        assert caught.value.code == NOT_FOUND
        # And the conversation itself was left exactly as it was found.
        assert list(await MemoryStream(storage=storage, id=UUID(sid)).history.get_events()) != []
