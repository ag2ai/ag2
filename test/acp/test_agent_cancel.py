# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Sequence
from typing import Any

import acp
import pytest
from acp import schema
from typing_extensions import Self

from ag2 import Agent, Context
from ag2.acp import ACPAgent, SessionConfig, StaticTokenAuth
from ag2.acp.executor import CANCELLED_TOOL_RESULT
from ag2.acp.testing import connect
from ag2.config import LLMClient, ModelConfig
from ag2.config.openai.mappers import convert_messages
from ag2.events import BaseEvent, ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from ag2.testing import TestConfig


class _GatedClient(LLMClient):
    """Blocks inside the LLM call until released, so a turn can be held mid-flight."""

    def __init__(self, client: LLMClient, entered: asyncio.Event, release: asyncio.Event) -> None:
        self.client = client
        self.entered = entered
        self.release = release

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        self.entered.set()
        await self.release.wait()
        return await self.client(messages, context=context, **kwargs)


class _GatedConfig(ModelConfig):
    """A config whose turns park until :attr:`release` is set."""

    def __init__(self, *turns: object) -> None:
        self.config = TestConfig(*(turns or ("ok",)))
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    def copy(self) -> Self:
        return self

    def create(self) -> _GatedClient:
        return _GatedClient(self.config.create(), self.entered, self.release)

    def create_files_client(self) -> None:
        raise NotImplementedError


class _NullSerializer:
    """Stand-in for the provider serializer; tool schemas are not exercised here."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def _texts(updates: list[Any]) -> list[str]:
    return [u.content.text for u in updates if isinstance(u, schema.AgentMessageChunk)]


@pytest.mark.asyncio
class TestCancel:
    async def test_a_cancelled_turn_reports_the_cancelled_stop_reason(self) -> None:
        config = _GatedConfig("never delivered")

        async with connect(ACPAgent(Agent("workie", config=config))) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            turn = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("slow")]))
            await config.entered.wait()

            await conn.cancel(session_id=session.session_id)
            response = await turn

        assert response.stop_reason == "cancelled"

    async def test_the_agent_never_completes_a_cancelled_turn(self) -> None:
        config = _GatedConfig("never delivered")

        async with connect(ACPAgent(Agent("workie", config=config))) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")
            turn = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("slow")]))
            await config.entered.wait()

            await conn.cancel(session_id=session.session_id)
            await turn

        assert "never delivered" not in _texts(recorder.updates_for(session.session_id))

    async def test_updates_already_sent_are_not_retracted(self) -> None:
        """Cancelling stops the turn; it does not undo what the Client already saw."""
        config = _GatedConfig("second turn text")

        async with connect(ACPAgent(Agent("workie", config=config))) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")

            # A completed turn first, so there is delivered history to preserve.
            config.release.set()
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("first")])
            delivered = list(recorder.updates_for(session.session_id))

            config.release.clear()
            config.entered.clear()
            turn = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("second")]))
            await config.entered.wait()
            await conn.cancel(session_id=session.session_id)
            await turn

        assert recorder.updates_for(session.session_id)[: len(delivered)] == delivered

    async def test_cancelling_one_session_leaves_another_running(self) -> None:
        config = _GatedConfig("reply", "reply")

        async with connect(ACPAgent(Agent("workie", config=config))) as (conn, _):
            cancelled = await conn.new_session(cwd="/tmp")
            other = await conn.new_session(cwd="/tmp")

            held = asyncio.create_task(conn.prompt(session_id=cancelled.session_id, prompt=[acp.text_block("slow")]))
            await config.entered.wait()
            await conn.cancel(session_id=cancelled.session_id)
            assert (await held).stop_reason == "cancelled"

            config.release.set()
            survivor = await conn.prompt(session_id=other.session_id, prompt=[acp.text_block("fine")])

        assert survivor.stop_reason == "end_turn"

    async def test_cancelling_an_unknown_session_is_ignored(self) -> None:
        """``session/cancel`` is a notification — there is no channel for an error."""
        async with connect(ACPAgent(Agent("workie", config=TestConfig("ok")))) as (conn, _):
            await conn.cancel(session_id="never-issued")

            session = await conn.new_session(cwd="/tmp")
            response = await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("hi")])

        assert response.stop_reason == "end_turn"

    async def test_a_session_is_usable_again_after_a_cancel(self) -> None:
        config = _GatedConfig("first", "after cancel")

        async with connect(ACPAgent(Agent("workie", config=config))) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            held = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("slow")]))
            await config.entered.wait()
            await conn.cancel(session_id=session.session_id)
            await held

            config.release.set()
            response = await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("again")])

        assert response.stop_reason == "end_turn"


@pytest.mark.asyncio
class TestBusySession:
    async def test_a_second_prompt_waits_rather_than_interleaving(self) -> None:
        config = _GatedConfig("first", "second")

        async with connect(ACPAgent(Agent("workie", config=config))) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            first = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("one")]))
            await config.entered.wait()
            second = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("two")]))
            await asyncio.sleep(0.01)

            assert not second.done()  # queued behind the running turn

            config.release.set()
            assert (await first).stop_reason == "end_turn"
            assert (await second).stop_reason == "end_turn"

    async def test_a_cancel_drops_prompts_queued_behind_the_running_turn(self) -> None:
        config = _GatedConfig("first", "second")

        async with connect(ACPAgent(Agent("workie", config=config))) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            first = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("one")]))
            await config.entered.wait()
            second = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("two")]))
            await asyncio.sleep(0.01)

            await conn.cancel(session_id=session.session_id)

            assert (await first).stop_reason == "cancelled"
            assert (await second).stop_reason == "cancelled"

    async def test_the_queue_is_bounded(self) -> None:
        config = _GatedConfig(*["reply"] * 6)
        server = ACPAgent(Agent("workie", config=config), sessions=SessionConfig(max_queued=2))

        async with connect(server) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            running = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("one")]))
            await config.entered.wait()

            queued = [
                asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("more")]))
                for _ in range(2)
            ]
            await asyncio.sleep(0.01)

            with pytest.raises(acp.RequestError):
                await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("overflow")])

            config.release.set()
            await running
            await asyncio.gather(*queued)


@pytest.mark.asyncio
class TestCancelLeavesAUsableSession:
    """Cancelling is meant to be recoverable, so it must not corrupt history.

    A cancel can land between a tool call and its result. Providers reject an
    assistant tool-call with nothing answering it, so a session left in that
    shape would fail on its *next* prompt — turning a normal stop into a broken
    conversation.
    """

    @staticmethod
    def _agent_with_a_hanging_tool(entered: asyncio.Event) -> Agent:
        agent = Agent(
            "workie",
            config=TestConfig(ToolCallEvent(name="slow", arguments="{}"), "done"),
        )

        @agent.tool
        async def slow() -> str:
            """Never finishes on its own."""
            entered.set()
            await asyncio.Event().wait()
            return "unreachable"  # pragma: no cover - the turn is cancelled first

        return agent

    async def _cancel_mid_tool(self, server: ACPAgent, conn: Any) -> str:
        session = await conn.new_session(cwd="/tmp")
        turn = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")]))
        await self.entered.wait()
        await conn.cancel(session_id=session.session_id)
        assert (await turn).stop_reason == "cancelled"
        return session.session_id

    async def test_every_tool_call_is_answered_after_a_cancel(self) -> None:
        self.entered = asyncio.Event()
        server = ACPAgent(self._agent_with_a_hanging_tool(self.entered))

        async with connect(server) as (conn, _):
            session_id = await self._cancel_mid_tool(server, conn)
            session = await server.sessions.get(session_id)
            events = list(await server.sessions.stream(session).history.get_events())

        calls = {c.id for e in events if isinstance(e, ToolCallsEvent) for c in e.calls}
        answered = {r.parent_id for e in events if isinstance(e, ToolResultsEvent) for r in e.results}
        assert calls and calls <= answered

    async def test_the_synthetic_result_says_what_happened(self) -> None:
        self.entered = asyncio.Event()
        server = ACPAgent(self._agent_with_a_hanging_tool(self.entered))

        async with connect(server) as (conn, _):
            session_id = await self._cancel_mid_tool(server, conn)
            session = await server.sessions.get(session_id)
            events = list(await server.sessions.stream(session).history.get_events())

        [results] = [e for e in events if isinstance(e, ToolResultsEvent)]
        assert CANCELLED_TOOL_RESULT in str(results.results[0].result)

    async def test_the_repaired_history_converts_for_a_strict_provider(self) -> None:
        """The failure this prevents: OpenAI rejects an unanswered tool call."""
        self.entered = asyncio.Event()
        server = ACPAgent(self._agent_with_a_hanging_tool(self.entered))

        async with connect(server) as (conn, _):
            session_id = await self._cancel_mid_tool(server, conn)
            session = await server.sessions.get(session_id)
            events = list(await server.sessions.stream(session).history.get_events())

        messages = convert_messages([], events, _NullSerializer())
        orphans = [
            message
            for index, message in enumerate(messages)
            if message.get("role") == "assistant"
            and message.get("tool_calls")
            and (index + 1 >= len(messages) or messages[index + 1].get("role") != "tool")
        ]
        assert orphans == []

    async def test_nothing_is_appended_when_no_tool_was_pending(self) -> None:
        """A cancel between turns must not invent results for calls already answered."""
        config = _GatedConfig("never delivered")
        server = ACPAgent(Agent("workie", config=config))

        async with connect(server) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            turn = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("hi")]))
            await config.entered.wait()
            await conn.cancel(session_id=session.session_id)
            await turn

            live = await server.sessions.get(session.session_id)
            events = list(await server.sessions.stream(live).history.get_events())

        assert [e for e in events if isinstance(e, ToolResultsEvent)] == []


@pytest.mark.asyncio
class TestCancelIsScoped:
    """Cancelling mutates someone's conversation, so it needs the same authority."""

    @staticmethod
    async def _hold(server: ACPAgent, session_id: str) -> "asyncio.Task[None]":
        session = await server.sessions.get(session_id)
        started = asyncio.Event()

        async def hold() -> None:
            async with session.turn():
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(hold())
        session.turn_task = task
        await started.wait()
        return task

    async def test_an_uninitialized_reconnect_cannot_cancel(self) -> None:
        server = ACPAgent(Agent("workie", config=TestConfig("ok")), auth=StaticTokenAuth("s3cret"))

        async with connect(server) as (conn, _):
            await conn.authenticate(method_id="token", token="s3cret")
            session = await conn.new_session(cwd="/tmp")
            task = await self._hold(server, session.session_id)

            async with connect(server, initialize=False) as (intruder, _):
                await intruder.cancel(session_id=session.session_id)
                await asyncio.sleep(0.05)

            assert not task.done()
            task.cancel()

    async def test_an_unauthenticated_connection_cannot_cancel(self) -> None:
        """Initialized is not enough — cancelling needs the credential too.

        The session is created through the store rather than the protocol,
        because a connection that has not authenticated cannot create one; the
        point here is only that a *cancel* from it is refused.
        """
        server = ACPAgent(Agent("workie", config=TestConfig("ok")), auth=StaticTokenAuth("s3cret"))

        async with connect(server) as (conn, _):
            session = await server.sessions.create()
            task = await self._hold(server, session.session_id)

            await conn.cancel(session_id=session.session_id)
            await asyncio.sleep(0.05)

            assert not task.done()
            task.cancel()

    async def test_an_authenticated_connection_can_still_cancel(self) -> None:
        server = ACPAgent(Agent("workie", config=TestConfig("ok")), auth=StaticTokenAuth("s3cret"))

        async with connect(server) as (conn, _):
            await conn.authenticate(method_id="token", token="s3cret")
            session = await conn.new_session(cwd="/tmp")
            task = await self._hold(server, session.session_id)

            await conn.cancel(session_id=session.session_id)
            await asyncio.sleep(0.05)

            assert task.cancelled()


@pytest.mark.asyncio
class TestRepairIsSerialised:
    """The repair rewrites the whole transcript, so no turn may run alongside it."""

    async def test_a_prompt_racing_the_repair_still_sees_valid_history(self) -> None:
        agent = Agent(
            "workie",
            config=TestConfig(ToolCallEvent(name="slow", arguments="{}"), "done", "second"),
        )
        entered = asyncio.Event()

        @agent.tool
        async def slow() -> str:
            """Never finishes on its own."""
            entered.set()
            await asyncio.Event().wait()
            return "unreachable"  # pragma: no cover - cancelled first

        server = ACPAgent(agent)

        async with connect(server) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            first = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")]))
            await entered.wait()

            # Fire the follow-up prompt without waiting for the repair to finish.
            cancelling = asyncio.create_task(conn.cancel(session_id=session.session_id))
            second = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("again")]))
            await first
            await cancelling
            await second

            live = await server.sessions.get(session.session_id)
            events = list(await server.sessions.stream(live).history.get_events())

        calls = {c.id for e in events if isinstance(e, ToolCallsEvent) for c in e.calls}
        answered = {r.parent_id for e in events if isinstance(e, ToolResultsEvent) for r in e.results}
        assert calls and calls <= answered


@pytest.mark.asyncio
class TestParallelToolCancellation:
    """A half-finished parallel batch must still serialize to valid history.

    Providers pair every tool call with a tool result. If one tool of a batch
    completed and another was cancelled, the repair has to emit results for the
    *whole* batch — the completed one included — or the transcript ends up with
    more calls than results and the next prompt is rejected.
    """

    @staticmethod
    def _agent(entered: asyncio.Event) -> Agent:
        agent = Agent(
            "workie",
            config=TestConfig(
                [ToolCallEvent(name="fast", arguments="{}"), ToolCallEvent(name="slow", arguments="{}")],
                "done",
            ),
        )

        @agent.tool
        async def fast() -> str:
            """Returns straight away."""
            return "quick result"

        @agent.tool
        async def slow() -> str:
            """Never finishes on its own."""
            entered.set()
            await asyncio.Event().wait()
            return "unreachable"  # pragma: no cover - cancelled first

        return agent

    async def test_every_call_in_the_batch_is_answered(self) -> None:
        entered = asyncio.Event()
        server = ACPAgent(self._agent(entered))

        async with connect(server) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            turn = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")]))
            await entered.wait()
            await asyncio.sleep(0.05)
            await conn.cancel(session_id=session.session_id)
            await turn
            live = await server.sessions.get(session.session_id)
            events = list(await server.sessions.stream(live).history.get_events())

        calls = {c.id for e in events if isinstance(e, ToolCallsEvent) for c in e.calls}
        answered = {r.parent_id for e in events if isinstance(e, ToolResultsEvent) for r in e.results}
        assert len(calls) == 2
        assert calls == answered

    async def test_the_completed_result_is_kept_not_overwritten(self) -> None:
        entered = asyncio.Event()
        server = ACPAgent(self._agent(entered))

        async with connect(server) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            turn = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")]))
            await entered.wait()
            await asyncio.sleep(0.05)
            await conn.cancel(session_id=session.session_id)
            await turn
            live = await server.sessions.get(session.session_id)
            events = list(await server.sessions.stream(live).history.get_events())

        rendered = str([r.result for e in events if isinstance(e, ToolResultsEvent) for r in e.results])
        assert "quick result" in rendered
        assert CANCELLED_TOOL_RESULT in rendered

    async def test_calls_and_results_balance_for_a_strict_provider(self) -> None:
        entered = asyncio.Event()
        server = ACPAgent(self._agent(entered))

        async with connect(server) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            turn = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")]))
            await entered.wait()
            await asyncio.sleep(0.05)
            await conn.cancel(session_id=session.session_id)
            await turn
            live = await server.sessions.get(session.session_id)
            events = list(await server.sessions.stream(live).history.get_events())

        messages = convert_messages([], events, _NullSerializer())
        calls = sum(len(m.get("tool_calls") or []) for m in messages)
        results = sum(1 for m in messages if m.get("role") == "tool")
        assert calls == results == 2


class _ConcurrencyProbeConfig(ModelConfig):
    """Parks every turn inside the LLM call so concurrency can be observed."""

    def __init__(self, replies: int = 50) -> None:
        self.config = TestConfig(*["ok"] * replies)
        self.running = 0
        self.peak = 0
        self.release = asyncio.Event()

    def copy(self) -> Self:
        return self

    def create(self) -> "_ConcurrencyProbeClient":
        return _ConcurrencyProbeClient(self.config.create(), self)

    def create_files_client(self) -> None:
        raise NotImplementedError


class _ConcurrencyProbeClient(LLMClient):
    def __init__(self, client: LLMClient, probe: _ConcurrencyProbeConfig) -> None:
        self.client = client
        self.probe = probe

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        self.probe.running += 1
        self.probe.peak = max(self.probe.peak, self.probe.running)
        try:
            await self.probe.release.wait()
            return await self.client(messages, context=context, **kwargs)
        finally:
            self.probe.running -= 1


@pytest.mark.asyncio
class TestConnectionWideLimits:
    """Per-session bounds cannot see each other; these bound the connection.

    Without them a Client can open the full session cap and start a paid turn in
    every one at once.
    """

    async def test_concurrent_turns_are_capped_across_sessions(self) -> None:
        probe = _ConcurrencyProbeConfig()
        server = ACPAgent(
            Agent("workie", config=probe),
            sessions=SessionConfig(max_concurrent_turns=3, max_active_prompts=6),
        )

        async with connect(server) as (conn, _):
            ids = [(await conn.new_session(cwd="/tmp")).session_id for _ in range(6)]
            turns = [asyncio.create_task(conn.prompt(session_id=sid, prompt=[acp.text_block("go")])) for sid in ids]
            await asyncio.sleep(0.1)

            assert probe.peak == 3

            probe.release.set()
            responses = await asyncio.gather(*turns)

        assert [r.stop_reason for r in responses] == ["end_turn"] * 6

    async def test_prompts_past_the_cap_wait_rather_than_fail(self) -> None:
        """A burst across separate conversations is traffic, not abuse."""
        probe = _ConcurrencyProbeConfig()
        server = ACPAgent(
            Agent("workie", config=probe),
            sessions=SessionConfig(max_concurrent_turns=2, max_active_prompts=6),
        )

        async with connect(server) as (conn, _):
            ids = [(await conn.new_session(cwd="/tmp")).session_id for _ in range(5)]
            turns = [asyncio.create_task(conn.prompt(session_id=sid, prompt=[acp.text_block("go")])) for sid in ids]
            await asyncio.sleep(0.1)

            assert not any(t.done() for t in turns)

            probe.release.set()
            responses = await asyncio.gather(*turns)

        assert all(r.stop_reason == "end_turn" for r in responses)

    async def test_admission_is_refused_past_max_active_prompts(self) -> None:
        probe = _ConcurrencyProbeConfig()
        server = ACPAgent(
            Agent("workie", config=probe),
            sessions=SessionConfig(max_concurrent_turns=2, max_active_prompts=4),
        )

        async with connect(server) as (conn, _):
            ids = [(await conn.new_session(cwd="/tmp")).session_id for _ in range(5)]
            turns = [asyncio.create_task(conn.prompt(session_id=sid, prompt=[acp.text_block("go")])) for sid in ids[:4]]
            await asyncio.sleep(0.1)

            with pytest.raises(acp.RequestError):
                await conn.prompt(session_id=ids[4], prompt=[acp.text_block("go")])

            probe.release.set()
            await asyncio.gather(*turns)

    async def test_a_slot_frees_up_once_a_turn_finishes(self) -> None:
        probe = _ConcurrencyProbeConfig()
        server = ACPAgent(
            Agent("workie", config=probe),
            sessions=SessionConfig(max_concurrent_turns=1, max_active_prompts=2),
        )

        async with connect(server) as (conn, _):
            first = await conn.new_session(cwd="/tmp")
            second = await conn.new_session(cwd="/tmp")
            probe.release.set()

            assert (await conn.prompt(session_id=first.session_id, prompt=[acp.text_block("a")])).stop_reason
            assert (await conn.prompt(session_id=second.session_id, prompt=[acp.text_block("b")])).stop_reason

        assert probe.running == 0
