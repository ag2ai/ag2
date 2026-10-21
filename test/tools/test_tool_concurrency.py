# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``max_concurrent_tool_calls`` bounds one tool batch without changing how a
batch settles: results still pair with their calls, tool failures stay ordinary
results, client tools still dispatch, and a cancelled batch starts nothing new."""

import asyncio
from contextlib import ExitStack

import pytest
from fast_depends.use import SerializerCls

from ag2 import Agent, Context, MemoryStream, tool
from ag2.events import (
    BaseEvent,
    ClientToolCallEvent,
    ModelMessage,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from ag2.exceptions import HumanInputError
from ag2.testing import TestConfig
from ag2.tools.executor import ToolExecutor
from ag2.tools.final.client_tool import ClientTool

CONCURRENCY_WAIT = 0.05


class _Peak:
    """Tracks how many tool bodies are running at once."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.started: list[str] = []

    def enter(self, name: str) -> None:
        self.started.append(name)
        self.active += 1
        self.peak = max(self.peak, self.active)

    def leave(self) -> None:
        self.active -= 1


def _calls(*names: str) -> list[ToolCallEvent]:
    return [ToolCallEvent(id=f"tc_{name}", name=name, arguments="{}") for name in names]


def _numbered_calls(count: int, name: str = "work") -> list[ToolCallEvent]:
    return [ToolCallEvent(id=f"tc_{i}", name=name, arguments=f'{{"tag": "{i}"}}') for i in range(count)]


def _tool_events(events: list[BaseEvent]) -> list[ToolResultEvent]:
    return [event for event in events if isinstance(event, ToolResultEvent)]


@pytest.mark.asyncio
class TestExecutorConcurrencyGate:
    """The gate itself, driven through the executor without an Agent."""

    async def test_rejects_non_positive_limit(self) -> None:
        for bad in (0, -1):
            with pytest.raises(ValueError, match="max_concurrent_tool_calls must be >= 1"):
                ToolExecutor(SerializerCls, max_concurrent_tool_calls=bad)

    async def test_limit_caps_peak(self) -> None:
        peak = _Peak()

        @tool
        async def work(tag: str) -> str:
            """Hold a slot so the peak is observable."""
            peak.enter(tag)
            await asyncio.sleep(CONCURRENCY_WAIT)
            peak.leave()
            return tag

        stream = MemoryStream()
        context = Context(stream=stream)
        calls = _numbered_calls(10)

        with ExitStack() as stack:
            executor = ToolExecutor(SerializerCls, max_concurrent_tool_calls=3)
            executor.register(stack, context, tools=[work], known_tools={"work"})
            await context.send(ToolCallsEvent(calls=calls))

        assert peak.peak == 3
        assert len(_tool_events(list(await stream.history.get_events()))) == 10

    async def test_results_keep_their_calls(self) -> None:
        # Each call sleeps for a different length so completion order is the
        # reverse of dispatch order; pairing must survive it.
        @tool
        async def work(tag: str) -> str:
            """Sleep longer for lower tags."""
            index = int(tag)
            await asyncio.sleep(CONCURRENCY_WAIT * (10 - index))
            return f"done-{tag}"

        stream = MemoryStream()
        context = Context(stream=stream)
        calls = _numbered_calls(10)

        with ExitStack() as stack:
            executor = ToolExecutor(SerializerCls, max_concurrent_tool_calls=4)
            executor.register(stack, context, tools=[work], known_tools={"work"})
            await context.send(ToolCallsEvent(calls=calls))

        results = _tool_events(list(await stream.history.get_events()))
        pairs = {event.parent_id: event.result.parts[0].content for event in results}
        assert pairs == {f"tc_{i}": f"done-{i}" for i in range(10)}


@pytest.mark.asyncio
class TestAgentConcurrencyGate:
    """The public parameter, driven through ``Agent.ask``."""

    async def test_limit_caps_peak(self) -> None:
        peak = _Peak()

        @tool
        async def work(tag: str) -> str:
            """Do some work."""
            peak.enter(tag)
            await asyncio.sleep(CONCURRENCY_WAIT)
            peak.leave()
            return tag

        agent = Agent(
            "bounded",
            config=TestConfig(ModelResponse(tool_calls=ToolCallsEvent(calls=_numbered_calls(10))), "done"),
            tools=[work],
            max_concurrent_tool_calls=3,
        )

        reply = await agent.ask("go")

        assert reply.body == "done"
        assert peak.peak == 3
        assert len(peak.started) == 10

    async def test_unset_keeps_unbounded_behaviour(self) -> None:
        peak = _Peak()

        @tool
        async def work(tag: str) -> str:
            """Do some work."""
            peak.enter(tag)
            await asyncio.sleep(CONCURRENCY_WAIT)
            peak.leave()
            return tag

        agent = Agent(
            "unbounded",
            config=TestConfig(ModelResponse(tool_calls=ToolCallsEvent(calls=_numbered_calls(10))), "done"),
            tools=[work],
        )

        assert (await agent.ask("go")).body == "done"
        assert peak.peak == 10

    async def test_cancelled_batch_does_not_start_queued_calls(self) -> None:
        peak = _Peak()

        @tool
        async def needs_human() -> str:
            """Ask a human who never answers."""
            peak.enter("needs_human")
            await asyncio.sleep(CONCURRENCY_WAIT)
            peak.leave()
            raise HumanInputError("nobody to ask")

        @tool
        async def worker(tag: str) -> str:
            """Queued behind the slot."""
            peak.enter(tag)
            await asyncio.sleep(CONCURRENCY_WAIT)
            peak.leave()
            return tag

        calls = [ToolCallEvent(id="tc_h", name="needs_human", arguments="{}")] + _numbered_calls(5, "worker")
        agent = Agent(
            "cancelling",
            config=TestConfig(ModelResponse(tool_calls=ToolCallsEvent(calls=calls)), "done"),
            tools=[needs_human, worker],
            max_concurrent_tool_calls=1,
        )

        with pytest.raises(HumanInputError):
            await agent.ask("go")

        await asyncio.sleep(CONCURRENCY_WAIT)
        queued_started = [name for name in peak.started if name != "needs_human"]
        assert queued_started == []

    async def test_tool_errors_are_reported_per_call(self) -> None:
        @tool
        async def work(tag: str) -> str:
            """Fail on one tag, succeed on the rest."""
            if tag == "1":
                raise ValueError("tool blew up")
            return tag

        agent = Agent(
            "failing",
            config=TestConfig(
                ModelResponse(tool_calls=ToolCallsEvent(calls=_numbered_calls(4))),
                ModelResponse(ModelMessage("done")),
                # A real provider receives a failed call as an ordinary result;
                # asserting that the batch survives it needs the same behaviour.
                raise_tool_errors=False,
            ),
            tools=[work],
            max_concurrent_tool_calls=2,
        )

        reply = await agent.ask("go")
        events = list(await reply.context.stream.history.get_events())
        batch = next(event for event in events if isinstance(event, ToolResultsEvent))

        errors = [event for event in batch.results if isinstance(event, ToolErrorEvent)]
        oks = [
            event
            for event in batch.results
            if isinstance(event, ToolResultEvent) and not isinstance(event, ToolErrorEvent)
        ]
        assert [event.parent_id for event in errors] == ["tc_1"]
        assert {event.parent_id for event in oks} == {"tc_0", "tc_2", "tc_3"}

    async def test_client_tools_still_dispatch(self) -> None:
        @tool
        async def work(tag: str) -> str:
            """Do some work."""
            return tag

        client_tool = ClientTool({"function": {"name": "remote_work", "description": "Client side.", "parameters": {}}})
        calls = _numbered_calls(2) + [ToolCallEvent(id="tc_remote", name="remote_work", arguments="{}")]
        agent = Agent(
            "mixed",
            config=TestConfig(
                ModelResponse(tool_calls=ToolCallsEvent(calls=calls)),
                ModelResponse(ModelMessage("done")),
            ),
            tools=[work, client_tool],
            max_concurrent_tool_calls=1,
        )

        call_events: list[BaseEvent] = []

        async def record(event: BaseEvent) -> None:
            call_events.append(event)

        stream = MemoryStream()
        with ExitStack() as stack:
            stack.enter_context(stream.where(ClientToolCallEvent).sub_scope(record))
            await agent.ask("go", stream=stream)

        # The client tool short-circuits the batch into a second model call; the
        # dispatched ClientToolCallEvent proves that path still ran rather than
        # deadlocking behind the gate.
        assert [event.id for event in call_events if isinstance(event, ClientToolCallEvent)] == ["tc_remote"]

    async def test_limit_applies_per_batch_not_per_turn(self) -> None:
        peak = _Peak()

        @tool
        async def work(tag: str) -> str:
            """Do some work."""
            peak.enter(tag)
            await asyncio.sleep(CONCURRENCY_WAIT)
            peak.leave()
            return tag

        batches: list[list[ToolCallEvent]] = [_numbered_calls(6), _numbered_calls(6)]
        for index, batch in enumerate(batches):
            for call in batch:
                call.id = f"tc_b{index}_{call.id}"

        agent = Agent(
            "two-batches",
            config=TestConfig(
                ModelResponse(tool_calls=ToolCallsEvent(calls=batches[0])),
                ModelResponse(tool_calls=ToolCallsEvent(calls=batches[1])),
                ModelResponse(ModelMessage("done")),
            ),
            tools=[work],
            max_concurrent_tool_calls=2,
        )

        reply = await agent.ask("go")

        assert reply.body == "done"
        assert peak.peak <= 2
        assert len(peak.started) == 12

    async def test_agent_with_limit_is_reusable_across_event_loops(self) -> None:
        """A capped Agent survives being driven from more than one event loop.

        ``asyncio.Semaphore`` binds to the first loop that awaits it while
        contended. Provisioning one per batch — not caching it on the executor —
        is what keeps the Agent reusable.
        """

        @tool
        async def work(tag: str) -> str:
            """Do some work."""
            await asyncio.sleep(0)
            return tag

        agent = Agent(
            "reused",
            config=TestConfig(ModelResponse(tool_calls=ToolCallsEvent(calls=_numbered_calls(4))), "done"),
            tools=[work],
            max_concurrent_tool_calls=1,
        )

        async def turn() -> str | None:
            return (await agent.ask("go", stream=MemoryStream())).body

        bodies = [await turn() for _ in range(3)]
        assert bodies == ["done", "done", "done"]


@pytest.mark.asyncio
class TestLiveAgentConcurrencyGate:
    async def test_live_agent_accepts_the_same_limit(self) -> None:
        """``LiveAgent`` shares ``PluginTarget``'s executor, so it takes the cap too."""
        from ag2.live.realtime import LiveAgent

        @tool
        async def work(tag: str) -> str:
            """Do some work."""
            return tag

        live = LiveAgent(
            "live",
            config=TestConfig(ModelResponse(ModelMessage("hi"))),
            tools=[work],
            max_concurrent_tool_calls=2,
        )

        assert live.name == "live"
        assert len(live.tools) == 1
        # A non-positive cap is refused on the live target exactly as on Agent.
        with pytest.raises(ValueError, match="max_concurrent_tool_calls must be >= 1"):
            LiveAgent(
                "live",
                config=TestConfig(ModelResponse(ModelMessage("hi"))),
                tools=[work],
                max_concurrent_tool_calls=0,
            )


def test_limit_accepts_keyword_only() -> None:
    """The cap is keyword-only so the serializer stays the only positional arg."""
    with pytest.raises(TypeError):
        ToolExecutor(SerializerCls, 3)  # type: ignore[misc]
