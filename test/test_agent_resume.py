# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``Agent.resume`` — resuming a turn from a recorded trajectory.

``resume`` blesses, as public API, the ``_execute(trigger)`` path: it seeds the
stream history with a recorded trajectory and re-enters the agent loop with an
arbitrary ``BaseEvent`` trigger (typically a ``ToolResultsEvent``), so the model
reacts to a tool result mid-loop without the tool being re-executed.
"""

import asyncio

import pytest

from ag2 import Agent, Context
from ag2.events import (
    BaseEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from ag2.stream import MemoryStream
from ag2.testing import TestConfig, TrackingConfig


def _seed_through_tool_call(question: str, tool_name: str, args: str, tc_id: str):
    """Build the trajectory [user question, assistant tool-call] to seed."""
    tc = ToolCallEvent(name=tool_name, arguments=args, id=tc_id)
    history = [
        ModelRequest([TextInput(question)]),
        ModelResponse(message=ModelMessage(""), tool_calls=ToolCallsEvent([tc])),
    ]
    return history, tc


class TestAgentResume:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("cancel", [False, True])
    @pytest.mark.parametrize("reconstruct_stream", [False, True])
    async def test_queued_resume_preserves_running_turn_history(self, cancel: bool, reconstruct_stream: bool) -> None:
        stream = MemoryStream()
        target = MemoryStream(storage=stream.history.storage, id=stream.id) if reconstruct_stream else stream
        previous = ModelRequest([TextInput("original conversation")])
        replacement = ModelRequest([TextInput("replacement conversation")])
        trigger = ModelRequest([TextInput("resume question")])
        await stream.history.replace([previous])
        agent = Agent("retriever", config=TestConfig("active answer"))
        started = asyncio.Event()

        async def resume() -> str:
            started.set()
            reply = await agent.resume(replacement, trigger, stream=target, config=TestConfig("resumed answer"))
            return reply.body

        pending = None
        try:
            async with agent.run("active question", stream=stream) as run:
                pending = asyncio.create_task(resume())
                await started.wait()
                if cancel:
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending

                assert list(await stream.history.get_events()) == [previous]
                assert (await run.result()).body == "active answer"

            if not cancel:
                assert await pending == "resumed answer"
                assert await stream.history.where(ModelRequest).all() == [replacement, trigger]
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("has_prefix", [False, True])
    async def test_resume_seeds_history_before_dynamic_prompt(self, has_prefix: bool) -> None:
        stream = MemoryStream()
        await stream.history.replace([ModelRequest([TextInput("stale conversation")])])
        prefix = [ModelRequest([TextInput("restored conversation")])] if has_prefix else []
        observed: list[list[BaseEvent]] = []

        async def prompt(context: Context) -> str:
            observed.append(list(await context.stream.history.get_events()))
            return "Answer the question."

        agent = Agent("retriever", config=TestConfig("answer"), prompt=prompt)
        trigger = ModelRequest([TextInput("resume question")])

        reply = await agent.resume(*prefix, trigger, stream=stream)

        assert reply.body == "answer"
        assert observed == [prefix]
        assert await stream.history.where(ModelRequest).all() == [*prefix, trigger]

    @pytest.mark.asyncio
    async def test_resume_from_tool_results_continues_turn(self) -> None:
        """A ToolResultsEvent trigger drives one LLM call → final answer."""
        history, tc = _seed_through_tool_call("Where is X?", "get_node_details", '{"id": "X"}', "call-1")
        agent = Agent("retriever", config=TestConfig("Final answer grounded on the result."))

        trigger = ToolResultsEvent([ToolResultEvent.from_call(tc, "X is at file.cc:42")])
        reply = await agent.resume(*history, trigger)

        assert reply.body == "Final answer grounded on the result."

        events = list(await reply.context.stream.history.get_events())
        # Seeded prefix is preserved, the trigger is appended, and the final
        # model response lands after it — contiguous, no re-execution.
        assert any(isinstance(e, ModelRequest) for e in events)
        assert any(isinstance(e, ToolResultsEvent) for e in events)
        assert isinstance(events[-1], ModelResponse)

    @pytest.mark.asyncio
    async def test_resume_feeds_seeded_history_and_trigger_to_model(self) -> None:
        """The LLM call sees the seeded history, with the trigger as the last message."""
        history, tc = _seed_through_tool_call("Q?", "search_docs", "{}", "call-1")
        tracking = TrackingConfig(TestConfig("done"))
        agent = Agent("retriever", config=tracking)

        trigger = ToolResultsEvent([ToolResultEvent.from_call(tc, "evidence")])
        await agent.resume(*history, trigger)

        # Exactly one LLM call; its last message is the tool-results trigger.
        tracking.mock.assert_called_once()
        last_message = tracking.mock.call_args.args[0]
        assert isinstance(last_message, ToolResultsEvent)

    @pytest.mark.asyncio
    async def test_resume_continues_into_more_tool_calls(self) -> None:
        """After reacting to the result, the model may issue further tool calls."""
        history, tc = _seed_through_tool_call("Q?", "search_docs", "{}", "call-1")

        def echo(value: str) -> str:
            return f"echoed:{value}"

        # React to the seeded result by calling another tool, then finish.
        next_call = ToolCallEvent(name="echo", arguments='{"value": "again"}', id="call-2")
        agent = Agent("retriever", tools=[echo], config=TestConfig(next_call, "All done."))

        trigger = ToolResultsEvent([ToolResultEvent.from_call(tc, "first result")])
        reply = await agent.resume(*history, trigger)

        assert reply.body == "All done."
        events = list(await reply.context.stream.history.get_events())
        # The live continuation actually executed the echo tool.
        assert any(
            isinstance(e, ToolResultsEvent) and any("echoed:again" in str(r.result) for r in e.results) for e in events
        )

    @pytest.mark.asyncio
    async def test_resume_replaces_existing_stream_history(self) -> None:
        """A supplied stream's pre-existing history is replaced by ``history``."""
        stream = MemoryStream()
        await stream.history.replace([ModelRequest([TextInput("stale prior turn")])])

        history, tc = _seed_through_tool_call("fresh question", "search_docs", "{}", "call-1")
        agent = Agent("retriever", config=TestConfig("answer"))

        trigger = ToolResultsEvent([ToolResultEvent.from_call(tc, "r")])
        reply = await agent.resume(*history, trigger, stream=stream)

        events = list(await reply.context.stream.history.get_events())
        texts = [p.content for e in events if isinstance(e, ModelRequest) for p in e.parts if isinstance(p, TextInput)]
        assert "fresh question" in texts
        assert "stale prior turn" not in texts
