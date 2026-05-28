# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from autogen.beta import Agent, Context, MemoryStream, tool
from autogen.beta.events import (
    DrainedModelRequest,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallEvent,
)
from autogen.beta.testing import TestConfig
from autogen.beta.tools.subagents import background_agent_tool


@pytest.mark.asyncio
class TestBackgroundDelivery:
    async def test_delivery_via_enqueue(self) -> None:
        """Background tool returns a task id; the subagent's result is delivered
        to the parent via enqueue, surfacing as a follow-up ModelRequest in
        history that feeds one more LLM turn producing the final reply."""
        researcher = Agent(
            "researcher",
            config=TestConfig(ModelResponse(ModelMessage("Research findings."))),
        )

        orchestrator = Agent(
            "orchestrator",
            config=TestConfig(
                ToolCallEvent(name="background_task_researcher", arguments='{"objective": "Find X"}'),
                ModelResponse(ModelMessage("Started background task, awaiting result.")),
                ModelResponse(ModelMessage("Final reply incorporating research.")),
            ),
            tools=[
                background_agent_tool(
                    researcher,
                    description="Run researcher in the background.",
                )
            ],
        )

        reply = await orchestrator.ask("What's X?")

        assert reply.body == "Final reply incorporating research."

        events = list(await reply.context.stream.history.get_events())
        # The first ModelRequest is the user's initial "What's X?". The second
        # is the loop's follow-up turn carrying the background result.
        requests = [e for e in events if isinstance(e, ModelRequest)]
        assert len(requests) == 2
        follow_up = requests[1]
        assert "Research findings." in follow_up.parts[0].content

    async def test_enqueue_merges_multiple_calls(self) -> None:
        """Multiple ctx.enqueue() calls coalesce into a single DrainedModelRequest."""

        @tool
        async def double_enqueue(ctx: Context) -> str:
            """Test tool that enqueues two messages."""
            ctx.enqueue("first part")
            ctx.enqueue("second part")
            return "queued"

        agent = Agent(
            "agent",
            config=TestConfig(
                ToolCallEvent(name="double_enqueue", arguments="{}"),
                ModelResponse(ModelMessage("Done.")),
            ),
            tools=[double_enqueue],
        )

        reply = await agent.ask("go")

        events = list(await reply.context.stream.history.get_events())
        drained = [e for e in events if isinstance(e, DrainedModelRequest)]
        assert len(drained) == 1
        assert [p.content for p in drained[0].parts] == ["first part", "second part"]

    async def test_background_exception_propagates(self) -> None:
        """An exception raised inside a spawn_background coroutine propagates out
        of ask() once the main loop reaches the background-task wait branch."""

        async def failing_bg() -> None:
            # Brief sleep so the task is still pending when the main loop reaches
            # the background-wait branch — otherwise the done-callback would
            # remove it from the set before _continue_turn observes it.
            await asyncio.sleep(0.01)
            raise ValueError("background boom")

        @tool
        async def start_failing(ctx: Context) -> str:
            ctx.spawn_background(failing_bg())
            return "started"

        agent = Agent(
            "agent",
            config=TestConfig(
                ToolCallEvent(name="start_failing", arguments="{}"),
                ModelResponse(ModelMessage("Will wait for bg.")),
            ),
            tools=[start_failing],
        )

        with pytest.raises(ValueError, match="background boom"):
            await agent.ask("go")


@pytest.mark.asyncio
class TestLifecycle:
    async def test_cleanup_cancels_background_on_main_exception(self) -> None:
        """If the main agent loop fails, spawned background tasks must be
        cancelled and awaited before _execute returns."""
        captured: list[asyncio.Task[None]] = []

        async def long_running() -> None:
            await asyncio.sleep(10)

        @tool
        async def start_long(ctx: Context) -> str:
            captured.append(ctx.spawn_background(long_running()))
            return "started"

        agent = Agent(
            "agent",
            config=TestConfig(
                ToolCallEvent(name="start_long", arguments="{}"),
                # No further responses — the second LLM call exhausts the
                # TestConfig iterator, the loop raises, and `_execute`'s
                # finally must clean up the background task.
            ),
            tools=[start_long],
        )

        with pytest.raises(Exception):
            await agent.ask("go")

        assert len(captured) == 1
        bg_task = captured[0]
        assert bg_task.done()
        assert bg_task.cancelled()

    async def test_enqueue_and_spawn_outside_ask_raise(self) -> None:
        """ctx.enqueue / ctx.spawn_background outside a live ask() raise RuntimeError."""
        ctx = Context(stream=MemoryStream())

        with pytest.raises(RuntimeError, match="live Agent.ask"):
            ctx.enqueue("hello")

        async def noop() -> None:
            return None

        coro = noop()
        try:
            with pytest.raises(RuntimeError, match="live Agent.ask"):
                ctx.spawn_background(coro)
        finally:
            coro.close()

    async def test_lifecycle_teardown_restores_pending_messages_to_none(self) -> None:
        """After ask() returns, the per-run state on Context is torn back down
        to None — guarding against an outer caller seeing a torn lifecycle as
        already-owned and skipping its own initialization."""
        agent = Agent("agent", config=TestConfig(ModelResponse(ModelMessage("hi"))))

        reply = await agent.ask("hi")

        assert reply.context.pending_messages is None
