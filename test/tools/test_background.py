# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from ag2 import Agent, Context, MemoryStream, tool
from ag2.events import (
    DrainedModelRequest,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallEvent,
)
from ag2.testing import TestConfig
from ag2.tools.subagents import background_agent_tool


@pytest.mark.asyncio
class TestBackgroundDelivery:
    async def test_delivery_via_enqueue(self) -> None:
        """Background tool returns a task id; its enqueued result lands in the
        inbox before the next LLM call thanks to a follow-up tool that polls
        until the bg subagent has delivered its message."""

        @tool
        async def wait_for_bg(ctx: Context) -> str:
            """Yield until the background subagent's message has been drained."""
            for _ in range(300):
                events = list(await ctx.stream.history.get_events())
                drained = [e for e in events if isinstance(e, DrainedModelRequest)]
                if any("Research findings." in p.content for r in drained for p in r.parts if hasattr(p, "content")):
                    return "ok"
                if ctx.pending_messages:
                    return "ok"
                await asyncio.sleep(0.01)
            raise AssertionError("background never delivered")

        researcher = Agent(
            "researcher",
            config=TestConfig(ModelResponse(ModelMessage("Research findings."))),
        )

        bg_tool = background_agent_tool(
            researcher,
            description="Run researcher in the background.",
        )

        orchestrator = Agent(
            "orchestrator",
            config=TestConfig(
                ToolCallEvent(name="background_task_researcher", arguments='{"objective": "Find X"}'),
                ToolCallEvent(name="wait_for_bg", arguments="{}"),
                ModelResponse(ModelMessage("Final reply incorporating research.")),
            ),
            tools=[bg_tool, wait_for_bg],
        )

        reply = await asyncio.wait_for(orchestrator.ask("What's X?"), timeout=3.0)

        assert reply.body == "Final reply incorporating research."

        events = list(await reply.context.stream.history.get_events())
        drained = [e for e in events if isinstance(e, DrainedModelRequest)]
        assert any("Research findings." in p.content for r in drained for p in r.parts if hasattr(p, "content"))

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

    async def test_background_does_not_block_ask(self) -> None:
        """spawn_background is fire-and-forget — ask returns without waiting for
        the task to complete."""
        bg_started = asyncio.Event()
        bg_finished = asyncio.Event()

        async def long_bg() -> None:
            bg_started.set()
            try:
                await asyncio.sleep(5)
            finally:
                bg_finished.set()

        @tool
        async def fire_and_forget(ctx: Context) -> str:
            ctx.spawn_background(long_bg())
            return "started"

        agent = Agent(
            "agent",
            config=TestConfig(
                ToolCallEvent(name="fire_and_forget", arguments="{}"),
                ModelResponse(ModelMessage("done")),
            ),
            tools=[fire_and_forget],
        )

        reply = await asyncio.wait_for(agent.ask("go"), timeout=1.0)

        assert reply.body == "done"
        assert bg_started.is_set()
        # ask returned without awaiting the long-running bg task.
        assert not bg_finished.is_set()

    async def test_background_exception_is_logged_not_raised(self, caplog: pytest.LogCaptureFixture) -> None:
        """An exception inside a spawn_background coroutine is captured by the
        done-callback (logged) instead of propagating out of ask()."""

        bg_failed = asyncio.Event()

        async def failing_bg() -> None:
            try:
                raise ValueError("background boom")
            finally:
                bg_failed.set()

        @tool
        async def start_failing(ctx: Context) -> str:
            ctx.spawn_background(failing_bg())
            return "started"

        agent = Agent(
            "agent",
            config=TestConfig(
                ToolCallEvent(name="start_failing", arguments="{}"),
                ModelResponse(ModelMessage("ok")),
            ),
            tools=[start_failing],
        )

        with caplog.at_level("ERROR", logger="ag2.context"):
            reply = await asyncio.wait_for(agent.ask("go"), timeout=1.0)

        # Give the done-callback a chance to fire after the task raised.
        await asyncio.wait_for(bg_failed.wait(), timeout=1.0)
        await asyncio.sleep(0)

        assert reply.body == "ok"
        assert any(
            "background boom" in rec.message or "Background task raised" in rec.message for rec in caplog.records
        )


@pytest.mark.asyncio
class TestLifecycle:
    async def test_enqueue_outside_ask_does_not_raise(self) -> None:
        """ctx.enqueue outside a live ask() silently appends to the inbox; no
        runtime check guards it."""
        ctx = Context(stream=MemoryStream())

        ctx.enqueue("hello")

        assert len(ctx.pending_messages) == 1
        assert ctx.pending_messages[0].parts[0].content == "hello"

    async def test_pending_messages_initialized_empty(self) -> None:
        """After ask() returns, the stream's inbox is an empty list and no
        background tasks are leaked into the stream's task set."""
        agent = Agent("agent", config=TestConfig(ModelResponse(ModelMessage("hi"))))

        reply = await agent.ask("hi")

        assert reply.context.pending_messages == []
        assert reply.context.stream._background_tasks == set()

    async def test_stream_inbox_persists_across_asks(self) -> None:
        """Inbox lives on the stream: a message enqueued from a background task
        that finishes after the first ``ask`` returned gets merged into the
        next ``ask`` on the same stream — the LLM sees both as one user turn."""

        stream = MemoryStream()
        bg_delivered = asyncio.Event()

        async def deliver_later(ctx: Context) -> None:
            # Yield, then enqueue *after* the first ask returns. The result
            # sits in stream.pending_messages until the second ask picks it up.
            await asyncio.sleep(0.05)
            ctx.enqueue("late bg result")
            bg_delivered.set()

        @tool
        async def start_bg(ctx: Context) -> str:
            ctx.spawn_background(deliver_later(ctx))
            return "started"

        first_config = TestConfig(
            ToolCallEvent(name="start_bg", arguments="{}"),
            ModelResponse(ModelMessage("kicked off bg")),
        )
        second_config = TestConfig(ModelResponse(ModelMessage("got it")))

        agent = Agent("agent", config=first_config, tools=[start_bg])

        first = await asyncio.wait_for(agent.ask("go", stream=stream), timeout=1.0)
        assert first.body == "kicked off bg"

        # Bg runs in the background; wait for it to enqueue post-return.
        await asyncio.wait_for(bg_delivered.wait(), timeout=1.0)
        assert len(stream.pending_messages) == 1
        assert stream.pending_messages[0].parts[0].content == "late bg result"

        # Second ask on the same stream picks up the leftover and merges it
        # into the initial ModelRequest.
        second = await asyncio.wait_for(
            agent.ask("anything?", stream=stream, config=second_config),
            timeout=1.0,
        )
        assert second.body == "got it"

        # The merged request from the second ask carries both the leftover
        # bg message and the new user msg as parts of one ModelRequest.
        events = list(await second.context.stream.history.get_events())
        requests = [e for e in events if isinstance(e, ModelRequest)]
        merged = requests[-1]
        contents = [p.content for p in merged.parts if hasattr(p, "content")]
        assert "late bg result" in contents
        assert "anything?" in contents
        # Order: bg leftover prepended before the new user message.
        assert contents.index("late bg result") < contents.index("anything?")
