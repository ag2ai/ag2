# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Contract for ``Agent.run`` — the observable counterpart to ``Agent.ask``.

``run`` opens a turn *scope* and yields an ``AgentRun`` handle. The turn does
not advance until the caller drives it with ``result()``; while it drives, every
event flows through ``run.stream``, so observation is push-based (subscribe a
callback, or attach ``observers=``). No background task is involved.

* ``await run.result()`` — the authoritative, idempotent ``AgentReply``.
* ``run.stream`` — the underlying stream; subscribe to observe the turn live.
* The turn runs *inside* the block, only while ``result()`` is awaited; cancelling
  that await (e.g. a timeout) cancels the turn inline.

See docs/adr/0005-agent-run-turn-is-scoped-to-its-context-manager.md.
"""

import asyncio

import pytest
from pydantic import BaseModel

from autogen.beta import Agent, tool
from autogen.beta.events import (
    BaseEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolResultEvent,
)
from autogen.beta.stream import MemoryStream
from autogen.beta.testing import TestConfig, TrackingConfig


def _texts(event: BaseEvent) -> list[str]:
    """Text parts of a ``ModelRequest``-like event (empty for anything else)."""
    return [p.content for p in getattr(event, "parts", []) if isinstance(p, TextInput)]


@pytest.mark.asyncio
class TestResult:
    async def test_returns_reply_body(self) -> None:
        agent = Agent("runner", config=TestConfig("hello"))

        async with agent.run("Hi!") as run:
            result = await run.result()

        assert result.body == "hello"

    async def test_is_idempotent(self) -> None:
        agent = Agent("runner", config=TestConfig("hello"))

        async with agent.run("Hi!") as run:
            first = await run.result()
            second = await run.result()

        assert first is second

    async def test_reraises_turn_failure(self) -> None:
        @tool
        async def explode() -> str:
            """Always raises."""
            raise RuntimeError("tool blew up")

        agent = Agent(
            "runner",
            config=TestConfig(
                ToolCallEvent(name="explode", arguments="{}"),
                "unreachable",
            ),
            tools=[explode],
        )

        with pytest.raises(RuntimeError, match="tool blew up"):
            async with agent.run("Hi!") as run:
                await run.result()

    async def test_content_validates_against_schema(self) -> None:
        class Answer(BaseModel):
            value: int

        agent = Agent(
            "runner",
            config=TestConfig('{"value": 42}'),
            response_schema=Answer,
        )

        async with agent.run("Hi!") as run:
            content = await (await run.result()).content()

        assert content == Answer(value=42)


@pytest.mark.asyncio
async def test_reply_run_continues_conversation() -> None:
    agent = Agent(
        "runner",
        config=TestConfig(
            "first",
            "second",
        ),
    )

    async with agent.run("Hi!") as run:
        first = await run.result()
    assert first.body == "first"

    async with first.run("again") as run2:
        second = await run2.result()

    assert second.body == "second"
    # Continuation: same context/stream, history accumulates across both turns.
    assert second.context is first.context


@pytest.mark.asyncio
async def test_events_are_observable_on_stream_while_driving() -> None:
    tool_call = ToolCallEvent(name="ping", arguments="{}")
    agent = Agent(
        "runner",
        config=TestConfig(tool_call, "done"),
        tools=[tool(lambda: "pong", name="ping")],
    )

    seen: list = []

    async def capture(event) -> None:
        seen.append(event)

    async with agent.run("Hi!") as run:
        run.stream.subscribe(capture)  # subscribe before driving
        result = await run.result()  # callbacks fire inline as the turn drives

    assert [e for e in seen if isinstance(e, ToolCallEvent)] == [tool_call]
    assert any(isinstance(e, ModelResponse) for e in seen)
    assert result.body == "done"


@pytest.mark.asyncio
async def test_stream_is_exposed() -> None:
    stream = MemoryStream()
    agent = Agent("runner", config=TestConfig(ModelResponse(ModelMessage("hi"))))

    async with agent.run("Hi!", stream=stream) as run:
        assert run.stream is stream
        await run.result()


@pytest.mark.asyncio
async def test_turn_runs_on_result_not_on_enter() -> None:
    stream = MemoryStream()
    agent = Agent("runner", config=TestConfig(ModelResponse(ModelMessage("hi"))))

    async with agent.run("Hi!", stream=stream) as run:
        await asyncio.sleep(0.05)  # a background task, if any, would drive the turn here
        before = await stream.history.get_events()
        assert not any(isinstance(e, ModelResponse) for e in before), "no model call before result()"
        result = await run.result()
        after = await stream.history.get_events()

    assert any(isinstance(e, ModelResponse) for e in after)
    assert result.body == "hi"


@pytest.mark.asyncio
async def test_turn_never_runs_without_result() -> None:
    stream = MemoryStream()
    agent = Agent("runner", config=TestConfig(ModelResponse(ModelMessage("hi"))))

    async with agent.run("Hi!", stream=stream):
        await asyncio.sleep(0.05)  # a background task, if any, would drive the turn here

    events = await stream.history.get_events()
    assert not any(isinstance(e, ModelResponse) for e in events), "an undriven run must not call the model"


@pytest.mark.asyncio
class TestEnqueue:
    async def test_forwards_to_stream_inbox(self) -> None:
        agent = Agent("runner", config=TestConfig("ok"))

        async with agent.run("Hi!") as run:
            run.enqueue("queued")
            via_handle = list(run.stream.pending_messages)
            await run.result()

        assert via_handle == [ModelRequest.ensure_request(["queued"])]

    async def test_before_result_merges_into_first_model_call(self) -> None:
        config = TrackingConfig(TestConfig("answer"))
        agent = Agent("runner", config=config)

        async with agent.run("Hi!") as run:
            run.enqueue("extra context")
            result = await run.result()

        first_seen = config.mock.call_args_list[0].args[0]
        assert _texts(first_seen) == ["extra context", "Hi!"]
        assert result.body == "answer"

    async def test_during_turn_is_consumed_by_the_same_turn(self) -> None:
        config = TrackingConfig(TestConfig(ToolCallEvent(name="ping", arguments="{}"), "r1", "r2", "r3"))
        agent = Agent("runner", config=config, tools=[tool(lambda: "pong", name="ping")])

        injected = False
        async with agent.run("Hi!") as run:

            @run.stream.where(ToolResultEvent).subscribe
            async def inject(_event: BaseEvent) -> None:
                nonlocal injected
                if not injected:
                    injected = True
                    run.enqueue("injected message")

            await run.result()

        seen = [text for call in config.mock.call_args_list for text in _texts(call.args[0])]
        assert "injected message" in seen, "a message enqueued mid-turn must reach a model call in that turn"


@pytest.mark.asyncio
async def test_cancelling_result_cancels_turn_inline() -> None:
    cancelled = asyncio.Event()

    @tool
    async def block() -> str:
        """Blocks forever, flagging if it is cancelled."""
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "never"

    agent = Agent(
        "runner",
        config=TestConfig(
            ToolCallEvent(name="block", arguments="{}"),
            "unreachable",
        ),
        tools=[block],
    )

    async with agent.run("Hi!") as run:
        with pytest.raises(asyncio.TimeoutError):
            # asyncio.wait_for (vs asyncio.timeout) keeps this runnable on Python 3.10.
            await asyncio.wait_for(run.result(), timeout=0.2)

    assert cancelled.is_set(), "cancelling the result() await must cancel the in-flight turn"
