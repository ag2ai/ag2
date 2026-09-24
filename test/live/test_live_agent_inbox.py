# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import ANY, MagicMock

import pytest
from dirty_equals import IsStr

pytest.importorskip("openai")

from ag2 import Agent, Context, observer
from ag2.events import MessageEnqueued, ModelMessage, ModelRequest, ModelResponse, TextInput, ToolCallEvent
from ag2.stream import MemoryStream
from ag2.testing import TestConfig
from ag2.tools.subagents import background_agent_tool
from test.live._helpers import created, done, function_call, live_agent


class Announcements:
    """Counts `MessageEnqueued` events once every earlier subscriber has handled them.

    Subscribed after the session opens, so `LiveAgent`'s own subscriber runs
    first: once `wait(n)` returns, the session has finished reacting to the
    first `n` announcements.
    """

    def __init__(self) -> None:
        self._seen = 0
        self._changed = asyncio.Condition()

    async def on_enqueued(self, event: MessageEnqueued) -> None:
        async with self._changed:
            self._seen += 1
            self._changed.notify_all()

    async def wait(self, count: int) -> None:
        async with self._changed:
            await asyncio.wait_for(self._changed.wait_for(lambda: self._seen >= count), timeout=3.0)


class Hold:
    """A subagent tool that finishes only once the test releases it."""

    def __init__(self) -> None:
        self._released = asyncio.Event()

    def release(self) -> None:
        self._released.set()

    async def hold(self) -> str:
        """Wait for the research to finish."""
        await self._released.wait()
        return "done"


def user_texts(items: list[dict[str, Any]]) -> list[list[str]]:
    return [[c["text"] for c in item["content"]] for item in items if item.get("role") == "user"]


async def model_requests(stream: MemoryStream) -> list[list[str]]:
    events = await stream.history.get_events()
    return [[p.content for p in e.parts if isinstance(p, TextInput)] for e in events if isinstance(e, ModelRequest)]


def sync_note(ctx: Context) -> str:
    """Leave a note for the model."""
    ctx.enqueue("from tool")
    return "noted"


async def async_note(ctx: Context) -> str:
    """Leave a note for the model."""
    ctx.enqueue("from tool")
    return "noted"


@pytest.mark.asyncio
class TestLiveAgentInbox:
    async def test_message_enqueued_before_open_is_delivered_at_open(self) -> None:
        stream = MemoryStream()
        stream.enqueue("left over")
        agent, conn = live_agent(stream=stream)

        async with agent.run():
            assert user_texts(conn.created_items()) == [["left over"]]
            assert conn.response_requests() == 1

    async def test_message_drained_at_open_reaches_observers(self) -> None:
        stream = MemoryStream()
        stream.enqueue("left over")
        agent, _ = live_agent(stream=stream)
        seen = MagicMock()

        async with agent.run(observers=[observer(ModelRequest, seen)]):
            seen.assert_called_once_with(ModelRequest([TextInput("left over")]), __ctx__=ANY)

    async def test_enqueue_during_session_reaches_connection_and_history(self) -> None:
        stream = MemoryStream()
        agent, conn = live_agent(stream=stream)

        async with agent.run() as context:
            announcements = Announcements()
            with context.stream.where(MessageEnqueued).sub_scope(announcements.on_enqueued):
                context.enqueue("typed")
                await announcements.wait(1)

            assert user_texts(conn.created_items()) == [["typed"]]
            assert conn.response_requests() == 1
            assert await model_requests(stream) == [["typed"]]

    async def test_enqueues_close_together_are_delivered_once_each(self) -> None:
        agent, conn = live_agent()

        async with agent.run() as context:
            announcements = Announcements()
            with context.stream.where(MessageEnqueued).sub_scope(announcements.on_enqueued):
                context.enqueue("first")
                context.enqueue("second")
                await announcements.wait(2)

            assert user_texts(conn.created_items()) == [["first", "second"]]

    async def test_empty_inbox_publishes_nothing(self) -> None:
        stream = MemoryStream()
        agent, _ = live_agent(stream=stream)

        async with agent.run() as context:
            announcements = Announcements()
            with context.stream.where(MessageEnqueued).sub_scope(announcements.on_enqueued):
                await context.send(MessageEnqueued())
                await announcements.wait(1)

            assert await model_requests(stream) == []

    @pytest.mark.parametrize("note", [sync_note, async_note], ids=["sync", "async"])
    async def test_tool_enqueue_reaches_connection(self, note: Callable[..., Any]) -> None:
        agent, conn = live_agent(note)

        async with agent.run() as context:
            announcements = Announcements()
            with context.stream.where(MessageEnqueued).sub_scope(announcements.on_enqueued):
                await conn.emit(created("resp-1"), function_call("resp-1", "call-1", name=note.__name__))
                await announcements.wait(1)

            assert user_texts(conn.created_items()) == [["from tool"]]

    async def test_background_subagent_result_is_delivered_while_silent(self) -> None:
        research = Hold()
        researcher = Agent(
            "researcher",
            config=TestConfig(
                ToolCallEvent(name="hold", arguments="{}"),
                ModelResponse(ModelMessage("Research findings.")),
            ),
            tools=[research.hold],
        )
        agent, conn = live_agent(background_agent_tool(researcher, description="Research in the background."))

        async with agent.run() as context:
            announcements = Announcements()
            with context.stream.where(MessageEnqueued).sub_scope(announcements.on_enqueued):
                await conn.emit(
                    created("resp-1"),
                    function_call(
                        "resp-1",
                        "call-1",
                        name="background_task_researcher",
                        arguments='{"objective": "Find X"}',
                    ),
                    done("resp-1"),
                    created("resp-2"),
                    done("resp-2"),
                )
                requests_while_silent = conn.response_requests()
                research.release()
                await announcements.wait(1)

            assert user_texts(conn.created_items()) == [[IsStr(regex=r"(?s).*Research findings\..*")]]
            assert conn.response_requests() == requests_while_silent + 1
