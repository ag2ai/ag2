# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace, TracebackType
from typing import Any

import pytest

pytest.importorskip("openai")

from openai.types.realtime import (
    RealtimeConversationItemFunctionCall,
    RealtimeError,
    RealtimeErrorEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseOutputItemDoneEvent,
)
from openai.types.realtime.realtime_response import RealtimeResponse

from ag2.events import ToolResult, ToolResultEvent
from ag2.live import LiveAgent
from ag2.live.openai import RealTimeConfig


class _Recorder:
    def __init__(self, name: str, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self._name = name
        self._calls = calls

    async def __call__(self, **kwargs: Any) -> None:
        self._calls.append((self._name, kwargs))


class FakeConnection:
    """Records client calls and replays server events scripted by the test.

    `emit` returns once the session has handled every emitted event, so a
    test observes the calls each event caused before emitting the next.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.session = SimpleNamespace(update=_Recorder("session.update", self.calls))
        self.input_audio_buffer = SimpleNamespace(append=_Recorder("input_audio_buffer.append", self.calls))
        self.conversation = SimpleNamespace(
            item=SimpleNamespace(create=_Recorder("conversation.item.create", self.calls)),
        )
        self.response = SimpleNamespace(create=_Recorder("response.create", self.calls))
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        self._handling = False

    async def emit(self, *events: Any) -> None:
        for event in events:
            await self._events.put(event)
        await self._events.join()

    def response_requests(self) -> int:
        return sum(name == "response.create" for name, _ in self.calls)

    def last_response_request_id(self) -> str:
        event_ids: list[str] = [kwargs["event_id"] for name, kwargs in self.calls if name == "response.create"]
        return event_ids[-1]

    def __aiter__(self) -> "FakeConnection":
        return self

    async def __anext__(self) -> Any:
        if self._handling:
            self._events.task_done()
        event = await self._events.get()
        self._handling = True
        return event

    async def __aenter__(self) -> "FakeConnection":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


class FakeClient:
    """Stands in for `AsyncOpenAI`: `client.realtime.connect(...)` opens `connection`."""

    def __init__(self) -> None:
        self.connection = FakeConnection()
        self.realtime = self

    def connect(self, **kwargs: Any) -> FakeConnection:
        return self.connection


def created(response_id: str) -> ResponseCreatedEvent:
    return ResponseCreatedEvent(
        event_id=f"ev-created-{response_id}",
        response=RealtimeResponse(id=response_id, status="in_progress"),
        type="response.created",
    )


def done(response_id: str) -> ResponseDoneEvent:
    return ResponseDoneEvent(
        event_id=f"ev-done-{response_id}",
        response=RealtimeResponse(id=response_id, status="completed"),
        type="response.done",
    )


def active_response_rejection(client_event_id: str) -> RealtimeErrorEvent:
    return RealtimeErrorEvent(
        event_id="ev-error",
        error=RealtimeError(
            message="Conversation already has an active response in progress",
            type="invalid_request_error",
            code="conversation_already_has_active_response",
            event_id=client_event_id,
        ),
        type="error",
    )


def function_call(response_id: str, call_id: str) -> ResponseOutputItemDoneEvent:
    return ResponseOutputItemDoneEvent(
        event_id=f"ev-call-{call_id}",
        item=RealtimeConversationItemFunctionCall(
            type="function_call",
            call_id=call_id,
            name="lookup",
            arguments="{}",
        ),
        output_index=0,
        response_id=response_id,
        type="response.output_item.done",
    )


def lookup() -> str:
    """Look the answer up."""
    return "42"


def live_agent() -> tuple[LiveAgent, FakeConnection]:
    client = FakeClient()
    config = RealTimeConfig("gpt-realtime", client=client)  # type: ignore[arg-type]
    return LiveAgent("assistant", config=config, tools=[lookup]), client.connection


@pytest.mark.asyncio
class TestResponseGate:
    async def test_tool_result_while_idle_requests_response_immediately(self) -> None:
        agent, conn = live_agent()

        async with agent.run() as context:
            await context.send(ToolResultEvent(parent_id="call-1", name="lookup", result=ToolResult("42")))

            assert conn.response_requests() == 1

    async def test_unacknowledged_request_counts_as_active(self) -> None:
        agent, conn = live_agent()

        async with agent.run() as context:
            await context.send(ToolResultEvent(parent_id="call-1", name="lookup", result=ToolResult("42")))
            await context.send(ToolResultEvent(parent_id="call-2", name="lookup", result=ToolResult("43")))

            assert conn.response_requests() == 1

            await conn.emit(created("resp-1"), done("resp-1"))

            assert conn.response_requests() == 2

    async def test_rejected_request_is_retried_at_boundary(self) -> None:
        agent, conn = live_agent()

        async with agent.run() as context:
            await context.send(ToolResultEvent(parent_id="call-1", name="lookup", result=ToolResult("42")))
            await conn.emit(
                created("resp-vad"),
                active_response_rejection(conn.last_response_request_id()),
                done("resp-vad"),
            )

            assert conn.response_requests() == 2

    async def test_rejected_request_does_not_block_later_requests(self) -> None:
        agent, conn = live_agent()

        async with agent.run() as context:
            await context.send(ToolResultEvent(parent_id="call-1", name="lookup", result=ToolResult("42")))
            await conn.emit(active_response_rejection(conn.last_response_request_id()))
            await context.send(ToolResultEvent(parent_id="call-2", name="lookup", result=ToolResult("43")))

            assert conn.response_requests() == 2

    async def test_error_for_another_client_event_leaves_request_active(self) -> None:
        agent, conn = live_agent()

        async with agent.run() as context:
            await context.send(ToolResultEvent(parent_id="call-1", name="lookup", result=ToolResult("42")))
            await conn.emit(active_response_rejection("another-client-event"))
            await context.send(ToolResultEvent(parent_id="call-2", name="lookup", result=ToolResult("43")))

            assert conn.response_requests() == 1

    async def test_tool_result_during_response_waits_for_boundary(self) -> None:
        agent, conn = live_agent()

        async with agent.run():
            await conn.emit(created("resp-1"), function_call("resp-1", "call-1"))

            assert (
                "conversation.item.create",
                {"item": {"type": "function_call_output", "call_id": "call-1", "output": "42"}},
            ) in conn.calls
            assert conn.response_requests() == 0

            await conn.emit(done("resp-1"))

            assert conn.response_requests() == 1

    async def test_tool_results_during_one_response_share_one_request(self) -> None:
        agent, conn = live_agent()

        async with agent.run():
            await conn.emit(
                created("resp-1"),
                function_call("resp-1", "call-1"),
                function_call("resp-1", "call-2"),
                done("resp-1"),
            )

            assert conn.response_requests() == 1

    async def test_deferred_request_dropped_when_turn_detection_response_is_active(self) -> None:
        agent, conn = live_agent()

        async with agent.run():
            await conn.emit(
                created("resp-1"),
                function_call("resp-1", "call-1"),
                created("resp-vad"),
                done("resp-1"),
                done("resp-vad"),
            )

            assert conn.response_requests() == 0
