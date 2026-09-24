# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Callable
from types import SimpleNamespace, TracebackType
from typing import Any

from openai.types.realtime import (
    RealtimeError,
    RealtimeErrorEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
)
from openai.types.realtime.realtime_response import RealtimeResponse

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

    def created_items(self) -> list[dict[str, Any]]:
        return [kwargs["item"] for name, kwargs in self.calls if name == "conversation.item.create"]

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


def live_agent(*tools: Callable[..., Any]) -> tuple[LiveAgent, FakeConnection]:
    """A `LiveAgent` on OpenAI realtime whose connection is a `FakeConnection`."""
    client = FakeClient()
    config = RealTimeConfig("gpt-realtime", client=client)  # type: ignore[arg-type]
    return LiveAgent("assistant", config=config, tools=tools), client.connection


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
