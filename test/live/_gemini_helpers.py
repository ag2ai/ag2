# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator, Callable
from types import TracebackType
from typing import Any

from google.genai import types as gtypes

from ag2.live import LiveAgent
from ag2.live.gemini import RealTimeConfig
from ag2.stream import Stream
from ag2.tools.tool import Tool


class _Recorder:
    def __init__(self, name: str, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self._name = name
        self._calls = calls

    async def __call__(self, **kwargs: Any) -> None:
        self._calls.append((self._name, kwargs))


class FakeSession:
    """Stands in for `google.genai.live.AsyncSession`.

    Records client calls and replays server messages scripted by the test.
    `receive()` ends after a message that completes the interaction, as the
    SDK's does. `emit` returns once the session has handled every emitted
    message, so a test observes the calls each message caused before
    emitting the next.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.send_client_content = _Recorder("send_client_content", self.calls)
        self.send_realtime_input = _Recorder("send_realtime_input", self.calls)
        self.send_tool_response = _Recorder("send_tool_response", self.calls)
        self._messages: asyncio.Queue[gtypes.LiveServerMessage] = asyncio.Queue()
        self._handling = False

    async def emit(self, *messages: gtypes.LiveServerMessage) -> None:
        for message in messages:
            await self._messages.put(message)
        await self._messages.join()

    def response_requests(self) -> int:
        """Count `send_client_content` calls that complete the client turn — each asks for an answer."""
        return sum(name == "send_client_content" and kwargs.get("turn_complete", True) for name, kwargs in self.calls)

    def added_turns(self) -> list[dict[str, Any]]:
        return [
            turn for name, kwargs in self.calls if name == "send_client_content" for turn in kwargs.get("turns") or ()
        ]

    def tool_responses(self) -> list[Any]:
        return [kwargs["function_responses"] for name, kwargs in self.calls if name == "send_tool_response"]

    async def receive(self) -> AsyncGenerator[gtypes.LiveServerMessage]:
        while True:
            if self._handling:
                self._messages.task_done()
            message = await self._messages.get()
            self._handling = True
            yield message
            if _completes_interaction(message):
                return

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


def _completes_interaction(message: gtypes.LiveServerMessage) -> bool:
    content = message.server_content
    if content is None or not content.turn_complete:
        return False
    return content.interaction_status != gtypes.InteractionStatus.IN_PROGRESS


class FakeClient:
    """Stands in for `google.genai.Client`: `client.aio.live.connect(...)` opens `session`."""

    def __init__(self) -> None:
        self.session = FakeSession()
        self.aio = self
        self.live = self

    def connect(self, **kwargs: Any) -> FakeSession:
        return self.session


def live_agent(
    *tools: Callable[..., Any] | Tool,
    stream: Stream | None = None,
) -> tuple[LiveAgent, FakeSession]:
    """A `LiveAgent` on Gemini Live whose session is a `FakeSession`."""
    client = FakeClient()
    config = RealTimeConfig("gemini-live-2.5-flash-preview", client=client)  # type: ignore[arg-type]
    return LiveAgent("assistant", config=config, tools=tools, stream=stream), client.session


def speech(text: str = "Hello.") -> gtypes.LiveServerMessage:
    """A piece of the model's answer: the first one of a turn starts a response."""
    return gtypes.LiveServerMessage(
        server_content=gtypes.LiveServerContent(
            model_turn=gtypes.Content(role="model", parts=[gtypes.Part(text=text)]),
        ),
    )


def turn_complete(*, more_coming: bool = False) -> gtypes.LiveServerMessage:
    """The response boundary; `more_coming` marks a turn the server has already started after it."""
    return gtypes.LiveServerMessage(
        server_content=gtypes.LiveServerContent(
            turn_complete=True,
            interaction_status=(gtypes.InteractionStatus.IN_PROGRESS if more_coming else gtypes.InteractionStatus.IDLE),
        ),
    )


def tool_call(call_id: str, name: str = "lookup", args: dict[str, Any] | None = None) -> gtypes.LiveServerMessage:
    return gtypes.LiveServerMessage(
        tool_call=gtypes.LiveServerToolCall(
            function_calls=[gtypes.FunctionCall(id=call_id, name=name, args=args or {})],
        ),
    )
