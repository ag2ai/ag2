# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("google.genai")

from google.genai import types as gtypes

from ag2 import MemoryStream
from ag2.context import ConversationContext
from ag2.events import (
    BaseEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelResponse,
    SynthesizedAudioEvent,
)
from ag2.live import gemini

pytestmark = pytest.mark.asyncio

AUDIO = b"\x00\x01"
THOUGHT = gtypes.Part(text="Planning a short greeting.", thought=True)
TURN_COMPLETE = gtypes.LiveServerMessage(server_content=gtypes.LiveServerContent(turn_complete=True))


class FakeSession:
    """Stands in for `google.genai.live.AsyncSession`: replays one model turn, then closes."""

    def __init__(self, turn: tuple[gtypes.LiveServerMessage, ...]) -> None:
        # `receive()` covers one turn per call; an empty one is a closed connection.
        self._turns = [turn, ()]

    async def receive(self) -> AsyncGenerator[gtypes.LiveServerMessage]:
        for message in self._turns.pop(0):
            yield message


class FakeClient:
    """Stands in for `google.genai.Client`: `client.aio.live.connect(...)` opens a `FakeSession`."""

    def __init__(self, *turn: gtypes.LiveServerMessage) -> None:
        self._turn = turn
        self.aio = self
        self.live = self

    @asynccontextmanager
    async def connect(self, **kwargs: Any) -> AsyncGenerator[FakeSession]:
        yield FakeSession(self._turn)


def model_turn(*parts: gtypes.Part, transcription: str | None = None) -> gtypes.LiveServerMessage:
    """A server message with `parts` of the model turn and, optionally, the transcription of its audio."""
    return gtypes.LiveServerMessage(
        server_content=gtypes.LiveServerContent(
            model_turn=gtypes.Content(role="model", parts=list(parts)),
            output_transcription=gtypes.Transcription(text=transcription) if transcription else None,
        ),
    )


async def run_turn(config: gemini.RealTimeConfig) -> list[BaseEvent]:
    """Run one model turn and return what the session put on the stream."""
    context = ConversationContext(stream=MemoryStream())
    seen: list[BaseEvent] = []

    async def collect(event: BaseEvent) -> None:
        seen.append(event)

    context.stream.subscribe(collect)
    async with context.stream.get(ModelResponse) as turn_done, config.session(context, serializer=None):  # type: ignore[arg-type]
        await turn_done

    return seen


async def test_thought_summary_is_reasoning_not_reply_text() -> None:
    client = FakeClient(
        model_turn(THOUGHT),
        model_turn(
            gtypes.Part(inline_data=gtypes.Blob(data=AUDIO, mime_type="audio/pcm;rate=24000")),
            transcription="Hello there.",
        ),
        TURN_COMPLETE,
    )
    config = gemini.RealTimeConfig("gemini-3.8-live-extended-thinking", client=client)  # type: ignore[arg-type]

    assert await run_turn(config) == [
        ModelReasoning("Planning a short greeting."),
        ModelMessageChunk("Hello there."),
        SynthesizedAudioEvent(AUDIO),
        ModelResponse(ModelMessage("Hello there.")),
    ]


async def test_text_parts_beside_a_thought_summary_are_the_reply_text() -> None:
    client = FakeClient(
        model_turn(THOUGHT, gtypes.Part(text="Hello ")),
        model_turn(gtypes.Part(text="there.")),
        TURN_COMPLETE,
    )
    config = gemini.RealTimeConfig(
        "gemini-3.8-live",
        output=gemini.TextOutput(),
        client=client,  # type: ignore[arg-type]
    )

    assert await run_turn(config) == [
        ModelReasoning("Planning a short greeting."),
        ModelMessageChunk("Hello "),
        ModelMessageChunk("there."),
        ModelResponse(ModelMessage("Hello there.")),
    ]
