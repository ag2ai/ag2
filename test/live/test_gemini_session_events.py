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

pytestmark = [pytest.mark.asyncio, pytest.mark.gemini]


class FakeSession:
    """Stands in for `google.genai.live.AsyncSession`: replays one model turn, then closes."""

    def __init__(self, turn: tuple[gtypes.LiveServerContent, ...]) -> None:
        # `receive()` covers one turn per call; an empty one is a closed connection.
        self._turns = [turn, ()]

    async def receive(self) -> AsyncGenerator[gtypes.LiveServerMessage]:
        for content in self._turns.pop(0):
            yield gtypes.LiveServerMessage(server_content=content)


class FakeClient:
    """Stands in for `google.genai.Client`: `client.aio.live.connect(...)` opens a `FakeSession`.

    The SDK always dials `wss://`, so a local websocket server (as in the Mistral
    tests) would need a TLS certificate; a duck-typed client is the lighter seam.
    """

    def __init__(self, *turn: gtypes.LiveServerContent) -> None:
        self._turn = turn
        self.aio = self
        self.live = self

    @asynccontextmanager
    async def connect(self, **kwargs: Any) -> AsyncGenerator[FakeSession]:
        yield FakeSession(self._turn)


class Recorder:
    """Collects every event put on the stream."""

    def __init__(self) -> None:
        self.seen: list[BaseEvent] = []

    async def record(self, event: BaseEvent) -> None:
        self.seen.append(event)


async def test_thought_summaries_are_reasoning_not_reply_text() -> None:
    # The message layout of one real turn of this model, with no thinking config
    # set: each thought summary arrives in a message of its own, then the
    # transcription and the audio of the reply.
    client = FakeClient(
        gtypes.LiveServerContent(
            model_turn=gtypes.Content(parts=[gtypes.Part(text="**Analyzing the Problem**", thought=True)]),
        ),
        gtypes.LiveServerContent(
            model_turn=gtypes.Content(parts=[gtypes.Part(text="**Calculating the Solution**", thought=True)]),
        ),
        gtypes.LiveServerContent(output_transcription=gtypes.Transcription(text="The ball")),
        gtypes.LiveServerContent(output_transcription=gtypes.Transcription(text=" costs 0.05.")),
        gtypes.LiveServerContent(
            model_turn=gtypes.Content(parts=[gtypes.Part(inline_data=gtypes.Blob(data=b"\x00\x01"))]),
        ),
        gtypes.LiveServerContent(turn_complete=True),
    )
    config = gemini.RealTimeConfig(
        "gemini-2.5-flash-native-audio-preview-09-2025",
        client=client,  # type: ignore[arg-type]
    )

    context = ConversationContext(stream=MemoryStream())
    recorder = Recorder()
    context.stream.subscribe(recorder.record)

    async with context.stream.get(ModelResponse) as turn_done, config.session(context, serializer=None):  # type: ignore[arg-type]
        await turn_done

    assert recorder.seen == [
        ModelReasoning("**Analyzing the Problem**"),
        ModelReasoning("**Calculating the Solution**"),
        ModelMessageChunk("The ball"),
        ModelMessageChunk(" costs 0.05."),
        SynthesizedAudioEvent(b"\x00\x01"),
        ModelResponse(ModelMessage("The ball costs 0.05.")),
    ]
