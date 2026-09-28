# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterable, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fast_depends.library.serializer import SerializerProto

from ag2.config.config import ModelConfig
from ag2.context import ConversationContext
from ag2.events import (
    AudioInterruptedEvent,
    AudioPlaybackCompletedEvent,
    AudioPlaybackStartedEvent,
    BaseEvent,
    Input,
    ModelMessageChunk,
    ModelRequest,
    ModelResponse,
    RecordedAudioEvent,
    SynthesizedAudioEvent,
    TextInput,
    ToolResultsEvent,
    UsageEvent,
    VoiceTurn,
)
from ag2.tools.schemas import ToolSchema

from ..config import LLMClient
from ._input import SENDABLE_INPUTS, sendable_parts
from .observer import Speech
from .protocols import TTSConfig
from .realtime import RealtimeConfig
from .stt import STTConfig, VoiceInput
from .turn import SilenceTurnDetector, TurnDetector

# Cap on tool round-trips within one turn. A cascade turn is driven by a live
# microphone; a model stuck in a tool loop would hold the floor indefinitely.
MAX_TOOL_ITERATIONS = 10

_PROVIDER = "cascade"


def _forget_answered(unread: list[ModelRequest], events: Sequence[BaseEvent]) -> None:
    """Remove from `unread` the pushed requests among `events`, which a model call read and answered."""
    unread[:] = [r for r in unread if not any(e == r and e.created_at == r.created_at for e in events)]


class CascadeConfig(RealtimeConfig):
    """A separate STT, LLM, and TTS wired to look like one realtime session.

    Implements `RealtimeConfig`, so `LiveAgent` drives it exactly as it drives
    `OpenAIRealTimeConfig` or `GeminiRealTimeConfig` — same events in, same
    events out, and `SoundDeviceRecorder` / `SoundDevicePlayer` need no changes:

        agent = LiveAgent(
            "assistant",
            config=CascadeConfig(
                stt=ElevenLabsTranscriber("scribe_v2"),
                model=config.OpenAIConfig("gpt-5-mini", streaming=True),
                tts=ElevenLabsStreamingTTSConfig("eleven_flash_v2_5"),
            ),
        )

    The session is half-duplex by default: while the reply is playing, the
    microphone is ignored. On speakers the mic hears the reply, and a cascade
    that listens to itself interrupts its own sentence and then answers its own
    words — see `barge_in` to trade that safety for interruptibility.

    barge_in=True allows for interruptions, but if you are on speakers, AI will detect
    its own speech as yours. With headphones, interruptions work nicely, no VAD logic currently.

    Each utterance is published as a `VoiceTurn` and answered. A `ModelRequest`
    pushed on the stream (typed text, or the inbox `LiveAgent` drains) joins
    the same conversation and is answered too. Turns run one at a time: a push
    that arrives while a turn is running is answered after it, and several
    pushes waiting on one turn get one answer — none at all if a model call
    of the running turn has already read them. Barge-in cuts off the reply,
    not pushed input still waiting for an answer: that is answered once the
    interrupting utterance ends. Pushed input is limited to
    `TextInput` and `DataInput`, as `RealtimeConfig` describes; other parts
    never reach the model.
    """

    def __init__(
        self,
        *,
        stt: STTConfig,
        model: ModelConfig,
        tts: TTSConfig[bytes],
        turn_detector: Callable[[], TurnDetector] = SilenceTurnDetector,
        barge_in: bool = False,
        min_chars: int = 60,
    ) -> None:
        self._stt = stt
        self._model = model
        self._tts = tts
        self._turn_detector = turn_detector
        # Off by default. Shouldn't be on speakers to be on, otherwise AI's own speech gets detected as user's input.
        self._barge_in = barge_in
        self._min_chars = min_chars

    @asynccontextmanager
    async def session(
        self,
        context: ConversationContext,
        *,
        instructions: Iterable[str] = (),
        tools: Iterable[ToolSchema] = (),
        serializer: SerializerProto,
    ) -> AsyncGenerator[None]:
        client: LLMClient = self._model.create()
        detector = self._turn_detector()
        schemas = list(tools)

        # `LiveAgent` builds the context without a prompt and hands the agent's
        # instructions to the session instead, so the system prompt has to be
        # installed here for the client to pick it up.
        prompt_mark = len(context.prompt)
        context.prompt.extend(instructions)

        # Every turn takes this lock, so turns run one at a time in the order
        # they were scheduled (`asyncio.Lock` wakes its waiters first in, first out).
        lock = asyncio.Lock()
        turns: set[asyncio.Task[None]] = set()
        # Pushed requests no successful model call has read yet
        unread: list[ModelRequest] = []
        # Whether our own voice is currently coming out of the speaker
        playing = False

        def _spawn(turn: Coroutine[Any, Any, None]) -> None:
            # `spawn_background` (not a bare `create_task`) so a turn that
            # raises is logged rather than vanishing into an unretrieved task.
            task = context.spawn_background(_one_at_a_time(lock, turn))
            turns.add(task)
            task.add_done_callback(turns.discard)

        async def _on_playback_started(event: AudioPlaybackStartedEvent) -> None:
            nonlocal playing
            playing = True

        async def _on_playback_completed(event: AudioPlaybackCompletedEvent) -> None:
            nonlocal playing
            playing = False
            # The tail of the reply is still in the detector's prefix buffer;
            # without this the first "user" turn after every answer opens on an
            # echo of the answer.
            detector.reset()

        async def _on_audio(event: RecordedAudioEvent) -> None:
            if playing and not self._barge_in:
                return

            was_speaking = detector.speaking
            voice = detector.push(event.content)

            # Barge-in: the user started talking over the reply.
            if self._barge_in and not was_speaking and detector.speaking and turns:
                for task in turns:
                    task.cancel()
                await context.send(AudioInterruptedEvent())

            if voice is not None:
                _spawn(self._turn(voice, context, client, schemas, serializer, unread))
            elif was_speaking and not detector.speaking:
                # The speech was too short to be a turn. It may still have cut
                # off the answer to pushed input, which is answered now.
                _spawn(self._answer_pushed(context, client, schemas, serializer, unread))

        async def _on_request(event: ModelRequest) -> None:
            if isinstance(event, VoiceTurn):
                return
            # Refuses (raises to the caller) or drops, with a warning, what the
            # model is never sent; `_conversation` keeps it out of every turn.
            if sendable_parts(event, provider=_PROVIDER):
                unread.append(event)
                _spawn(self._answer_pushed(context, client, schemas, serializer, unread))

        with (
            context.stream.where(RecordedAudioEvent).sub_scope(_on_audio),
            context.stream.where(ModelRequest).sub_scope(_on_request),
            context.stream.where(AudioPlaybackStartedEvent).sub_scope(_on_playback_started),
            context.stream.where(AudioPlaybackCompletedEvent).sub_scope(_on_playback_completed),
        ):
            try:
                yield

            finally:
                running = list(turns)
                for task in running:
                    task.cancel()
                await asyncio.gather(*running, return_exceptions=True)
                del context.prompt[prompt_mark:]

    async def _turn(
        self,
        voice: VoiceInput,
        context: ConversationContext,
        client: LLMClient,
        schemas: list[ToolSchema],
        serializer: SerializerProto,
        unread: list[ModelRequest],
    ) -> None:
        """One user utterance: transcribe, answer, speak.

        An utterance without words (a cough) may still have cut off the answer
        to pushed input, which is answered in its place.
        """
        text = await self._stt.transcribe(voice, context)
        if not text.strip():
            await self._answer_pushed(context, client, schemas, serializer, unread)
            return

        await context.send(VoiceTurn([TextInput(text)]))
        await self._respond(context, client, schemas, serializer, unread)

    async def _answer_pushed(
        self,
        context: ConversationContext,
        client: LLMClient,
        schemas: list[ToolSchema],
        serializer: SerializerProto,
        unread: list[ModelRequest],
    ) -> None:
        """Answer pushed input, unless a model call has already read all of it."""
        if unread:
            await self._respond(context, client, schemas, serializer, unread)

    async def _respond(
        self,
        context: ConversationContext,
        client: LLMClient,
        schemas: list[ToolSchema],
        serializer: SerializerProto,
        unread: list[ModelRequest],
    ) -> None:
        speech = Speech(self._tts, self._min_chars)

        with context.stream.where(ModelMessageChunk).sub_scope(speech.on_chunk):
            for _ in range(MAX_TOOL_ITERATIONS):
                # A copy: the stored history keeps growing during the model call.
                events = list(await context.stream.history.get_events())
                response = await client(  # type: ignore[operator]
                    _conversation(events),
                    context,
                    tools=schemas,
                    response_schema=None,
                    serializer=serializer,
                )
                _forget_answered(unread, events)

                if response.usage:
                    await context.send(
                        UsageEvent(
                            response.usage,
                            kind="model_call",
                            model=response.model,
                            provider=response.provider,
                            finish_reason=response.finish_reason,
                        )
                    )
                await context.send(response)

                if not response.tool_calls:
                    await speech.finish(response.content, context)
                    return

                async with context.stream.get(ToolResultsEvent | ModelResponse) as results:
                    await context.send(response.tool_calls)
                    settled = await results

                if isinstance(settled, ModelResponse):
                    await speech.finish(settled.content, context)
                    return

        raise RuntimeError(f"Tool loop exceeded {MAX_TOOL_ITERATIONS} iterations in one voice turn")


def _conversation(events: "Sequence[BaseEvent]") -> "list[BaseEvent]":
    """Select the events of the stream's history the model is sent.

    Raw audio is left out. A live session logs every microphone chunk and
    every synthesized chunk — ten-plus events per second, each carrying PCM.
    The provider mappers ignore them, but they would still be walked (and
    held) on every model call, so a long conversation pays for the whole
    recording on each turn. Transcripts carry the meaning; the bytes are for
    the recorder and the speaker.

    Parts a live session does not send are removed from every request —
    refused with a direct push, dropped from a drained inbox, or published on
    a shared stream by anyone else — and a request left with no parts is
    removed. History is re-read on every turn, so a part kept here would
    reach the model on every later turn too.
    """
    messages: list[BaseEvent] = []
    for event in events:
        if isinstance(event, (RecordedAudioEvent, SynthesizedAudioEvent)):
            continue
        if isinstance(event, ModelRequest) and not all(isinstance(p, SENDABLE_INPUTS) for p in event.parts):
            parts: list[Input] = [p for p in event.parts if isinstance(p, SENDABLE_INPUTS)]
            if not parts:
                continue
            request = type(event)(parts)
            request.created_at = event.created_at
            event = request
        messages.append(event)
    return messages


async def _one_at_a_time(lock: asyncio.Lock, turn: Coroutine[Any, Any, None]) -> None:
    """Run `turn` holding `lock`."""
    try:
        await lock.acquire()
    except BaseException:
        # Cancelled while waiting: `turn` never started.
        turn.close()
        raise
    try:
        await turn
    finally:
        lock.release()
