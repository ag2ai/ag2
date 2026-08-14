# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from contextlib import asynccontextmanager, suppress

from fast_depends.library.serializer import SerializerProto

from ag2.config.config import ModelConfig
from ag2.context import ConversationContext
from ag2.events import (
    AudioInterruptedEvent,
    AudioPlaybackCompletedEvent,
    AudioPlaybackStartedEvent,
    BaseEvent,
    ModelMessageChunk,
    ModelResponse,
    RecordedAudioEvent,
    SynthesizedAudioEvent,
    TextInput,
    ToolResultsEvent,
    UsageEvent,
)
from ag2.events.input_events import ModelRequest
from ag2.tools.schemas import ToolSchema

from ..config import LLMClient
from .observer import Speech
from .protocols import TTSConfig
from .realtime import RealtimeConfig
from .stt import STTConfig, VoiceInput
from .turn import SilenceTurnDetector, TurnDetector

# Cap on tool round-trips within one turn. A cascade turn is driven by a live
# microphone; a model stuck in a tool loop would hold the floor indefinitely.
MAX_TOOL_ITERATIONS = 10


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
    ) -> AsyncIterator[None]:
        client: LLMClient = self._model.create()
        detector = self._turn_detector()
        schemas = list(tools)

        # `LiveAgent` builds the context without a prompt and hands the agent's
        # instructions to the session instead, so the system prompt has to be
        # installed here for the client to pick it up.
        prompt_mark = len(context.prompt)
        context.prompt.extend(instructions)

        turn: asyncio.Task[None] | None = None
        # Whether our own voice is currently coming out of the speaker
        playing = False

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
            nonlocal turn

            if playing and not self._barge_in:
                return

            was_speaking = detector.speaking
            voice = detector.push(event.content)

            # Barge-in: the user started talking over the reply.
            if self._barge_in and not was_speaking and detector.speaking and _running(turn):
                assert turn is not None
                turn.cancel()
                await context.send(AudioInterruptedEvent())

            if voice is None:
                return

            if _running(turn):
                assert turn is not None
                await turn

            # `spawn_background` (not a bare `create_task`) so a turn that
            # raises is logged rather than vanishing into an unretrieved task.
            turn = context.spawn_background(self._turn(voice, context, client, schemas, serializer))

        with (
            context.stream.where(RecordedAudioEvent).sub_scope(_on_audio),
            context.stream.where(AudioPlaybackStartedEvent).sub_scope(_on_playback_started),
            context.stream.where(AudioPlaybackCompletedEvent).sub_scope(_on_playback_completed),
        ):
            try:
                yield

            finally:
                if _running(turn):
                    assert turn is not None
                    turn.cancel()
                    with suppress(asyncio.CancelledError):
                        await turn
                del context.prompt[prompt_mark:]

    async def _turn(
        self,
        voice: VoiceInput,
        context: ConversationContext,
        client: LLMClient,
        schemas: list[ToolSchema],
        serializer: SerializerProto,
    ) -> None:
        """One user utterance: transcribe, answer, speak."""
        text = await self._stt.transcribe(voice, context)
        if not text.strip():
            return

        await context.send(ModelRequest([TextInput(text)]))
        await self._respond(context, client, schemas, serializer)

    async def _respond(
        self,
        context: ConversationContext,
        client: LLMClient,
        schemas: list[ToolSchema],
        serializer: SerializerProto,
    ) -> None:
        speech = Speech(self._tts, self._min_chars)

        with context.stream.where(ModelMessageChunk).sub_scope(speech.on_chunk):
            for _ in range(MAX_TOOL_ITERATIONS):
                messages = _conversation(await context.stream.history.get_events())
                response = await client(  # type: ignore[operator]
                    messages,
                    context,
                    tools=schemas,
                    response_schema=None,
                    serializer=serializer,
                )

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
    """Drop the raw audio before handing history to the model.

    A live session logs every microphone chunk and every synthesized chunk —
    ten-plus events per second, each carrying PCM. The provider mappers ignore
    them, but they would still be walked (and held) on every model call, so a
    long conversation pays for the whole recording on each turn. Transcripts
    carry the meaning; the bytes are for the recorder and the speaker.
    """
    return [e for e in events if not isinstance(e, (RecordedAudioEvent, SynthesizedAudioEvent))]


def _running(task: "asyncio.Task[None] | None") -> bool:
    return task is not None and not task.done()
