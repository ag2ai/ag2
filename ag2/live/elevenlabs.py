# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from typing import Literal

from elevenlabs.client import AsyncElevenLabs

from ag2 import events
from ag2.annotations import Context
from ag2.observers import CompositeObserver, observer

from .observer import SentenceBuffer
from .protocols import TTSConfig as TTSConfigProtocol

DEFAULT_VOICE_ID = "Rachel"

OutputFormat = Literal[
    "pcm_16000",
    "pcm_22050",
    "pcm_24000",
    "pcm_44100",
    "mp3_44100_128",
]
DEFAULT_OUTPUT_FORMAT: OutputFormat = "pcm_24000"

# Not exhaustive — any model id string is accepted. `eleven_v3` is
# synchronous-only; the flash/turbo families support streaming.

StreamingModelId = Literal[
    "eleven_multilingual_v2",
    "eleven_turbo_v2_5",
    "eleven_flash_v2_5",
]
SyncOutputModelId = Literal["eleven_v3",]


class TTSConfig(TTSConfigProtocol[bytes]):
    """Synchronous TTS. Waits for the full clip, then returns it.

    This is the mode required by ``eleven_v3``, which cannot stream. Implements
    the `TTSConfig` protocol and plugs into `TTSObserver` unchanged.

    If you want streaming, use StreamingTTSConfig.
    """

    def __init__(
        self,
        model_id: "SyncOutputModelId | str" = "eleven_v3",
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        output_format: "OutputFormat | str" = DEFAULT_OUTPUT_FORMAT,
        client: "AsyncElevenLabs | None" = None,
    ) -> None:
        self._model_id = model_id
        self._voice_id = voice_id
        self._output_format = output_format
        self._client = client or AsyncElevenLabs()

    async def synthesize(self, text: str) -> bytes:
        # `convert` hits the non-streaming endpoint; it still returns an async
        # iterator of byte chunks, so we drain it into a single buffer.
        audio = self._client.text_to_speech.convert(
            voice_id=self._voice_id,
            text=text,
            model_id=self._model_id,
            output_format=self._output_format,
        )
        return b"".join([chunk async for chunk in audio])


class StreamingTTSConfig(TTSConfigProtocol[bytes]):
    """Streaming TTS via the ``/stream`` endpoint.

    Yields audio chunks as they are generated for lower time-to-first-byte.
    Also implements `synthesize` (by buffering the full stream) so it satisfies
    the `TTSConfig` protocol, but prefer `stream` / `StreamingTTSObserver` to
    actually take advantage of streaming. Not supported by ``eleven_v3``.

    Needs an api key in ELEVENLABS_API_KEY env var or can accept an api_key argument
    """

    def __init__(
        self,
        model_id: "StreamingModelId | str" = "eleven_flash_v2_5",
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        output_format: "OutputFormat | str" = DEFAULT_OUTPUT_FORMAT,
        client: "AsyncElevenLabs | None" = None,
    ) -> None:
        self._model_id = model_id
        self._voice_id = voice_id
        self._output_format = output_format
        self._client = client or AsyncElevenLabs()

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        audio = self._client.text_to_speech.stream(
            voice_id=self._voice_id,
            text=text,
            model_id=self._model_id,
            output_format=self._output_format,
        )
        async for chunk in audio:
            if chunk:
                yield chunk

    async def synthesize(self, text: str) -> bytes:
        return b"".join([chunk async for chunk in self.stream(text)])


def StreamingTTSObserver(config: StreamingTTSConfig, *, min_chars: int = 60) -> CompositeObserver:  # noqa: N802
    """Speak the model's reply, emitting each streamed audio chunk as it lands.

    Incremental on both axes. Text is flushed to ElevenLabs at sentence
    boundaries as the model produces it (like `TTSObserver`), and the audio for
    each sentence is forwarded chunk by chunk rather than buffered — so playback
    starts a sentence into the reply, not after the last token.

    `min_chars` is the smallest amount of text worth a request; below it the
    buffer keeps accumulating even past a sentence boundary.
    """
    buffer = SentenceBuffer(min_chars)
    # Non-streaming model configs emit no `ModelMessageChunk` at all, which
    # would otherwise leave the buffer empty and the observer silent.
    saw_chunks = False

    async def speak(text: str, context: Context) -> None:
        if not text:
            return
        async for chunk in config.stream(text):
            await context.send(events.SynthesizedAudioEvent(chunk))

    @observer(events.ModelMessageChunk)
    async def on_model_message_chunk(event: events.ModelMessageChunk, context: Context) -> None:
        nonlocal saw_chunks
        if not event.content:
            return
        saw_chunks = True
        if text := buffer.push(event.content):
            await speak(text, context)

    @observer(events.ModelMessage)
    async def on_model_message(event: events.ModelMessage, context: Context) -> None:
        nonlocal saw_chunks
        text = buffer.flush() if saw_chunks else (event.content or "").strip()
        saw_chunks = False
        await speak(text, context)

    return CompositeObserver(on_model_message_chunk, on_model_message)
