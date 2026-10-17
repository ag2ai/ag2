# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import re

from ag2 import events
from ag2.annotations import Context
from ag2.observers import CompositeObserver, observer

from .protocols import StreamingTTSConfig, TTSConfig


def TTSObserver(config: TTSConfig[bytes], *, min_chars: int = 60) -> CompositeObserver:  # noqa: N802
    """Speak the model's reply as it streams, a sentence at a time.

    Configs that also implement `StreamingTTSConfig` are streamed chunk by
    chunk, so playback starts at time-to-first-byte rather than after the last
    sample of the sentence; plain configs are synthesized whole.

    `min_chars` is the smallest amount of text worth a request; below it the
    buffer keeps accumulating even past a sentence boundary.
    """
    speech = Speech(config, min_chars)

    @observer(events.ModelMessageChunk)
    async def on_model_message_chunk(event: events.ModelMessageChunk, context: Context) -> None:
        await speech.on_chunk(event, context)

    @observer(events.ModelMessage)
    async def on_model_message(event: events.ModelMessage, context: Context) -> None:
        await speech.finish(event.content, context)

    return CompositeObserver(on_model_message_chunk, on_model_message)


class SentenceBuffer:
    """Accumulates streamed text and releases it at sentence boundaries.

    Pure text bookkeeping — no events, no synthesis — so `Speech` can decide
    "enough text to speak yet?" without caring how the audio is produced.
    """

    def __init__(self, min_chars: int = 60) -> None:
        self._min_chars = min_chars
        self._pending = ""

    def push(self, chunk: str) -> str | None:
        """Add a chunk of text; return whatever is now ready to speak."""
        if not chunk:
            return None

        self._pending += chunk

        if len(self._pending) < self._min_chars:
            return None

        last_match = 0
        for match in _SENTENCE_BOUNDARY_RE.finditer(self._pending):
            last_match = match.end()

        if not last_match:
            return None

        ready = self._pending[:last_match].strip()
        self._pending = self._pending[last_match:]
        return ready or None

    def flush(self) -> str:
        """Drain the tail. Resets, so the next turn starts clean."""
        text, self._pending = self._pending.strip(), ""
        return text


class Speech:
    """Turns streamed reply text into `SynthesizedAudioEvent`s, sentence by sentence.

    The one place that decides *when* text is worth a TTS request and *how* the
    resulting audio reaches the stream. `TTSObserver` drives it from observer
    callbacks; `CascadeConfig` owns an instance directly, because a cascade has
    to speak on its own to satisfy `RealtimeConfig` rather than relying on an
    observer the caller might not attach.
    """

    def __init__(self, config: TTSConfig[bytes], min_chars: int = 60) -> None:
        self._config = config
        self._buffer = SentenceBuffer(min_chars)
        self._saw_chunks = False

    async def on_chunk(self, event: events.ModelMessageChunk, context: Context) -> None:
        if not event.content:
            return
        self._saw_chunks = True
        if text := self._buffer.push(event.content):
            await self._emit(text, context)

    async def finish(self, content: str | None, context: Context) -> None:
        # Non-streaming configs emit no chunks at all, leaving the buffer empty
        # — fall back to the completed text so the reply is not lost silently.
        text = self._buffer.flush() if self._saw_chunks else (content or "").strip()
        self._saw_chunks = False
        await self._emit(text, context)

    async def _emit(self, text: str, context: Context) -> None:
        if not text:
            return

        if isinstance(self._config, StreamingTTSConfig):
            async for chunk in self._config.stream(text):
                if chunk:
                    await context.send(events.SynthesizedAudioEvent(chunk))
            return

        if audio := await self._config.synthesize(text):
            await context.send(events.SynthesizedAudioEvent(audio))


_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?\n]")
