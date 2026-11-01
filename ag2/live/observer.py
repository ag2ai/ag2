# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import re
from collections.abc import AsyncIterator
from contextlib import suppress

from ag2 import events
from ag2.annotations import Context
from ag2.observers import CompositeObserver, observer

from .protocols import StreamingTTSConfig, TTSConfig, TextStreamingTTSConfig


def TTSObserver(config: TTSConfig[bytes], *, min_chars: int = 60) -> CompositeObserver:  # noqa: N802
    """Speak the model's reply as it streams, a sentence at a time.

    Configs that also implement `StreamingTTSConfig` are streamed chunk by
    chunk, so playback starts at time-to-first-byte rather than after the last
    sample of the sentence; plain configs are synthesized whole.
    `TextStreamingTTSConfig` configs get the whole reply as it streams instead,
    and `min_chars` does not apply to them.

    `min_chars` is the smallest amount of text worth a request; below it the
    buffer keeps accumulating even past a sentence boundary.

    The observer only sees a reply end through its `ModelMessage`. A reply that
    fails mid-stream never sends one, so its unspoken text, or its open
    text-streaming request, carries over into the next reply.
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

    A `TextStreamingTTSConfig` gets no sentences: the reply's text goes to one
    `stream_text` call as it arrives, and `finish` waits until all of its audio
    has been sent.
    """

    def __init__(self, config: TTSConfig[bytes], min_chars: int = 60) -> None:
        self._config = config
        self._buffer = SentenceBuffer(min_chars)
        self._saw_chunks = False
        # The reply being spoken through `stream_text`: its text inbox, ended by
        # None, and the task forwarding its audio.
        self._text: asyncio.Queue[str | None] | None = None
        self._speaking: asyncio.Task[None] | None = None

    async def on_chunk(self, event: events.ModelMessageChunk, context: Context) -> None:
        if not event.content:
            return
        self._saw_chunks = True

        if isinstance(self._config, TextStreamingTTSConfig):
            self._push(event.content, context)
            return

        if text := self._buffer.push(event.content):
            await self._emit(text, context)

    async def finish(self, content: str | None, context: Context) -> None:
        if isinstance(self._config, TextStreamingTTSConfig):
            # Non-streaming configs emit no chunks: speak the completed text instead.
            if not self._saw_chunks:
                self._push((content or "").strip(), context)
            self._saw_chunks = False
            await self._end_reply()
            return

        # Non-streaming configs emit no chunks at all, leaving the buffer empty
        # — fall back to the completed text so the reply is not lost silently.
        text = self._buffer.flush() if self._saw_chunks else (content or "").strip()
        self._saw_chunks = False
        await self._emit(text, context)

    async def aclose(self) -> None:
        """Abandon the reply being spoken, closing its text-streaming request."""
        speaking, self._speaking, self._text = self._speaking, None, None
        if speaking is None:
            return

        speaking.cancel()
        with suppress(asyncio.CancelledError):
            await speaking

    def _push(self, text: str, context: Context) -> None:
        # A request that already failed keeps taking (and dropping) text, so its
        # error surfaces at `finish` instead of the reply restarting mid-sentence.
        if self._text is None:
            # Leading whitespace is not worth opening a request for.
            if not text.strip():
                return
            assert isinstance(self._config, TextStreamingTTSConfig)
            self._text = asyncio.Queue()
            self._speaking = asyncio.create_task(_speak_text(self._config, self._text, context))

        self._text.put_nowait(text)

    async def _end_reply(self) -> None:
        text, speaking = self._text, self._speaking
        self._text, self._speaking = None, None
        if text is None or speaking is None:
            return

        text.put_nowait(None)
        await speaking

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


async def _speak_text(
    config: TextStreamingTTSConfig[bytes], text: "asyncio.Queue[str | None]", context: Context
) -> None:
    async for chunk in config.stream_text(_drain(text)):
        if chunk:
            await context.send(events.SynthesizedAudioEvent(chunk))


async def _drain(text: "asyncio.Queue[str | None]") -> AsyncIterator[str]:
    while (piece := await text.get()) is not None:
        yield piece


_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?\n]")
