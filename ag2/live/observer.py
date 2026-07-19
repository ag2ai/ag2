# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import re

from ag2 import events
from ag2.annotations import Context
from ag2.observers import CompositeObserver, observer

from .protocols import TTSConfig


def TTSObserver(config: TTSConfig[bytes]) -> CompositeObserver:  # noqa: N802
    tts = _ChunkToSpeech(config=config)

    @observer(events.ModelMessageChunk)
    async def on_model_message_chunk(event: events.ModelMessageChunk, context: Context) -> None:
        await tts.on_chunk(event, context)

    @observer(events.ModelMessage)
    async def on_model_message(event: events.ModelMessage, context: Context) -> None:
        await tts.on_complete(context)

    return CompositeObserver(on_model_message_chunk, on_model_message)


class SentenceBuffer:
    """Accumulates streamed text and releases it at sentence boundaries.

    Pure text bookkeeping — no events, no synthesis — so both the buffering
    (`TTSObserver`) and streaming (`elevenlabs.StreamingTTSObserver`) paths can
    share one notion of "enough text to speak yet?".
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


class _ChunkToSpeech:
    def __init__(
        self,
        *,
        config: TTSConfig[bytes],
        min_chars: int = 60,
    ) -> None:
        self._config = config
        self._buffer = SentenceBuffer(min_chars)

    async def on_chunk(self, event: events.ModelMessageChunk, context: Context) -> None:
        if text := self._buffer.push(event.content or ""):
            await self._emit(text, context)

    async def on_complete(self, context: Context) -> None:
        await self._emit(self._buffer.flush(), context)

    async def _emit(self, text: str, context: Context) -> None:
        if not text:
            return

        pcm = await self._config.synthesize(text)

        if pcm:
            await context.send(events.SynthesizedAudioEvent(pcm))


_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?\n]")
