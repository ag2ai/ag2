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


class _ChunkToSpeech:
    def __init__(
        self,
        *,
        config: TTSConfig[bytes],
        min_chars: int = 60,
    ) -> None:
        self._config = config
        self._min_chars = min_chars
        self._pending_text = ""

    async def on_chunk(self, event: events.ModelMessageChunk, context: Context) -> None:
        chunk = event.content
        if not chunk:
            return

        self._pending_text += chunk

        if text := self._should_emit(self._pending_text):
            await self._emit(text, context)

    async def on_complete(self, context: Context) -> None:
        await self._emit(self._pending_text.strip(), context)
        self._pending_text = ""

    def _should_emit(self, text: str) -> str | None:
        if len(text) < self._min_chars:
            return None

        last_match = 0
        for match in _SENTENCE_BOUNDARY_RE.finditer(text):
            last_match = match.end()

        if last_match:
            ready = text[:last_match].strip()
            self._pending_text = text[last_match:]
            return ready

        return None

    async def _emit(self, text: str, context: Context) -> None:
        if not text:
            return

        pcm = await self._config.synthesize(text)

        if pcm:
            await context.send(events.SynthesizedAudioEvent(pcm))


_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?\n]")
