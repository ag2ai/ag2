# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import TypeVar

from autogen.beta import events
from autogen.beta.observers import CompositeObserver, observer

from .protocols import AudioPlayer, TTSConfig

T = TypeVar("T")
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?\n]")


def TTSObserver(  # noqa: N802
    player: AudioPlayer[T],
    config: TTSConfig[T],
) -> CompositeObserver:
    tts = _ChunkToSpeech(player, config=config)

    @observer(events.ModelMessageChunk)
    async def on_model_message_chunk(event: events.ModelMessageChunk) -> None:
        await tts.on_chunk(event)

    @observer(events.ModelMessage)
    async def on_model_message(event: events.ModelMessage) -> None:
        await tts.on_complete()

    return CompositeObserver(on_model_message_chunk, on_model_message)


class _ChunkToSpeech:
    def __init__(
        self,
        player: AudioPlayer[T],
        *,
        config: TTSConfig[T],
        min_chars: int = 60,
    ) -> None:
        self._player = player
        self._config = config
        self._min_chars = min_chars
        self._pending_text = ""

    async def on_chunk(self, event: events.ModelMessageChunk) -> None:
        chunk = event.content
        if not chunk:
            return

        self._pending_text += chunk

        if text := self._should_emit(self._pending_text):
            await self._enqueue_pending_text(text)

    async def on_complete(self) -> None:
        await self._enqueue_pending_text(self._pending_text.strip())
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

    async def _enqueue_pending_text(self, text: str) -> None:
        if not text:
            return

        pcm = await self._config.synthesize(text)

        if pcm:
            await self._player.play(pcm)
