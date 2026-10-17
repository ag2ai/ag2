# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import math
import struct
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sounddevice")

from dirty_equals import IsInstance

from ag2.context import ConversationContext
from ag2.events import (
    AudioPlaybackCompletedEvent,
    AudioPlaybackStartedEvent,
    BaseEvent,
    SynthesizedAudioEvent,
)
from ag2.live import SoundDevicePlayer
from ag2.stream import MemoryStream

SAMPLE_RATE = 24000


def pcm(seconds: float, amplitude: int) -> bytes:
    n = int(SAMPLE_RATE * seconds)
    return struct.pack(f"<{n}h", *[int(amplitude * math.sin(i * 0.1)) for i in range(n)])


class TestPlayerFlush:
    """`SoundDevicePlayer.clear` is what makes barge-in audible: stopping
    synthesis is not enough while queued audio is still on its way out."""

    def test_drops_queued_audio(self) -> None:
        player = SoundDevicePlayer()
        player._audio_queue.put(b"a")
        player._audio_queue.put(b"b")

        player.clear()

        assert player._audio_queue.qsize() == 0

    def test_keeps_the_stop_sentinel(self) -> None:
        """A flush racing `close` must not swallow the worker's stop signal —
        nor leave audio queued behind it."""
        player = SoundDevicePlayer()
        player._audio_queue.put(b"a")
        player._audio_queue.put(None)
        player._audio_queue.put(b"b")

        player.clear()

        assert player._audio_queue.qsize() == 1
        assert player._audio_queue.get() is None


class FakeOutputStream:
    """Stands in for `sd.OutputStream` so playback needs no audio device."""

    def __init__(self) -> None:
        self.writes: list[Any] = []

    def __enter__(self) -> "FakeOutputStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        pass

    def write(self, data: Any) -> None:
        self.writes.append(data)


@pytest.mark.asyncio
class TestPlaybackEvents:
    """The player is the only component that knows when sound is actually in
    the room — a half-duplex session gates its microphone on these."""

    async def test_reports_the_start_and_end_of_a_reply(self) -> None:
        context = ConversationContext(stream=MemoryStream())
        edges: list[BaseEvent] = []
        context.stream.where(AudioPlaybackStartedEvent | AudioPlaybackCompletedEvent).subscribe(
            lambda e: edges.append(e),  # type: ignore[arg-type,return-value]
        )

        async with SoundDevicePlayer(context=context, output_stream=FakeOutputStream()):  # type: ignore[arg-type]
            await context.send(SynthesizedAudioEvent(pcm(0.05, 6000)))
            await asyncio.sleep(0.5)  # outlasts the settle window

        assert edges == [IsInstance(AudioPlaybackStartedEvent), IsInstance(AudioPlaybackCompletedEvent)]

    async def test_a_gap_between_streamed_chunks_does_not_end_playback(self) -> None:
        """Streaming TTS arrives in bursts and the speaker outruns the network;
        an empty queue mid-reply must not read as silence."""
        context = ConversationContext(stream=MemoryStream())
        edges: list[BaseEvent] = []
        context.stream.where(AudioPlaybackStartedEvent | AudioPlaybackCompletedEvent).subscribe(
            lambda e: edges.append(e),  # type: ignore[arg-type,return-value]
        )

        async with SoundDevicePlayer(context=context, output_stream=FakeOutputStream()):  # type: ignore[arg-type]
            for _ in range(3):
                await context.send(SynthesizedAudioEvent(pcm(0.05, 6000)))
                await asyncio.sleep(0.1)  # shorter than the settle window
            await asyncio.sleep(0.05)

        assert edges == [IsInstance(AudioPlaybackStartedEvent)]
