# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import queue
import threading
from types import TracebackType

import numpy as np
import sounddevice as sd

from ag2 import MemoryStream
from ag2.context import ConversationContext, Stream, SubId
from ag2.events import RecordedAudioEvent, SynthesizedAudioEvent

from .protocols import AudioPlayer
from .stt import VoiceInput

# Bounded buffer between the audio thread and the asyncio bus. Drop-oldest
# when full — stale mic bytes are useless for STT, so we keep the most
# recent ~1 second (10 × 100ms blocks) under sustained load.
_AUDIO_BUFFER_CHUNKS = 10


class Recorder:
    def __init__(
        self,
        *,
        context: ConversationContext | None = None,
        sample_rate: int = 24000,
        channels: int = 1,
        block_size: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        # Default 100ms blocks — small enough for low-latency realtime, large
        # enough to keep callback overhead reasonable.
        self.block_size = block_size or int(sample_rate * 0.1)

        self.context = context or ConversationContext(stream=MemoryStream())
        self._loop: asyncio.AbstractEventLoop | None = None
        self._input: sd.InputStream | None = None
        self._queue: asyncio.Queue[bytes] | None = None
        self._drain_task: asyncio.Task[None] | None = None

    @property
    def stream(self) -> Stream:
        return self.context.stream

    def record(self, duration: float) -> VoiceInput:
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
        )
        sd.wait()

        return VoiceInput(
            # sounddevice returns normalized float32 in [-1.0, 1.0];
            # VoiceInput expects 16-bit PCM bytes, so scale to int16 range.
            (np.clip(recording.squeeze(), -1.0, 1.0) * 32767).astype(np.int16).tobytes(),
            self.sample_rate,
            self.channels,
        )

    async def __aenter__(self) -> "Recorder":
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=_AUDIO_BUFFER_CHUNKS)
        self._drain_task = self._loop.create_task(self._drain_to_bus())
        self._input = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.block_size,
            callback=self._callback,
        )
        self._input.start()
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_value: object | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._input is not None:
            self._input.stop()
            self._input.close()
            self._input = None
        if self._drain_task is not None:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
            self._drain_task = None
        self._queue = None
        self._loop = None

    def _callback(self, indata: np.ndarray, _frames: int, _time, _status) -> None:
        # Runs on sounddevice's audio thread; hand off to the loop thread.
        # asyncio.Queue is NOT thread-safe, so we MUST go via call_soon_threadsafe.
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._enqueue, indata.copy().tobytes())

    def _enqueue(self, chunk: bytes) -> None:
        # Runs on the loop thread. Drop-oldest if the buffer is full so we
        # always carry the freshest audio into the bus.
        if self._queue is None:
            return
        if self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
        self._queue.put_nowait(chunk)

    async def _drain_to_bus(self) -> None:
        assert self._queue is not None
        while True:
            chunk = await self._queue.get()
            await self.context.send(RecordedAudioEvent(chunk))


class Player(AudioPlayer[bytes]):
    def __init__(
        self,
        *,
        context: ConversationContext | None = None,
        output_stream: sd.OutputStream | None = None,
    ) -> None:
        self.context = context or ConversationContext(stream=MemoryStream())
        self._output_stream = output_stream
        self._audio_queue: queue.Queue[bytes | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._sub_id: SubId | None = None

        self._speaker_lock = threading.Lock()

    @property
    def stream(self) -> Stream:
        return self.context.stream

    async def __aenter__(self) -> "Player":
        if self._output_stream is None:
            self._output_stream = sd.OutputStream(
                samplerate=24000,
                channels=1,
                dtype=np.int16,
            )
        self._output_stream.__enter__()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()
        self._sub_id = self.context.stream.where(SynthesizedAudioEvent).subscribe(self._on_audio)
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_value: object | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._sub_id is not None:
            self.context.stream.unsubscribe(self._sub_id)
            self._sub_id = None
        self.close()
        if self._output_stream is not None:
            self._output_stream.__exit__(exc_type, exc_value, traceback)

    async def play(self, content: bytes) -> None:
        if not content:
            return
        self._audio_queue.put(content)

    def join(self) -> None:
        if self._worker is None:
            return

        while True:
            with self._speaker_lock:
                if self._audio_queue.qsize() == 0:
                    break

    def close(self) -> None:
        if self._worker is None:
            return
        self._audio_queue.put(None)
        self._worker.join()
        self._worker = None

    async def _on_audio(self, event: SynthesizedAudioEvent) -> None:
        await self.play(event.content)

    def _run_worker(self) -> None:
        while True:
            pcm = self._audio_queue.get()
            if pcm is None:
                return

            np_bytes = np.frombuffer(pcm, dtype=np.int16).reshape(-1, 1)

            with self._speaker_lock:
                assert self._output_stream is not None
                self._output_stream.write(np_bytes)
