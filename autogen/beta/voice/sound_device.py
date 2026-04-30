# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import queue
import threading
from types import TracebackType

import numpy as np
import sounddevice as sd

from .protocols import AudioPlayer
from .stt import VoiceInput


class Recorder:
    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        channels: int = 1,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels

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


class Player(AudioPlayer[bytes]):
    def __init__(self, stream: sd.OutputStream | None = None) -> None:
        self._stream = stream or sd.OutputStream(
            samplerate=24000,
            channels=1,
            dtype=np.int16,
        )
        self._audio_queue: queue.Queue[bytes | None] = queue.Queue()
        self._worker: threading.Thread | None = None

        self._speaker_lock = threading.Lock()

    def __enter__(self) -> "Player":
        self._stream.__enter__()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_value: object | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
        self._stream.__exit__(exc_type, exc_value, traceback)

    async def play(self, content: bytes) -> None:
        if not content:
            return
        self._audio_queue.put(content)

    def stop(self) -> None:
        if not self._worker:
            self._worker = None

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

    def _run_worker(self) -> None:
        while True:
            pcm = self._audio_queue.get()
            if pcm is None:
                return

            np_bytes = np.frombuffer(pcm, dtype=np.int16).reshape(-1, 1)

            with self._speaker_lock:
                self._stream.write(np_bytes)
