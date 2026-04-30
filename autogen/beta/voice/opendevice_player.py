# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import queue
import threading
from types import TracebackType

import numpy as np
import sounddevice as sd

from .protocols import AudioPlayer


class OpenDevicePlayer(AudioPlayer[bytes]):
    def __init__(self, stream: sd.OutputStream | None = None) -> None:
        self._stream = stream or sd.OutputStream(
            samplerate=24000,
            channels=1,
            dtype=np.int16,
        )
        self._audio_queue: queue.Queue[bytes | None] = queue.Queue()
        self._worker: threading.Thread | None = None

        self._speaker_lock = threading.Lock()

    def __enter__(self) -> "OpenDevicePlayer":
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
