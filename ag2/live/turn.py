# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Turn detection for cascade voice sessions.

Speech-to-speech providers segment the microphone stream server-side (see
``openai.InputConfig.turn_detection``). A cascade has no such server, so the
decision "the user stopped talking, transcribe now" has to be made locally —
that is what a `TurnDetector` does.

`SoundDeviceRecorder` emits fixed-size chunks with no notion of a turn; the
detector consumes those chunks and returns a complete `VoiceInput` at the
moment a turn closes.
"""

from array import array
from math import sqrt
from typing import Protocol

from .stt import VoiceInput


class TurnDetector(Protocol):
    """Segments a continuous stream of PCM chunks into user turns."""

    @property
    def speaking(self) -> bool:
        """Whether the user is mid-utterance right now.

        Read by the session to decide whether newly arriving speech should
        interrupt an in-flight reply (barge-in).
        """
        ...

    def push(self, chunk: bytes) -> VoiceInput | None:
        """Feed one chunk; return the finished turn's audio, or None."""
        ...

    def reset(self) -> None:
        """Drop any buffered audio and return to the idle state."""
        ...


class SilenceTurnDetector(TurnDetector):
    """Energy-gated turn detection: speech ends after a run of quiet chunks.

    Deliberately simple — RMS against a fixed threshold, no model. That is
    enough for a close-talking mic in a quiet room, which is the setup the
    voice examples assume. For noisy input, swap in a detector backed by a
    real VAD; `CascadeConfig` takes any `TurnDetector`.

    Audio from just before the trigger is kept (`prefix_padding`), because the
    chunk that crosses the threshold is normally already a few milliseconds
    into the first word — without it, turns start clipped.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        channels: int = 1,
        threshold: float = 500.0,
        silence: float = 0.7,
        min_speech: float = 0.2,
        prefix_padding: float = 0.3,
        max_duration: float = 30.0,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._threshold = threshold
        self._silence = silence
        self._min_speech = min_speech
        self._prefix_padding = prefix_padding
        self._max_duration = max_duration

        self._speaking = False
        self._speech: list[bytes] = []
        self._prefix: list[bytes] = []
        self._prefix_seconds = 0.0
        self._speech_seconds = 0.0
        self._silence_seconds = 0.0

    @property
    def speaking(self) -> bool:
        return self._speaking

    def push(self, chunk: bytes) -> VoiceInput | None:
        if not chunk:
            return None

        duration = self._duration(chunk)
        loud = _rms(chunk) >= self._threshold

        if not self._speaking:
            if not loud:
                self._remember_prefix(chunk, duration)
                return None

            # Onset: the padding buffer becomes the head of the turn.
            self._speaking = True
            self._speech = [*self._prefix, chunk]
            self._speech_seconds = self._prefix_seconds + duration
            self._silence_seconds = 0.0
            self._drop_prefix()
            return None

        self._speech.append(chunk)
        self._speech_seconds += duration
        self._silence_seconds = 0.0 if loud else self._silence_seconds + duration

        if self._silence_seconds >= self._silence or self._speech_seconds >= self._max_duration:
            return self._close()

        return None

    def reset(self) -> None:
        self._speaking = False
        self._speech = []
        self._speech_seconds = 0.0
        self._silence_seconds = 0.0
        self._drop_prefix()

    def _close(self) -> VoiceInput | None:
        audio = b"".join(self._speech)
        # Speech time excludes the trailing silence that closed the turn; a
        # cough or a door slam should not cost an STT round-trip.
        speech_only = self._speech_seconds - self._silence_seconds
        self.reset()

        if speech_only < self._min_speech:
            return None
        return VoiceInput(audio, self._sample_rate, self._channels)

    def _remember_prefix(self, chunk: bytes, duration: float) -> None:
        self._prefix.append(chunk)
        self._prefix_seconds += duration
        while self._prefix and self._prefix_seconds > self._prefix_padding:
            self._prefix_seconds -= self._duration(self._prefix.pop(0))

    def _drop_prefix(self) -> None:
        self._prefix = []
        self._prefix_seconds = 0.0

    def _duration(self, chunk: bytes) -> float:
        # 16-bit samples, hence the //2.
        return len(chunk) / 2 / self._channels / self._sample_rate


def _rms(chunk: bytes) -> float:
    """Root-mean-square amplitude of a 16-bit PCM chunk."""
    # `array` keeps this stdlib-only: numpy is an optional extra pulled in by
    # sounddevice, and turn detection must not depend on it.
    samples = array("h")
    # An odd trailing byte cannot form a sample; drop it rather than raise.
    samples.frombytes(chunk[: len(chunk) - (len(chunk) % 2)])
    if not samples:
        return 0.0
    return sqrt(sum(s * s for s in samples) / len(samples))
