# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import struct
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from typing import Any, Literal

from mistralai.client import Mistral
from mistralai.client.models import (
    AudioFormat,
    File,
    RealtimeTranscriptionError,
    SpeechStreamAudioDelta,
    TranscriptionStreamDone,
)
from mistralai.client.models import TranscriptionStreamTextDelta as TextDelta

from ag2 import events
from ag2.annotations import Context

from .protocols import TTSConfig as TTSConfigProtocol
from .stt import STTConfig as STTConfigProtocol
from .stt import VoiceInput, voice_to_wav_buffer

TranscriptionModel = Literal["voxtral-mini-latest",]
RealtimeModel = Literal["voxtral-mini-transcribe-realtime-2602",]

# Duration of the PCM slice sent per websocket frame, at any sample rate.
CHUNK_MS = 100


class STTConfig(STTConfigProtocol):
    """Whole-file transcription via Voxtral. Upload the clip, get the text back.

    Synchronous by nature: the endpoint takes a complete recording, so only
    `TranscriptionCompletedEvent` is sent.

    Needs an api key in MISTRAL_API_KEY env var, or pass `api_key` / a preconfigured `client`.
    """

    def __init__(
        self,
        model: "TranscriptionModel | str" = "voxtral-mini-latest",
        *,
        language: str | None = None,
        context_bias: Sequence[str] = (),
        diarize: bool = False,
        client: "Mistral | None" = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._language = language
        self._context_bias = list(context_bias)
        self._diarize = diarize
        self._client = client or (Mistral(api_key=api_key) if api_key else Mistral())

    async def transcribe(self, voice: VoiceInput, context: Context) -> str:
        # `language` and `context_bias` are only sent when set — the endpoint
        # auto-detects the language, and an explicit None is not the same thing.
        extra: dict[str, Any] = {}
        if self._language:
            extra["language"] = self._language
        if self._context_bias:
            extra["context_bias"] = self._context_bias
        # Without streaming, the API rejects `diarize` unless segment timestamps are requested.
        if self._diarize:
            extra["timestamp_granularities"] = ["segment"]

        wav = voice_to_wav_buffer(voice)
        response = await self._client.audio.transcriptions.complete_async(
            model=self._model,
            file=File(file_name=wav.name, content=wav.read()),
            diarize=self._diarize,
            **extra,
        )

        await context.send(events.TranscriptionCompletedEvent(response.text))
        return response.text


class RealtimeSTTConfig(STTConfigProtocol):
    """Voxtral realtime transcription over a websocket: text arrives while the clip is still being read.

    One finished clip in, the full text out. A `TranscriptionChunkEvent` is sent per delta,
    then a closing `TranscriptionCompletedEvent` carrying the model's final transcript.
    Raises `RuntimeError` if the stream closes before that transcript arrives. Cancelling
    `transcribe` cancels the transcription and sends no `TranscriptionCompletedEvent`.

    The clip must be mono. Needs an api key in MISTRAL_API_KEY env var, or pass
    `api_key` / a preconfigured `client`.
    """

    def __init__(
        self,
        model: "RealtimeModel | str" = "voxtral-mini-transcribe-realtime-2602",
        *,
        target_streaming_delay_ms: int | None = None,
        client: "Mistral | None" = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._delay_ms = target_streaming_delay_ms
        self._client = client or (Mistral(api_key=api_key) if api_key else Mistral())

    async def transcribe(self, voice: VoiceInput, context: Context) -> str:
        if voice.channels != 1:
            raise ValueError(f"Mistral realtime transcription takes mono audio, got {voice.channels} channels")

        # The SDK swallows `CancelledError` while it waits for the next event, so a cancelled
        # stream ends like a finished one; `shield` keeps the cancellation on this task.
        pump = asyncio.create_task(_pump_transcript(self._client, voice, context, self._model, self._delay_ms))

        try:
            text = await asyncio.shield(pump)

        finally:
            pump.cancel()
            with suppress(asyncio.CancelledError):
                await pump

        # No final transcript means the connection dropped: the deltas alone are a partial sentence.
        if text is None:
            raise RuntimeError("Mistral realtime stream closed before the final transcript")

        await context.send(events.TranscriptionCompletedEvent(text))
        return text


async def _pump_transcript(
    client: Mistral,
    voice: VoiceInput,
    context: Context,
    model: str,
    target_streaming_delay_ms: int | None,
) -> str | None:
    async for event in client.audio.realtime.transcribe_stream(
        _frames(voice),
        model=model,
        audio_format=AudioFormat(encoding="pcm_s16le", sample_rate=voice.frame_rate),
        target_streaming_delay_ms=target_streaming_delay_ms,
    ):
        if isinstance(event, TextDelta):
            await context.send(events.TranscriptionChunkEvent(event.text))

        elif isinstance(event, TranscriptionStreamDone):
            return event.text

        elif isinstance(event, RealtimeTranscriptionError):
            raise RuntimeError(f"Mistral realtime transcription failed: {event.error.message}")

    return None


async def _frames(voice: VoiceInput) -> AsyncIterator[bytes]:
    size = voice.frame_rate * 2 * CHUNK_MS // 1000
    for start in range(0, len(voice.content), size):
        yield voice.content[start : start + size]


# Not exhaustive — any model id string is accepted. Only the model named in
# Mistral's speech docs is listed.
SpeechModel = Literal["voxtral-mini-tts-2603",]

DEFAULT_MODEL: SpeechModel = "voxtral-mini-tts-2603"

_FLOAT32_SIZE = 4
_INT16_MAX = 32767


def _float32_to_int16(pcm: bytes) -> bytes:
    """Convert little-endian float32 PCM to little-endian int16 PCM."""
    if len(pcm) % _FLOAT32_SIZE:
        raise ValueError(f"float32 PCM must be a multiple of {_FLOAT32_SIZE} bytes, got {len(pcm)}")

    count = len(pcm) // _FLOAT32_SIZE
    samples = struct.unpack(f"<{count}f", pcm)
    return struct.pack(f"<{count}h", *(round(max(-1.0, min(1.0, s)) * _INT16_MAX) for s in samples))


class TTSConfig(TTSConfigProtocol[bytes]):
    """Synchronous TTS via Mistral's speech API. Waits for the full clip, then returns it.

    Mistral returns float32 PCM; it is converted to the int16 PCM that
    `SoundDevicePlayer` plays. Exactly one of `voice_id` (a saved voice) or
    `ref_audio` (a base64 reference clip for one-off cloning) is required; an
    empty string counts as not given.

    If you want streaming, use StreamingTTSConfig.

    Needs an api key in MISTRAL_API_KEY env var, or pass `api_key` / a preconfigured `client`.
    """

    def __init__(
        self,
        model: SpeechModel | str = DEFAULT_MODEL,
        *,
        voice_id: str | None = None,
        ref_audio: str | None = None,
        client: Mistral | None = None,
        api_key: str | None = None,
    ) -> None:
        if bool(voice_id) == bool(ref_audio):
            raise ValueError("Mistral TTS needs exactly one of `voice_id` or `ref_audio`")

        self._model = model
        self._voice_id = voice_id
        self._ref_audio = ref_audio
        self._client = client or Mistral(api_key=api_key)

    def _request(self, text: str, *, stream: bool) -> Any:
        # Only the selected voice is sent, never the other one as None.
        voice = {"voice_id": self._voice_id} if self._voice_id else {"ref_audio": self._ref_audio}
        return self._client.audio.speech.complete_async(
            input=text,
            model=self._model,
            response_format="pcm",
            stream=stream,
            **voice,
        )

    async def synthesize(self, text: str) -> bytes:
        response = await self._request(text, stream=False)
        return _float32_to_int16(base64.b64decode(response.audio_data))


class StreamingTTSConfig(TTSConfig):
    """Streaming TTS via Mistral's speech API.

    Yields int16 PCM chunks as they are generated for lower time-to-first-byte.
    Also implements `synthesize` (by buffering the full stream) so it satisfies
    the plain `TTSConfig` protocol. `TTSObserver` and `CascadeConfig` both
    detect `stream` and use it automatically.
    """

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        events = await self._request(text, stream=True)
        async for chunk in _int16_chunks(_audio_deltas(events)):
            yield chunk

    async def synthesize(self, text: str) -> bytes:
        return b"".join([chunk async for chunk in self.stream(text)])


async def _audio_deltas(events: AsyncIterator[Any]) -> AsyncIterator[bytes]:
    async for event in events:
        if isinstance(event.data, SpeechStreamAudioDelta):
            yield base64.b64decode(event.data.audio_data)


async def _int16_chunks(float32: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Convert a stream of float32 PCM deltas to int16 PCM chunks."""
    # Delta boundaries are not sample-aligned: carry the partial float32 sample over.
    pending = b""
    async for delta in float32:
        pending += delta
        usable = len(pending) - len(pending) % _FLOAT32_SIZE
        if usable:
            yield _float32_to_int16(pending[:usable])
            pending = pending[usable:]

    if pending:
        raise ValueError(f"Mistral stream ended mid-sample with {len(pending)} stray bytes")
