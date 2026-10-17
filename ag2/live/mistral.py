# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from typing import Any, Literal

from mistralai.client import Mistral
from mistralai.client.models import AudioFormat, File, RealtimeTranscriptionError, TranscriptionStreamDone
from mistralai.client.models import TranscriptionStreamTextDelta as TextDelta

from ag2 import events
from ag2.annotations import Context

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
