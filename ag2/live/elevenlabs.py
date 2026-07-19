# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from typing import Literal

from elevenlabs.client import AsyncElevenLabs

from ag2 import events
from ag2.annotations import Context

from .protocols import TTSConfig as TTSConfigProtocol
from .stt import STTConfig as STTConfigProtocol
from .stt import VoiceInput, voice_to_wav_buffer

DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel

ScribeModelId = Literal["scribe_v2",]

OutputFormat = Literal[
    "pcm_16000",
    "pcm_22050",
    "pcm_24000",
    "pcm_44100",
    "mp3_44100_128",
]
DEFAULT_OUTPUT_FORMAT: OutputFormat = "pcm_24000"

# Not exhaustive — any model id string is accepted. `eleven_v3` is
# synchronous-only; the flash/turbo families support streaming.

StreamingModelId = Literal[
    "eleven_multilingual_v2",
    "eleven_turbo_v2_5",
    "eleven_flash_v2_5",
]
SyncOutputModelId = Literal["eleven_v3",]


class STTConfig(STTConfigProtocol):
    """Whole-file transcription via Scribe. Upload the clip, get the text back.

    Synchronous by nature: the endpoint takes a complete recording, so there is
    nothing to emit until it answers. Unlike OpenAI's transcriber — which
    streams deltas and sends `TranscriptionChunkEvent` as they arrive — this
    only ever sends `TranscriptionCompletedEvent`.

    Needs an api key in ELEVENLABS_API_KEY env var.
    """

    def __init__(
        self,
        model_id: "ScribeModelId | str" = "scribe_v2",
        *,
        language_code: str | None = None,
        diarize: bool = False,
        client: "AsyncElevenLabs | None" = None,
    ) -> None:
        self._model_id = model_id
        self._language_code = language_code
        self._diarize = diarize
        self._client = client or AsyncElevenLabs()

    async def transcribe(self, voice: VoiceInput, context: Context) -> str:
        # `language_code` is only sent when set — the endpoint auto-detects
        # when it is absent, and an explicit None is not the same thing.
        extra = {"language_code": self._language_code} if self._language_code else {}

        response = await self._client.speech_to_text.convert(
            model_id=self._model_id,
            file=voice_to_wav_buffer(voice),
            diarize=self._diarize,
            **extra,
        )

        # The response is a union; only the single-channel shape carries `text`
        # directly. Multichannel and webhook jobs would need their own handling.
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            raise TypeError(f"Scribe returned {type(response).__name__}, which carries no transcript text")

        await context.send(events.TranscriptionCompletedEvent(text))
        return text


class TTSConfig(TTSConfigProtocol[bytes]):
    """Synchronous TTS. Waits for the full clip, then returns it.

    This is the mode required by ``eleven_v3``, which cannot stream. Implements
    the `TTSConfig` protocol and plugs into `TTSObserver` unchanged.

    If you want streaming, use StreamingTTSConfig.
    """

    def __init__(
        self,
        model_id: "SyncOutputModelId | str" = "eleven_v3",
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        output_format: "OutputFormat | str" = DEFAULT_OUTPUT_FORMAT,
        client: "AsyncElevenLabs | None" = None,
    ) -> None:
        self._model_id = model_id
        self._voice_id = voice_id
        self._output_format = output_format
        self._client = client or AsyncElevenLabs()

    async def synthesize(self, text: str) -> bytes:
        # `convert` hits the non-streaming endpoint; it still returns an async
        # iterator of byte chunks, so we drain it into a single buffer.
        audio = self._client.text_to_speech.convert(
            voice_id=self._voice_id,
            text=text,
            model_id=self._model_id,
            output_format=self._output_format,
        )
        return b"".join([chunk async for chunk in audio])


class StreamingTTSConfig(TTSConfigProtocol[bytes]):
    """Streaming TTS via the ``/stream`` endpoint.

    Yields audio chunks as they are generated for lower time-to-first-byte.
    Also implements `synthesize` (by buffering the full stream) so it satisfies
    the plain `TTSConfig` protocol. `TTSObserver` and `CascadeConfig` both
    detect `stream` and use it automatically. Not supported by ``eleven_v3``.

    Needs an api key in ELEVENLABS_API_KEY env var or can accept an api_key argument
    """

    def __init__(
        self,
        model_id: "StreamingModelId | str" = "eleven_flash_v2_5",
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        output_format: "OutputFormat | str" = DEFAULT_OUTPUT_FORMAT,
        client: "AsyncElevenLabs | None" = None,
    ) -> None:
        self._model_id = model_id
        self._voice_id = voice_id
        self._output_format = output_format
        self._client = client or AsyncElevenLabs()

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        audio = self._client.text_to_speech.stream(
            voice_id=self._voice_id,
            text=text,
            model_id=self._model_id,
            output_format=self._output_format,
        )
        async for chunk in audio:
            if chunk:
                yield chunk

    async def synthesize(self, text: str) -> bytes:
        return b"".join([chunk async for chunk in self.stream(text)])
