# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
import struct
from collections.abc import AsyncIterator
from typing import Any, Literal

from mistralai.client import Mistral
from mistralai.client.models import SpeechStreamAudioDelta

from .protocols import TTSConfig as TTSConfigProtocol

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
        return self._client.speech.complete_async(
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

        # Delta boundaries are not sample-aligned: carry the partial float32 sample over.
        pending = b""
        async for event in events:
            if not isinstance(event.data, SpeechStreamAudioDelta):
                continue

            pending += base64.b64decode(event.data.audio_data)
            usable = len(pending) - len(pending) % _FLOAT32_SIZE
            if usable:
                yield _float32_to_int16(pending[:usable])
                pending = pending[usable:]

        if pending:
            raise ValueError(f"Mistral stream ended mid-sample with {len(pending)} stray bytes")

    async def synthesize(self, text: str) -> bytes:
        return b"".join([chunk async for chunk in self.stream(text)])
