# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any, Literal

from mistralai.client import Mistral
from mistralai.client.models import File

from ag2 import events
from ag2.annotations import Context

from .stt import STTConfig as STTConfigProtocol
from .stt import VoiceInput, voice_to_wav_buffer

TranscriptionModel = Literal["voxtral-mini-latest",]


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

        wav = voice_to_wav_buffer(voice)
        response = await self._client.audio.transcriptions.complete_async(
            model=self._model,
            file=File(file_name=wav.name, content=wav.read()),
            diarize=self._diarize,
            **extra,
        )

        await context.send(events.TranscriptionCompletedEvent(response.text))
        return response.text
