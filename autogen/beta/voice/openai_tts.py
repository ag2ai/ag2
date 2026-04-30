# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from openai import AsyncOpenAI
from openai.types.audio.speech_model import SpeechModel

from .protocols import TTSConfig


class OpenAITTSConfig(TTSConfig[bytes]):
    def __init__(
        self,
        model: SpeechModel | str,
        *,
        client: AsyncOpenAI | None = None,
        voice: str = "alloy",
    ) -> None:
        self._client = client or AsyncOpenAI()
        self._model = model
        self._voice = voice

    async def synthesize(self, text: str) -> bytes:
        response = await self._client.audio.speech.create(
            model=self._model,
            voice=self._voice,
            input=text,
            response_format="pcm",
        )
        return await response.aread()
