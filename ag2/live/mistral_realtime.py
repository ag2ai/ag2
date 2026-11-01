# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
from collections.abc import AsyncIterator
from typing import Any

from mistralai.client import Mistral
from mistralai.client.models import (
    RealtimeSpeechError,
    RealtimeSpeechStreamingMode,
    SpeechStreamAudioDelta,
    SpeechStreamDone,
)

# Needs `websockets` (the SDK's `realtime` extra), which `ag2[mistral]` does not ship:
# this module is kept apart from `mistral.py` so the other Mistral configs work without it.
from mistralai.extra.realtime import RealtimeSpeechConnection

from .mistral import DEFAULT_MODEL, SpeechModel, _int16_chunks
from .protocols import TextStreamingTTSConfig as TextStreamingTTSConfigProtocol


class RealtimeTTSConfig(TextStreamingTTSConfigProtocol[bytes]):
    """Mistral realtime TTS: one websocket per reply, fed the reply's text as the model streams it.

    Mistral decides where to cut the audio, so speech starts before the first
    sentence is complete and the voice keeps its context across sentences.
    Yields int16 PCM, converted from Mistral's float32 PCM like the other
    Mistral configs. An `error` event from the server raises `RuntimeError`.

    Exactly one of `voice_id` (a saved voice) or `ref_audio` (a base64
    reference clip for one-off cloning) is required; an empty string counts as
    not given. `streaming_mode` ("responsive" or "long_form") and
    `expressiveness` are sent only when set.

    Needs `pip install "mistralai[realtime]"` and an api key in
    MISTRAL_API_KEY env var, or pass `api_key` / a preconfigured `client`.
    """

    def __init__(
        self,
        model: SpeechModel | str = DEFAULT_MODEL,
        *,
        voice_id: str | None = None,
        ref_audio: str | None = None,
        streaming_mode: RealtimeSpeechStreamingMode | None = None,
        expressiveness: float | None = None,
        client: Mistral | None = None,
        api_key: str | None = None,
    ) -> None:
        if bool(voice_id) == bool(ref_audio):
            raise ValueError("Mistral TTS needs exactly one of `voice_id` or `ref_audio`")

        self._model = model
        self._voice_id = voice_id
        self._ref_audio = ref_audio
        self._streaming_mode = streaming_mode
        self._expressiveness = expressiveness
        self._client = client or Mistral(api_key=api_key)

    async def stream_text(self, text: AsyncIterator[str]) -> AsyncIterator[bytes]:
        # Only the selected voice is sent, never the other one as None.
        voice = {"voice_id": self._voice_id} if self._voice_id else {"ref_audio": self._ref_audio}
        connection = await self._client.audio.realtime_speech.connect(
            self._model,
            response_format="pcm",
            streaming_mode=self._streaming_mode,
            expressiveness=self._expressiveness,
            **voice,
        )

        # The SDK swallows `CancelledError` while it waits for the next event, so a
        # cancelled reply would end like a finished one. Sending and receiving run as
        # tasks feeding one inbox instead, which keeps the cancellation on the caller.
        inbox: asyncio.Queue[Any] = asyncio.Queue()
        tasks = (
            asyncio.create_task(_send_text(connection, text, inbox)),
            asyncio.create_task(_receive_speech(connection, inbox)),
        )

        try:
            async for chunk in _int16_chunks(_speech_audio(inbox)):
                yield chunk

        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await connection.close()

    async def synthesize(self, text: str) -> bytes:
        return b"".join([chunk async for chunk in self.stream_text(_once(text))])


async def _send_text(connection: RealtimeSpeechConnection, text: AsyncIterator[str], inbox: asyncio.Queue[Any]) -> None:
    try:
        async for piece in text:
            if piece:
                await connection.send_text(piece)
        await connection.end()

    except Exception as e:
        inbox.put_nowait(e)


async def _receive_speech(connection: RealtimeSpeechConnection, inbox: asyncio.Queue[Any]) -> None:
    try:
        async for event in connection:
            inbox.put_nowait(event)

    except Exception as e:
        inbox.put_nowait(e)

    # The connection closed: nothing more is coming.
    inbox.put_nowait(None)


async def _speech_audio(inbox: asyncio.Queue[Any]) -> AsyncIterator[bytes]:
    while True:
        item = await inbox.get()

        if isinstance(item, BaseException):
            raise item

        if isinstance(item, SpeechStreamAudioDelta):
            yield base64.b64decode(item.audio_data)

        elif isinstance(item, SpeechStreamDone):
            return

        elif isinstance(item, RealtimeSpeechError):
            raise RuntimeError(f"Mistral realtime speech failed: {item.error.message}")

        elif item is None:
            raise RuntimeError("Mistral realtime speech closed before the audio finished")


async def _once(text: str) -> AsyncIterator[str]:
    yield text
