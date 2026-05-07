# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING

from openai import AsyncOpenAI, Omit, omit
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from openai.types.audio.speech_create_params import Voice
from openai.types.beta.realtime import session_update_event_param

from autogen.beta.context import ConversationContext
from autogen.beta.events import (
    ModelMessage,
    ModelMessageChunk,
    ModelResponse,
    RecordedAudioEvent,
    SynthesizedAudioEvent,
    TranscriptionChunkEvent,
    TranscriptionCompletedEvent,
)

from .protocols import TTSConfig as TTSConfigProtocol
from .stt import STTConfig as STTConfigProtocol
from .stt import VoiceInput

if TYPE_CHECKING:
    from openai.types.audio.speech_model import SpeechModel
    from openai.types.audio_model import AudioModel

    from autogen.beta.annotations import Context


class STTConfig(STTConfigProtocol):
    def __init__(
        self,
        model: "AudioModel | str",
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.client = client or AsyncOpenAI()

    async def transcribe(self, voice: "VoiceInput", context: "Context") -> str:
        stream = await self.client.audio.transcriptions.create(
            model=self.model,
            file=_voice_to_wav_buffer(voice),
            response_format="text",
            stream=True,
        )

        text = ""
        async for event in stream:
            if event.type == "transcript.text.delta":
                text += event.delta
                await context.send(TranscriptionChunkEvent(event.delta))

        await context.send(TranscriptionCompletedEvent(text))
        return text


class STTTranslationConfig(STTConfigProtocol):
    def __init__(
        self,
        model: "AudioModel | str",
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.client = client or AsyncOpenAI()

    async def transcribe(self, voice: "VoiceInput", context: "Context") -> str:
        result = await self.client.audio.translations.create(
            model=self.model,
            file=_voice_to_wav_buffer(voice),
            response_format="text",
        )

        await context.send(TranscriptionCompletedEvent(result))
        return result


class TTSConfig(TTSConfigProtocol[bytes]):
    def __init__(
        self,
        model: "SpeechModel | str",
        *,
        client: AsyncOpenAI | None = None,
        voice: Voice = "alloy",
        speed: float | Omit = omit,
    ) -> None:
        self._client = client or AsyncOpenAI()

        self._model = model
        self._voice = voice
        self._speed = speed

    async def synthesize(self, text: str) -> bytes:
        response = await self._client.audio.speech.create(
            model=self._model,
            voice=self._voice,
            input=text,
            speed=self._speed,
            response_format="pcm",
        )
        return await response.aread()


class OpenAIRealTimeConfig:
    """Realtime STT config backed by OpenAI's bidirectional realtime API.

    Implements the `RealtimeSTTConfig` protocol — call `session(...)` to open
    a connection that pumps captured audio into the API and emits transcription
    events on the supplied context.
    """

    def __init__(
        self,
        model: str,
        *,
        session: session_update_event_param.Session | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self._session_config = session
        self.client = client or AsyncOpenAI()

    @asynccontextmanager
    async def session(
        self,
        context: ConversationContext,
    ) -> AsyncIterator[None]:
        async with self.client.beta.realtime.connect(model=self.model) as conn:
            if self._session_config:
                await conn.session.update(session=self._session_config)

            async def _pump_audio(event: RecordedAudioEvent) -> None:
                await conn.input_audio_buffer.append(audio=base64.b64encode(event.content).decode())

            with context.stream.where(RecordedAudioEvent).sub_scope(_pump_audio):
                recv_task = asyncio.create_task(_pump_events(conn, context))

                try:
                    yield

                finally:
                    recv_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await recv_task


async def _pump_events(
    conn: AsyncRealtimeConnection,
    context: ConversationContext,
) -> None:
    text = ""
    async for event in conn:
        if event.type == "conversation.item.input_audio_transcription.delta":
            await context.send(TranscriptionChunkEvent(event.delta))
        elif event.type == "conversation.item.input_audio_transcription.completed":
            # TODO: process usage
            await context.send(TranscriptionCompletedEvent(event.transcript))
        elif event.type == "response.audio.delta":
            await context.send(SynthesizedAudioEvent(base64.b64decode(event.delta)))
        elif event.type == "response.text.delta":
            text += event.delta
            await context.send(ModelMessageChunk(event.delta))
        elif event.type == "response.done":
            # done event emits after all text and audio chunks are emitted
            # so, we can emit the final message and usage here
            # without `response.text.done` event processing
            await context.send(
                ModelResponse(
                    # text always none for audio output
                    message=ModelMessage(text) if text else None,
                    # TODO: map usage
                    # usage=event.response.usage,
                )
            )
            text = ""


def _voice_to_wav_buffer(voice: "VoiceInput") -> io.BytesIO:
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as wav_file:
        wav_file.setnchannels(voice.channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.frame_rate)
        wav_file.writeframes(voice.content)
    audio_buffer.seek(0)
    audio_buffer.name = "speech.wav"
    return audio_buffer
