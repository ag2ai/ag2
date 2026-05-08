# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
import wave
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from fast_depends.library.serializer import SerializerProto
from openai import AsyncOpenAI, Omit, omit
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openai.types.audio.speech_create_params import Voice
from openai.types.realtime import (
    AudioTranscriptionParam,
    RealtimeAudioConfigInputParam,
    RealtimeAudioConfigOutputParam,
    RealtimeAudioConfigParam,
    RealtimeAudioFormatsParam,
    RealtimeAudioInputTurnDetectionParam,
    RealtimeFunctionToolParam,
    RealtimeSessionCreateRequestParam,
    RealtimeToolChoiceConfigParam,
    RealtimeTracingConfigParam,
)
from openai.types.realtime.realtime_audio_config_input_param import NoiseReduction

from autogen.beta.context import ConversationContext
from autogen.beta.events import (
    DataInput,
    ModelMessage,
    ModelMessageChunk,
    ModelResponse,
    RecordedAudioEvent,
    SynthesizedAudioEvent,
    TextInput,
    ToolCallEvent,
    ToolResultEvent,
    TranscriptionChunkEvent,
    TranscriptionCompletedEvent,
)
from autogen.beta.tools.final import FunctionToolSchema
from autogen.beta.tools.schemas import ToolSchema

from .protocols import TTSConfig as TTSConfigProtocol
from .realtime import RealtimeSTTConfig
from .stt import STTConfig as STTConfigProtocol
from .stt import VoiceInput

if TYPE_CHECKING:
    from openai.types.audio.speech_model import SpeechModel
    from openai.types.audio_model import AudioModel

    from autogen.beta.annotations import Context


RealtimeVoice = Literal[
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
]
ModelName = Literal[
    "gpt-realtime",
    "gpt-realtime-1.5",
    "gpt-realtime-2",
    "gpt-realtime-2025-08-28",
    "gpt-4o-realtime-preview",
    "gpt-4o-realtime-preview-2024-10-01",
    "gpt-4o-realtime-preview-2024-12-17",
    "gpt-4o-realtime-preview-2025-06-03",
    "gpt-4o-mini-realtime-preview",
    "gpt-4o-mini-realtime-preview-2024-12-17",
    "gpt-realtime-mini",
    "gpt-realtime-mini-2025-10-06",
    "gpt-realtime-mini-2025-12-15",
    "gpt-audio-1.5",
    "gpt-audio-mini",
    "gpt-audio-mini-2025-10-06",
    "gpt-audio-mini-2025-12-15",
]


@dataclass(slots=True)
class AudioConfig:
    """Output-side audio knobs for the realtime session.

    Mirrors ``openai.types.realtime.RealtimeAudioConfigOutputParam``.
    """

    voice: RealtimeVoice | str = "alloy"
    format: RealtimeAudioFormatsParam = field(
        default_factory=lambda: {"type": "audio/pcm", "rate": 24000},
    )
    speed: float = 1.0


@dataclass(slots=True)
class InputConfig:
    """Input-side audio knobs for the realtime session.

    Mirrors ``openai.types.realtime.RealtimeAudioConfigInputParam``, with the
    extra ``transcribe_text`` flag that opts the session into the ``text``
    output modality.
    """

    format: RealtimeAudioFormatsParam = field(
        default_factory=lambda: {"type": "audio/pcm", "rate": 24000},
    )
    transcribe_text: bool = False
    transcription: AudioTranscriptionParam | None = None
    noise_reduction: NoiseReduction | None = None
    turn_detection: RealtimeAudioInputTurnDetectionParam | None = field(
        default_factory=lambda: {
            "type": "semantic_vad",
            "create_response": True,
            "interrupt_response": True,
        }
    )


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


class OpenAIRealTimeConfig(RealtimeSTTConfig):
    """Realtime STT config backed by OpenAI's bidirectional realtime API.

    Implements the `RealtimeSTTConfig` protocol — call `session(...)` to open
    a connection that pumps captured audio into the API and emits transcription
    events on the supplied context.
    """

    def __init__(
        self,
        model: "ModelName | str",
        *,
        audio: AudioConfig | None = None,
        input: InputConfig | None = None,
        max_output_tokens: int | Literal["inf"] | None = None,
        tool_choice: RealtimeToolChoiceConfigParam | None = None,
        tracing: RealtimeTracingConfigParam | None = None,
        session: RealtimeSessionCreateRequestParam | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model

        self._session: RealtimeSessionCreateRequestParam = {"type": "realtime"}
        if max_output_tokens is not None:
            self._session["max_output_tokens"] = max_output_tokens
        if tool_choice is not None:
            self._session["tool_choice"] = tool_choice
        if tracing is not None:
            self._session["tracing"] = tracing

        modalities: list[Literal["text", "audio"]] = []
        audio_config: RealtimeAudioConfigParam = {}
        if audio is not None:
            modalities.append("audio")
            audio_config["output"] = RealtimeAudioConfigOutputParam(
                voice=audio.voice,
                format=audio.format,
                speed=audio.speed,
            )
        if input is not None:
            input_param: RealtimeAudioConfigInputParam = {
                "format": input.format,
                "turn_detection": input.turn_detection,
            }
            if input.transcribe_text:
                modalities.append("text")
            if input.transcription is not None:
                input_param["transcription"] = input.transcription
            if input.noise_reduction is not None:
                input_param["noise_reduction"] = input.noise_reduction
            audio_config["input"] = input_param

        if audio_config:
            self._session["audio"] = audio_config
        if modalities:
            self._session["output_modalities"] = list(dict.fromkeys(modalities))

        self._session_overrides: RealtimeSessionCreateRequestParam = session or {"type": "realtime"}

        self.client = client or AsyncOpenAI()

    def _build_session(
        self,
        *,
        instructions: Iterable[str] = (),
        tools: Iterable[ToolSchema] = (),
    ) -> RealtimeSessionCreateRequestParam:
        overlay: RealtimeSessionCreateRequestParam = {"type": "realtime"}
        if instructions:
            overlay["instructions"] = "\n".join(instructions)
        if tools:
            overlay["tools"] = [_tool_schema_to_session_tool(t) for t in tools]
        return self._session | overlay | self._session_overrides

    @asynccontextmanager
    async def session(
        self,
        context: ConversationContext,
        *,
        instructions: Iterable[str] = (),
        tools: Iterable[ToolSchema] = (),
        serializer: SerializerProto,
    ) -> AsyncIterator[None]:
        final_session = self._build_session(instructions=instructions, tools=tools)

        async with self.client.realtime.connect(model=self.model) as conn:
            await conn.session.update(session=final_session)

            async def _pump_audio(event: RecordedAudioEvent) -> None:
                await conn.input_audio_buffer.append(audio=base64.b64encode(event.content).decode())

            async def _forward_tool_result(event: ToolResultEvent) -> None:
                await _send_tool_result(conn, event, serializer)

            with (
                context.stream.where(RecordedAudioEvent).sub_scope(_pump_audio),
                context.stream.where(ToolResultEvent).sub_scope(_forward_tool_result),
            ):
                recv_task = asyncio.create_task(_pump_events(conn, context))

                try:
                    yield

                finally:
                    recv_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await recv_task


def _tool_schema_to_session_tool(t: ToolSchema) -> RealtimeFunctionToolParam:
    if isinstance(t, FunctionToolSchema):
        return RealtimeFunctionToolParam(
            type="function",
            name=t.function.name,
            description=t.function.description,
            parameters=t.function.parameters,
        )
    raise NotImplementedError(f"OpenAI realtime does not support tool type {t.type!r}")


async def _send_tool_result(
    conn: AsyncRealtimeConnection,
    event: ToolResultEvent,
    serializer: SerializerProto,
) -> None:
    chunks: list[str] = []
    for part in event.result.parts:
        if isinstance(part, TextInput):
            chunks.append(part.content)
        elif isinstance(part, DataInput):
            chunks.append(serializer.encode(part.data).decode())
        else:
            chunks.append(str(part))

    await conn.conversation.item.create(
        item={
            "type": "function_call_output",
            "call_id": event.parent_id,
            "output": "\n".join(chunks),
        },
    )
    await conn.response.create()


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
        elif event.type == "response.output_audio.delta":
            await context.send(SynthesizedAudioEvent(base64.b64decode(event.delta)))
        elif event.type == "response.output_text.delta":
            text += event.delta
            await context.send(ModelMessageChunk(event.delta))
        elif event.type == "response.output_item.done":
            item = event.item
            if item.type == "function_call" and item.call_id and item.name:
                await context.send(
                    ToolCallEvent(
                        id=item.call_id,
                        name=item.name,
                        arguments=item.arguments or "{}",
                    ),
                )
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
