# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Literal

from fast_depends.library.serializer import SerializerProto
from google.genai import Client
from google.genai import types as gtypes
from google.genai.live import AsyncSession

from ag2.context import ConversationContext
from ag2.events import (
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
    Usage,
    UsageEvent,
)
from ag2.tools.final import FunctionToolSchema
from ag2.tools.schemas import ToolSchema

from .realtime import RealtimeConfig

# Gemini Live audio I/O is fixed by the API contract:
# input is 16-bit PCM @ 16kHz, output is 16-bit PCM @ 24kHz, both little-endian.
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
INPUT_MIME_TYPE = f"audio/pcm;rate={INPUT_SAMPLE_RATE}"


LiveVoice = Literal[
    "Aoede",
    "Charon",
    "Fenrir",
    "Kore",
    "Leda",
    "Orus",
    "Puck",
    "Zephyr",
]
ModelName = Literal[
    "gemini-2.0-flash-live-001",
    "gemini-live-2.5-flash-preview",
    "gemini-2.5-flash-preview-native-audio-dialog",
    "gemini-2.5-flash-native-audio-preview-09-2025",
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-3.1-flash-live-preview",
]


@dataclass(slots=True)
class AudioOutput:
    """Audio output config for the Gemini live session.

    Mirrors a subset of ``google.genai.types.SpeechConfig``. Selecting
    ``AudioOutput`` opts the session into the ``AUDIO`` response modality
    and enables ``output_audio_transcription`` so the assistant's spoken
    response is also surfaced as ``ModelMessageChunk`` events.
    """

    voice: "LiveVoice | str" = "Kore"
    language_code: str | None = None


@dataclass(slots=True)
class TextOutput:
    """Text-only output for the Gemini live session.

    Selects the ``TEXT`` modality; the model returns raw text via
    ``ModelMessageChunk`` events with no audio playback.
    """


@dataclass(slots=True)
class InputConfig:
    """Input-side audio knobs for the Gemini live session.

    Set ``transcribe=True`` to receive the user's speech as
    ``TranscriptionChunkEvent`` / ``TranscriptionCompletedEvent`` events.
    """

    transcribe: bool = False
    transcription_languages: list[str] | None = None
    automatic_activity_detection: gtypes.AutomaticActivityDetectionDict | None = None
    activity_handling: gtypes.ActivityHandling | None = None
    turn_coverage: gtypes.TurnCoverage | None = None


class RealTimeConfig(RealtimeConfig):
    """Realtime config backed by Gemini's bidirectional Live API.

    Implements the `RealtimeConfig` protocol — call `session(...)` to
    open a websocket connection that pumps captured audio into the API
    and emits transcription, audio, and tool-call events on the supplied
    context.
    """

    def __init__(
        self,
        model: "ModelName | str",
        *,
        output: AudioOutput | TextOutput | None = None,
        input: InputConfig | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        config: gtypes.LiveConnectConfigDict | None = None,
        client: Client | None = None,
    ) -> None:
        self.model = model

        if output is None:
            output = AudioOutput()
        if input is None:
            input = InputConfig()

        base: gtypes.LiveConnectConfigDict = {}
        if temperature is not None:
            base["temperature"] = temperature
        if max_output_tokens is not None:
            base["max_output_tokens"] = max_output_tokens

        if isinstance(output, AudioOutput):
            base["response_modalities"] = [gtypes.Modality.AUDIO]
            speech: gtypes.SpeechConfigDict = {
                "voice_config": {"prebuilt_voice_config": {"voice_name": output.voice}},
            }
            if output.language_code is not None:
                speech["language_code"] = output.language_code
            base["speech_config"] = speech
            # Surface assistant text alongside the audio so observers can
            # consume `ModelMessageChunk` without parsing audio bytes.
            base["output_audio_transcription"] = {}
        else:
            base["response_modalities"] = [gtypes.Modality.TEXT]

        if input.transcribe:
            input_tx: gtypes.AudioTranscriptionConfigDict = {}
            if input.transcription_languages is not None:
                input_tx["language_codes"] = list(input.transcription_languages)
            base["input_audio_transcription"] = input_tx

        realtime: gtypes.RealtimeInputConfigDict = {}
        if input.automatic_activity_detection is not None:
            realtime["automatic_activity_detection"] = input.automatic_activity_detection
        if input.activity_handling is not None:
            realtime["activity_handling"] = input.activity_handling
        if input.turn_coverage is not None:
            realtime["turn_coverage"] = input.turn_coverage
        if realtime:
            base["realtime_input_config"] = realtime

        self._config: gtypes.LiveConnectConfigDict = base
        self._config_overrides: gtypes.LiveConnectConfigDict = config or {}

        self.client = client or Client()

    def _build_session(
        self,
        *,
        instructions: Iterable[str] = (),
        tools: Iterable[ToolSchema] = (),
    ) -> gtypes.LiveConnectConfigDict:
        overlay: gtypes.LiveConnectConfigDict = {}
        joined = "\n".join(instructions)
        if joined:
            overlay["system_instruction"] = joined
        function_decls = [_tool_schema_to_function_declaration(t) for t in tools]
        if function_decls:
            overlay["tools"] = [{"function_declarations": function_decls}]
        return {**self._config, **overlay, **self._config_overrides}

    @asynccontextmanager
    async def session(
        self,
        context: ConversationContext,
        *,
        instructions: Iterable[str] = (),
        tools: Iterable[ToolSchema] = (),
        serializer: SerializerProto,
    ) -> AsyncIterator[None]:
        final_config = self._build_session(instructions=instructions, tools=tools)

        async with self.client.aio.live.connect(model=self.model, config=final_config) as session:

            async def _pump_audio(event: RecordedAudioEvent) -> None:
                await session.send_realtime_input(
                    audio=gtypes.Blob(data=event.content, mime_type=INPUT_MIME_TYPE),
                )

            async def _forward_tool_result(event: ToolResultEvent) -> None:
                await _send_tool_result(session, event, serializer)

            with (
                context.stream.where(RecordedAudioEvent).sub_scope(_pump_audio),
                context.stream.where(ToolResultEvent).sub_scope(_forward_tool_result),
            ):
                recv_task = asyncio.create_task(_pump_events(session, context, self.model))

                try:
                    yield

                finally:
                    recv_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await recv_task


def _tool_schema_to_function_declaration(t: ToolSchema) -> gtypes.FunctionDeclarationDict:
    if isinstance(t, FunctionToolSchema):
        return {
            "name": t.function.name,
            "description": t.function.description,
            "parameters": _ensure_object_schema(t.function.parameters),
        }
    raise NotImplementedError(f"Gemini Live does not support tool type {t.type!r}")


def _ensure_object_schema(params: dict[str, Any] | None) -> dict[str, Any]:
    raw_type = str((params or {}).get("type", "")).lower()
    if not params or raw_type in ("null", "none", ""):
        return {"type": "object", "properties": {}}
    return params


async def _send_tool_result(
    session: AsyncSession,
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

    await session.send_tool_response(
        function_responses=gtypes.FunctionResponse(
            id=event.parent_id,
            name=event.name,
            response={"result": "\n".join(chunks)},
        ),
    )


def normalize_realtime_usage(metadata: "gtypes.UsageMetadata | None") -> Usage:
    if metadata is None:
        return Usage()
    return Usage(
        prompt_tokens=metadata.prompt_token_count,
        completion_tokens=metadata.response_token_count,
        total_tokens=metadata.total_token_count,
        cache_read_input_tokens=metadata.cached_content_token_count,
        thinking_tokens=metadata.thoughts_token_count,
    )


async def _pump_events(
    session: AsyncSession,
    context: ConversationContext,
    model: str,
) -> None:
    # `session.receive()` is a per-turn iterator (it breaks after each
    # `turn_complete`), so wrap it in a loop to keep the conversation
    # alive across turns. An empty receive() means the connection closed.
    text = ""
    usage = Usage()
    while True:
        had_message = False
        async for message in session.receive():
            had_message = True

            emitted = await _handle_server_content(message.server_content, context)
            text += emitted

            # Usage arrives on its own message field (not server_content), often
            # only once per turn; keep the latest seen value for the rollup.
            if message.usage_metadata is not None:
                usage = normalize_realtime_usage(message.usage_metadata)

            if message.tool_call is not None:
                for fc in message.tool_call.function_calls or ():
                    if fc.name and fc.id:
                        await context.send(
                            ToolCallEvent(
                                id=fc.id,
                                name=fc.name,
                                arguments=_encode_args(fc.args),
                            ),
                        )

            if message.server_content is not None and message.server_content.turn_complete:
                if usage:
                    await context.send(
                        UsageEvent(usage, kind="model_call", model=model, provider="gemini"),
                    )
                await context.send(
                    ModelResponse(
                        message=ModelMessage(text) if text else None,
                        usage=usage,
                        model=model,
                        provider="gemini",
                    ),
                )
                text = ""
                usage = Usage()

        if not had_message:
            return


async def _handle_server_content(
    content: gtypes.LiveServerContent | None,
    context: ConversationContext,
) -> str:
    if content is None:
        return ""

    emitted_text = ""
    if content.input_transcription is not None and content.input_transcription.text:
        await context.send(TranscriptionChunkEvent(content.input_transcription.text))
        if content.input_transcription.finished:
            await context.send(TranscriptionCompletedEvent(content.input_transcription.text))

    if content.output_transcription is not None and content.output_transcription.text:
        chunk = content.output_transcription.text
        emitted_text += chunk
        await context.send(ModelMessageChunk(chunk))

    if content.model_turn is not None:
        for part in content.model_turn.parts or ():
            if part.inline_data is not None and part.inline_data.data:
                await context.send(SynthesizedAudioEvent(part.inline_data.data))
            elif part.text:
                # Non-native-audio models stream text directly here rather
                # than via output_audio_transcription.
                emitted_text += part.text
                await context.send(ModelMessageChunk(part.text))

    return emitted_text


def _encode_args(args: dict[str, Any] | None) -> str:
    if not args:
        return "{}"
    return json.dumps(args)
