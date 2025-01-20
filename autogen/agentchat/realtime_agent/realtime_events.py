# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel


class RealtimeEvent(BaseModel):
    raw_message: dict[str, Any]


class SessionCreated(RealtimeEvent):
    type: str = "session.created"


class SessionUpdated(RealtimeEvent):
    type: str = "session.updated"


class AudioDelta(RealtimeEvent):
    type: str = "response.audio.delta"
    delta: str
    item_id: Any


class SpeechStarted(RealtimeEvent):
    type: str = "input_audio_buffer.speech_started"


class FunctionCall(RealtimeEvent):
    type: str = "response.function_call_arguments.done"
    name: str
    arguments: dict[str, Any]
    call_id: str
