# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.events import BaseEvent, Field


class TranscriptionChunkEvent(BaseEvent):
    content: str = Field(kw_only=False)


class TranscriptionCompletedEvent(BaseEvent):
    content: str = Field(kw_only=False)


class SynthesizedAudioEvent(BaseEvent):
    content: bytes = Field(kw_only=False)


class RecordedAudioEvent(BaseEvent):
    content: bytes = Field(kw_only=False)
