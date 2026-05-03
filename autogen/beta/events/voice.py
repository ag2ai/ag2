# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.events import BaseEvent, Field


class TranscriptionChunkEvent(BaseEvent):
    content: str = Field(kw_only=False)
