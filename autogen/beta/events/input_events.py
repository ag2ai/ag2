# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from autogen.beta.types import MediaType

from .base import BaseEvent, Field


class Input(BaseEvent):
    """Base class for all input events sent to the model."""

    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ensure_input(cls, content: str | Input) -> Input:
        if isinstance(content, Input):
            return content
        return TextInput(content=content)


class TextInput(Input):
    """Text input event sent to the model."""

    content: str

    def to_api(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "role": "user",
        }


class BinaryInput(Input):
    """Binary data input event sent to the model."""

    data: bytes
    media_type: MediaType | str
    vendor_metadata: dict[str, Any] = Field(default_factory=dict)


class ImageInput(Input):
    """Image input event sent to the model."""

    url: str
