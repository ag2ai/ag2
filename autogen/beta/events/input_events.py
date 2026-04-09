# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, overload

from autogen.beta.types import ImageMediaType, MediaType

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

    content: str = Field(kw_only=False)

    def to_api(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "role": "user",
        }


class BinaryInput(Input):
    """Binary data input event sent to the model."""

    data: bytes = Field(kw_only=False)
    media_type: MediaType | str
    vendor_metadata: dict[str, Any] = Field(default_factory=dict)


class ImageUrlInput(Input):
    """Image input event sent to the model."""

    url: str = Field(kw_only=False)


@overload
def ImageInput(url: str) -> ImageUrlInput: ...


@overload
def ImageInput(*, data: bytes, media_type: ImageMediaType) -> BinaryInput: ...


@overload
def ImageInput(*, path: str | PathLike[str], media_type: ImageMediaType | None = None) -> BinaryInput: ...


def ImageInput(  # noqa: N802
    url: str | None = None,
    *,
    data: bytes | None = None,
    media_type: ImageMediaType | None = None,
    path: str | PathLike[str] | None = None,
) -> ImageUrlInput | BinaryInput:
    """Factory for creating image input events.

    Usage:
        ImageInput("https://example.com/img.png")          # URL
        ImageInput(data=raw_bytes, media_type="image/png")  # raw binary
        ImageInput(path="photo.jpg")                        # local file
    """
    if url is not None:
        return ImageUrlInput(url)

    if path is not None:
        p = Path(path)
        suffix = p.suffix.lower()
        resolved_type = _EXTENSION_TO_MEDIA_TYPE.get(suffix)

        if resolved_type is None:
            if media_type is None:
                raise ValueError(
                    f"Cannot infer image media type from extension '{suffix}'. Provide 'media_type' explicitly."
                )

            resolved_type = media_type

        return BinaryInput(p.read_bytes(), media_type=resolved_type)

    if data is not None:
        if media_type is None:
            raise ValueError("'media_type' is required when using 'data'")
        return BinaryInput(data, media_type=media_type)

    raise ValueError("Image() requires one of: 'url', 'data' + 'media_type', or 'path'")


_EXTENSION_TO_MEDIA_TYPE: dict[str, ImageMediaType] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
