# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, overload

from autogen.beta.types import AudioMediaType, DocumentMediaType, ImageMediaType, MediaType, VideoMediaType

from .base import BaseEvent, Field


class Input(BaseEvent):
    """Base class for all input events sent to the model."""

    metadata: dict[str, Any] = Field(default_factory=dict, repr=False)

    @classmethod
    def ensure_input(cls, content: "str | Input") -> "Input":
        if isinstance(content, Input):
            return content
        return TextInput(content)


class ModelRequest(BaseEvent):
    """Event representing a user turn sent to the model, containing one or more inputs."""

    inputs: "list[Input]" = Field(kw_only=False)

    @classmethod
    def ensure_request(cls, msgs: "Iterable[str | Input]") -> "ModelRequest":
        return cls([Input.ensure_input(m) for m in msgs])


class TextInput(Input):
    """Text input event sent to the model."""

    content: str = Field(kw_only=False)

    def to_api(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "role": "user",
        }


class BinaryType(str, Enum):
    BINARY = "binary"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    VIDEO = "video"


class BinaryInput(Input):
    """Binary data input event sent to the model."""

    data: bytes = Field(kw_only=False)
    media_type: MediaType | str
    vendor_metadata: dict[str, Any] = Field(default_factory=dict)

    kind: BinaryType = BinaryType.BINARY


class FileIdInput(Input):
    """Input event referencing a pre-uploaded file by ID."""

    file_id: str = Field(kw_only=False)
    filename: str | None = None


class UrlInput(Input):
    kind: BinaryType

    url: str = Field(kw_only=False)


@overload
def ImageInput(url: str) -> UrlInput: ...


@overload
def ImageInput(*, file_id: str, filename: str | None = None) -> FileIdInput: ...


@overload
def ImageInput(*, data: bytes, media_type: ImageMediaType) -> BinaryInput: ...


@overload
def ImageInput(*, path: str | os.PathLike[str], media_type: ImageMediaType | None = None) -> BinaryInput: ...


def ImageInput(  # noqa: N802
    url: str | None = None,
    *,
    file_id: str | None = None,
    filename: str | None = None,
    data: bytes | None = None,
    media_type: ImageMediaType | None = None,
    path: str | os.PathLike[str] | None = None,
) -> UrlInput | FileIdInput | BinaryInput:
    """Factory for creating image input events.

    Usage:
        ImageInput("https://example.com/img.png")           # URL
        ImageInput(file_id="file-abc123")                   # pre-uploaded file
        ImageInput(data=raw_bytes, media_type="image/png")  # raw binary
        ImageInput(path="photo.jpg")                        # local file
    """
    if url is not None:
        return UrlInput(url, kind=BinaryType.IMAGE)

    if file_id is not None:
        return FileIdInput(file_id, filename=filename)

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

        return BinaryInput(
            p.read_bytes(),
            media_type=resolved_type,
            vendor_metadata={"filename": p.name},
            kind=BinaryType.IMAGE,
        )

    if data is not None:
        if media_type is None:
            raise ValueError("'media_type' is required when using 'data'")
        return BinaryInput(
            data,
            media_type=media_type,
            kind=BinaryType.IMAGE,
        )

    raise ValueError("ImageInput() requires one of: 'url', 'file_id', 'data' + 'media_type', or 'path'")


@overload
def DocumentInput(url: str) -> UrlInput: ...


@overload
def DocumentInput(*, file_id: str, filename: str | None = None) -> FileIdInput: ...


@overload
def DocumentInput(*, data: bytes, media_type: DocumentMediaType) -> BinaryInput: ...


@overload
def DocumentInput(*, path: str | os.PathLike[str], media_type: DocumentMediaType | None = None) -> BinaryInput: ...


def DocumentInput(  # noqa: N802
    url: str | None = None,
    *,
    file_id: str | None = None,
    filename: str | None = None,
    data: bytes | None = None,
    media_type: DocumentMediaType | None = None,
    path: str | os.PathLike[str] | None = None,
) -> UrlInput | FileIdInput | BinaryInput:
    """Factory for creating document input events.

    Usage:
        DocumentInput("https://example.com/doc.pdf")               # URL
        DocumentInput(file_id="file-abc123")                       # pre-uploaded file
        DocumentInput(data=raw_bytes, media_type="application/pdf")  # raw binary
        DocumentInput(path="report.pdf")                            # local file
    """
    if url is not None:
        return UrlInput(url, kind=BinaryType.DOCUMENT)

    if file_id is not None:
        return FileIdInput(file_id, filename=filename)

    if path is not None:
        p = Path(path)
        suffix = p.suffix.lower()
        resolved_type = _DOC_EXTENSION_TO_MEDIA_TYPE.get(suffix)

        if resolved_type is None:
            if media_type is None:
                raise ValueError(
                    f"Cannot infer document media type from extension '{suffix}'. Provide 'media_type' explicitly."
                )

            resolved_type = media_type

        return BinaryInput(
            p.read_bytes(),
            media_type=resolved_type,
            vendor_metadata={"filename": p.name},
            kind=BinaryType.DOCUMENT,
        )

    if data is not None:
        if media_type is None:
            raise ValueError("'media_type' is required when using 'data'")
        return BinaryInput(
            data,
            media_type=media_type,
            kind=BinaryType.DOCUMENT,
        )

    raise ValueError("DocumentInput() requires one of: 'url', 'file_id', 'data' + 'media_type', or 'path'")


@overload
def AudioInput(url: str) -> UrlInput: ...


@overload
def AudioInput(*, file_id: str, filename: str | None = None) -> FileIdInput: ...


@overload
def AudioInput(*, data: bytes, media_type: AudioMediaType) -> BinaryInput: ...


@overload
def AudioInput(*, path: str | os.PathLike[str], media_type: AudioMediaType | None = None) -> BinaryInput: ...


def AudioInput(  # noqa: N802
    url: str | None = None,
    *,
    file_id: str | None = None,
    filename: str | None = None,
    data: bytes | None = None,
    media_type: AudioMediaType | None = None,
    path: str | os.PathLike[str] | None = None,
) -> UrlInput | FileIdInput | BinaryInput:
    """Factory for creating audio input events.

    Usage:
        AudioInput("https://example.com/audio.wav")            # URL
        AudioInput(file_id="file-abc123")                      # pre-uploaded file
        AudioInput(data=raw_bytes, media_type="audio/wav")      # raw binary
        AudioInput(path="recording.wav")                        # local file
    """
    if url is not None:
        return UrlInput(url, kind=BinaryType.AUDIO)

    if file_id is not None:
        return FileIdInput(file_id, filename=filename)

    if path is not None:
        p = Path(path)
        suffix = p.suffix.lower()
        resolved_type = _AUDIO_EXTENSION_TO_MEDIA_TYPE.get(suffix)

        if resolved_type is None:
            if media_type is None:
                raise ValueError(
                    f"Cannot infer audio media type from extension '{suffix}'. Provide 'media_type' explicitly."
                )

            resolved_type = media_type

        return BinaryInput(
            p.read_bytes(),
            media_type=resolved_type,
            vendor_metadata={"filename": p.name},
            kind=BinaryType.AUDIO,
        )

    if data is not None:
        if media_type is None:
            raise ValueError("'media_type' is required when using 'data'")
        return BinaryInput(
            data,
            media_type=media_type,
            kind=BinaryType.AUDIO,
        )

    raise ValueError("AudioInput() requires one of: 'url', 'file_id', 'data' + 'media_type', or 'path'")


@overload
def VideoInput(url: str) -> UrlInput: ...


@overload
def VideoInput(*, file_id: str, filename: str | None = None) -> FileIdInput: ...


@overload
def VideoInput(*, data: bytes, media_type: VideoMediaType) -> BinaryInput: ...


@overload
def VideoInput(*, path: str | os.PathLike[str], media_type: VideoMediaType | None = None) -> BinaryInput: ...


def VideoInput(  # noqa: N802
    url: str | None = None,
    *,
    file_id: str | None = None,
    filename: str | None = None,
    data: bytes | None = None,
    media_type: VideoMediaType | None = None,
    path: str | os.PathLike[str] | None = None,
) -> UrlInput | FileIdInput | BinaryInput:
    """Factory for creating video input events.

    Usage:
        VideoInput("https://example.com/video.mp4")            # URL
        VideoInput(file_id="file-abc123")                      # pre-uploaded file
        VideoInput(data=raw_bytes, media_type="video/mp4")     # raw binary
        VideoInput(path="clip.mp4")                            # local file
    """
    if url is not None:
        return UrlInput(url, kind=BinaryType.VIDEO)

    if file_id is not None:
        return FileIdInput(file_id, filename=filename)

    if path is not None:
        p = Path(path)
        suffix = p.suffix.lower()
        resolved_type = _VIDEO_EXTENSION_TO_MEDIA_TYPE.get(suffix)

        if resolved_type is None:
            if media_type is None:
                raise ValueError(
                    f"Cannot infer video media type from extension '{suffix}'. Provide 'media_type' explicitly."
                )

            resolved_type = media_type

        return BinaryInput(
            p.read_bytes(),
            media_type=resolved_type,
            vendor_metadata={"filename": p.name},
            kind=BinaryType.VIDEO,
        )

    if data is not None:
        if media_type is None:
            raise ValueError("'media_type' is required when using 'data'")
        return BinaryInput(
            data,
            media_type=media_type,
            kind=BinaryType.VIDEO,
        )

    raise ValueError("VideoInput() requires one of: 'url', 'file_id', 'data' + 'media_type', or 'path'")


_VIDEO_EXTENSION_TO_MEDIA_TYPE: dict[str, VideoMediaType] = {
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".flv": "video/x-flv",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".wmv": "video/x-ms-wmv",
    ".3gp": "video/3gpp",
}


_AUDIO_EXTENSION_TO_MEDIA_TYPE: dict[str, AudioMediaType] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".aac": "audio/aac",
}


_EXTENSION_TO_MEDIA_TYPE: dict[str, ImageMediaType] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


_DOC_EXTENSION_TO_MEDIA_TYPE: dict[str, DocumentMediaType] = {
    # PDF
    ".pdf": "application/pdf",
    # Text and code
    ".txt": "text/plain",
    ".text": "text/plain",
    ".log": "text/plain",
    ".asm": "text/plain",
    ".bat": "text/plain",
    ".c": "text/plain",
    ".cc": "text/plain",
    ".conf": "text/plain",
    ".cpp": "text/plain",
    ".cxx": "text/plain",
    ".def": "text/plain",
    ".dic": "text/plain",
    ".h": "text/plain",
    ".hh": "text/plain",
    ".in": "text/plain",
    ".ksh": "text/plain",
    ".list": "text/plain",
    ".nws": "text/plain",
    ".pl": "text/plain",
    ".py": "text/plain",
    ".rst": "text/plain",
    ".s": "text/plain",
    ".csv": "text/csv",
    ".tsv": "text/tsv",
    ".iif": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
    ".mht": "text/html",
    ".mhtml": "text/html",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".xml": "text/xml",
    ".json": "application/json",
    ".css": "text/css",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".sql": "application/sql",
    ".eml": "message/rfc822",
    ".mime": "message/rfc822",
    ".ics": "text/calendar",
    ".ifb": "text/calendar",
    ".vcf": "text/vcard",
    ".srt": "text/srt",
    ".vtt": "text/vtt",
    # Rich documents
    ".doc": "application/msword",
    ".dot": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".rtf": "application/rtf",
    ".odt": "application/vnd.oasis.opendocument.text",
    # Spreadsheets
    ".xls": "application/vnd.ms-excel",
    ".xla": "application/vnd.ms-excel",
    ".xlb": "application/vnd.ms-excel",
    ".xlc": "application/vnd.ms-excel",
    ".xlm": "application/vnd.ms-excel",
    ".xlt": "application/vnd.ms-excel",
    ".xlw": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    # Presentations
    ".ppt": "application/vnd.ms-powerpoint",
    ".pot": "application/vnd.ms-powerpoint",
    ".ppa": "application/vnd.ms-powerpoint",
    ".pps": "application/vnd.ms-powerpoint",
    ".pwz": "application/vnd.ms-powerpoint",
    ".wiz": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
