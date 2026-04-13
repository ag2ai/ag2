# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import types
from typing import Literal, TypeAlias, TypeVar

AudioMediaType: TypeAlias = Literal[
    "audio/wav",
    "audio/mpeg",
    "audio/ogg",
    "audio/flac",
    "audio/aiff",
    "audio/aac",
]
ImageMediaType: TypeAlias = Literal[
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
]
DocumentMediaType: TypeAlias = Literal[
    "application/pdf",
    # Text and code
    "text/plain",
    "text/csv",
    "text/tsv",
    "text/html",
    "text/markdown",
    "text/xml",
    "text/css",
    "text/javascript",
    "text/calendar",
    "text/vcard",
    "text/srt",
    "text/vtt",
    "application/json",
    "application/sql",
    "message/rfc822",
    # Rich documents
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/rtf",
    "application/vnd.oasis.opendocument.text",
    # Spreadsheets
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    # Presentations
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
]
VideoMediaType: TypeAlias = Literal[
    "video/x-matroska",
    "video/quicktime",
    "video/mp4",
    "video/webm",
    "video/x-flv",
    "video/mpeg",
    "video/x-ms-wmv",
    "video/3gpp",
]

MediaType: TypeAlias = AudioMediaType | ImageMediaType | DocumentMediaType | VideoMediaType

ClassInfo: TypeAlias = type | types.UnionType | tuple["ClassInfo", ...]


class Omit:
    def __bool__(self) -> Literal[False]:
        return False


omit = Omit()

_T = TypeVar("_T")

Omittable = _T | Omit
