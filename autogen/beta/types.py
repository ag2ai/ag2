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
    "text/plain",
    "text/csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/html",
    "text/markdown",
    "application/msword",
    "application/vnd.ms-excel",
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
