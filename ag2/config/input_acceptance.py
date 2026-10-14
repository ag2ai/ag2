# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Provider mapper acceptance for inbound media, by position and source."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import get_args

from ag2.config import (
    AnthropicConfig,
    BedrockConfig,
    DashScopeConfig,
    GeminiConfig,
    MistralConfig,
    ModelConfig,
    OllamaConfig,
    OpenAIConfig,
    OpenAIResponsesConfig,
    TypeSafeConfig,
    VertexAIConfig,
    XAIConfig,
    ZAIConfig,
)
from ag2.events import BinaryInput, BinaryType, FileIdInput, Input, UrlInput
from ag2.types import AudioMediaType, DocumentMediaType, ImageMediaType, VideoMediaType

# Each entry names (position, source, kind). Text is accepted everywhere.
# The table describes mapper behavior, not model-specific ability.
_MEDIA: dict[type[object], set[tuple[str, str, BinaryType | None]]] = {
    OpenAIConfig: {
        ("user", "data", BinaryType.IMAGE),
        ("user", "url", BinaryType.IMAGE),
        ("user", "data", BinaryType.AUDIO),
        ("user", "data", BinaryType.DOCUMENT),
        ("user", "file", None),
    },
    OpenAIResponsesConfig: {
        *(
            (position, source, kind)
            for position in ("user", "tool")
            for source in ("data", "url")
            for kind in (BinaryType.IMAGE, BinaryType.DOCUMENT)
        ),
        ("user", "file", None),
        ("tool", "file", None),
    },
    AnthropicConfig: {
        *(
            (position, source, kind)
            for position in ("user", "tool")
            for source in ("data", "url")
            for kind in (BinaryType.IMAGE, BinaryType.DOCUMENT)
        ),
        ("user", "file", None),
        ("tool", "file", None),
    },
    BedrockConfig: {
        *(
            (position, "data", kind)
            for position in ("user", "tool")
            for kind in (BinaryType.IMAGE, BinaryType.DOCUMENT, BinaryType.VIDEO)
        ),
    },
    DashScopeConfig: {
        *((position, source, BinaryType.IMAGE) for position in ("user", "tool") for source in ("data", "url")),
    },
    GeminiConfig: {
        *(
            ("user", source, kind)
            for source in ("data", "url")
            for kind in (BinaryType.IMAGE, BinaryType.AUDIO, BinaryType.VIDEO, BinaryType.DOCUMENT)
        ),
        *(("tool", source, kind) for source in ("data", "url") for kind in (BinaryType.IMAGE, BinaryType.DOCUMENT)),
        ("user", "file", None),
    },
    VertexAIConfig: {
        *(
            ("user", source, kind)
            for source in ("data", "url")
            for kind in (BinaryType.IMAGE, BinaryType.AUDIO, BinaryType.VIDEO, BinaryType.DOCUMENT)
        ),
        *(("tool", source, kind) for source in ("data", "url") for kind in (BinaryType.IMAGE, BinaryType.DOCUMENT)),
        ("user", "file", None),
    },
    MistralConfig: {
        *(
            (position, source, kind)
            for position in ("user", "tool")
            for source in ("data", "url")
            for kind in (BinaryType.IMAGE, BinaryType.DOCUMENT)
        ),
        ("user", "file", None),
        ("tool", "file", None),
    },
    OllamaConfig: {("user", "data", BinaryType.IMAGE)},
    XAIConfig: {
        *(("user", source, kind) for source in ("data", "url") for kind in (BinaryType.IMAGE, BinaryType.DOCUMENT)),
        ("user", "file", None),
    },
    ZAIConfig: set(),
    TypeSafeConfig: set(),
}


def _described(config: ModelConfig | None) -> type[object] | None:
    """The config class the table describes `config` by, or `None` if it does not."""
    if config is None:
        return None
    # Compared by identity, never `isinstance`: a provider whose extra is not
    # installed is a stand-in here, not a class.
    return type(config) if type(config) in _MEDIA else None


def accepts_input(config: ModelConfig | None, position: str, part: Input) -> bool:
    """Whether the config's mapper accepts this media in this position.

    A config the table does not describe is taken to accept anything: its mapper
    cannot be read from here.
    """
    described = _described(config)
    if described is None:
        return True
    if isinstance(part, BinaryInput):
        source = "data"
        kind = part.kind
    elif isinstance(part, UrlInput):
        source = "url"
        kind = part.kind
    elif isinstance(part, FileIdInput):
        source = "file"
        kind = None
    else:
        return True
    if (position, source, kind) not in _MEDIA[described]:
        return False
    if isinstance(part, BinaryInput):
        if described is BedrockConfig:
            supported_media = {
                BinaryType.IMAGE: {"image/png", "image/jpeg", "image/gif", "image/webp"},
                BinaryType.DOCUMENT: {
                    "application/pdf",
                    "text/csv",
                    "application/msword",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/vnd.ms-excel",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "text/html",
                    "text/plain",
                    "text/markdown",
                },
                BinaryType.VIDEO: {
                    "video/x-matroska",
                    "video/quicktime",
                    "video/mp4",
                    "video/webm",
                    "video/x-flv",
                    "video/mpeg",
                    "video/x-ms-wmv",
                    "video/3gpp",
                },
            }
            return kind is not None and part.media_type in supported_media.get(kind, set())
        if described is OpenAIConfig and kind is BinaryType.AUDIO:
            return part.media_type in ("audio/wav", "audio/mpeg", "audio/mp3")
        if described is AnthropicConfig and kind is BinaryType.IMAGE:
            return part.media_type in ("image/jpeg", "image/png", "image/gif", "image/webp")
        if described is AnthropicConfig and kind is BinaryType.DOCUMENT:
            return part.media_type in ("application/pdf", "text/plain")
    return True


def input_modalities(config: ModelConfig | None) -> dict[str, bool]:
    """Media kinds accepted in user messages, in AG-UI capability vocabulary.

    Empty for a config the table does not describe: a modality left undeclared
    says nothing, where `True` would promise what cannot be checked.
    """
    described = _described(config)
    if described is None:
        return {}
    accepted = _MEDIA[described]
    return {
        name: any(position == "user" and kind is binary_kind for position, _, kind in accepted)
        for name, binary_kind in (
            ("image", BinaryType.IMAGE),
            ("audio", BinaryType.AUDIO),
            ("video", BinaryType.VIDEO),
            ("pdf", BinaryType.DOCUMENT),
        )
    }


_KIND_OF_MEDIA_TYPE: Mapping[str, BinaryType] = MappingProxyType({
    **dict.fromkeys(get_args(ImageMediaType), BinaryType.IMAGE),
    **dict.fromkeys(get_args(AudioMediaType), BinaryType.AUDIO),
    **dict.fromkeys(get_args(VideoMediaType), BinaryType.VIDEO),
    **dict.fromkeys(get_args(DocumentMediaType), BinaryType.DOCUMENT),
})


def binary_kind_of(media_type: str) -> BinaryType | None:
    """Kind of a typed media type (``image/png`` -> IMAGE), or ``None`` for one `ag2.types` does not list."""
    return _KIND_OF_MEDIA_TYPE.get(media_type)


def input_media_types(config: ModelConfig | None) -> tuple[str, ...]:
    """Media types the config's mapper accepts as inline user data, in `ag2.types` order.

    Asks `accepts_input` about each typed media type, so the answer follows the same
    per-provider rules as the mapper. A config the table does not describe accepts
    every typed media type, as `accepts_input` does.
    """
    return tuple(
        media_type
        for media_type, kind in _KIND_OF_MEDIA_TYPE.items()
        if accepts_input(config, "user", BinaryInput(b"", media_type=media_type, kind=kind))
    )


__all__ = ("accepts_input", "binary_kind_of", "input_media_types", "input_modalities")
