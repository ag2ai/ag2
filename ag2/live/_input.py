# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging

from fast_depends.library.serializer import SerializerProto

from ag2.events import BinaryInput, DataInput, DrainedModelRequest, Input, ModelRequest, TextInput, UrlInput
from ag2.exceptions import UnsupportedInputError

logger = logging.getLogger(__name__)


# The input kinds every live session sends; the rest of the contract is in `RealtimeConfig`.
SENDABLE_INPUTS = (TextInput, DataInput)


def sendable_parts(request: ModelRequest, *, provider: str) -> list[Input]:
    """Select the parts of a pushed request a live session sends: `TextInput` and `DataInput`.

    Any other part raises `UnsupportedInputError`, except in a
    `DrainedModelRequest`, where it is logged and dropped so the rest of the
    drained inbox still reaches the model.
    """
    parts: list[Input] = []
    for part in request.parts:
        if isinstance(part, SENDABLE_INPUTS):
            parts.append(part)
        elif isinstance(request, DrainedModelRequest):
            logger.warning(
                "Dropped %s from an inbox message: input type not supported by provider `%s`",
                _input_kind(part),
                provider,
            )
        else:
            raise UnsupportedInputError(_input_kind(part), provider)
    return parts


def request_texts(request: ModelRequest, serializer: SerializerProto, *, provider: str) -> list[str]:
    """Convert the sendable parts of a pushed request to the texts a text-only live session sends.

    `TextInput` is taken as is and `DataInput` encoded by `serializer`; other
    parts are refused or dropped as `sendable_parts` describes.
    """
    texts: list[str] = []
    for part in sendable_parts(request, provider=provider):
        if isinstance(part, TextInput):
            texts.append(part.content)
        elif isinstance(part, DataInput):
            texts.append(serializer.encode(part.data).decode())
    return texts


def _input_kind(part: Input) -> str:
    if isinstance(part, (UrlInput, BinaryInput)):
        return f"{type(part).__name__}({part.kind.value})"
    return type(part).__name__
