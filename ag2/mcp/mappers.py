# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

from mcp.types import (
    AudioContent,
    BlobResourceContents,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from pydantic import AnyUrl, BaseModel

from ag2.events import BinaryResult

if TYPE_CHECKING:
    from ag2.agent import AgentReply

# AG2 ``BinaryResult`` carries no media type of its own; the model layer stashes
# one in ``metadata`` under one of these keys. ``reply.files`` is documented as
# images, so an image default keeps the common path lossless.
_MEDIA_TYPE_KEYS = ("media_type", "mime_type", "mimeType")
_DEFAULT_MEDIA_TYPE = "image/png"


def reply_to_content(reply: "AgentReply[Any, Any]") -> list[ContentBlock]:
    """Convert an :class:`AgentReply` into MCP ``tools/call`` content blocks.

    Inverse of the consume-side ``_extract_content``
    (``ag2/tools/toolkits/mcp_server/toolkit.py``): the reply body
    becomes a :class:`TextContent`, and each generated binary file becomes the
    closest content variant (image / audio / embedded blob resource).
    """
    blocks: list[ContentBlock] = []
    body = reply.body
    if body:
        blocks.append(TextContent(type="text", text=body))
    for file in reply.files:
        blocks.append(_file_to_content(file))
    if not blocks:
        # MCP requires at least one content block; an empty reply maps to "".
        blocks.append(TextContent(type="text", text=""))
    return blocks


def _file_to_content(file: BinaryResult) -> ContentBlock:
    media_type = _media_type(file)
    encoded = base64.b64encode(file.data).decode("ascii")
    if media_type.startswith("image/"):
        return ImageContent(type="image", data=encoded, mimeType=media_type)
    if media_type.startswith("audio/"):
        return AudioContent(type="audio", data=encoded, mimeType=media_type)
    return EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=AnyUrl(f"resource://{file.name}"),
            blob=encoded,
            mimeType=media_type,
        ),
    )


def _media_type(file: BinaryResult) -> str:
    for key in _MEDIA_TYPE_KEYS:
        value = file.metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return _DEFAULT_MEDIA_TYPE


def to_structured_dict(value: Any) -> dict[str, Any] | None:
    """Coerce a validated structured-output value into a JSON-able object dict.

    Returns ``None`` when the value cannot be represented as an object (so the
    caller can skip ``structuredContent`` rather than emit a malformed result).
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return value
    return None
