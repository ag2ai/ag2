# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from base64 import b64decode
from collections.abc import Iterable
from enum import Enum
from typing import Any, cast

from a2a.types import Part

from autogen.beta.events import (
    BinaryInput,
    DataInput,
    FileIdInput,
    TextInput,
    UrlInput,
)
from autogen.beta.events.input_events import BinaryType, Input

from ._proto import dict_to_struct, dict_to_value, struct_to_dict, value_to_python
from .wire import (
    BINARY_TYPE_KEY,
    FILENAME_KEY,
    METADATA_PREFIX,
    PART_KIND_KEY,
    VENDOR_METADATA_KEY,
)


class PartKind(str, Enum):
    """Marker for the original beta ``Input`` subtype, recorded in ``Part.metadata``."""

    TEXT = "text"
    BINARY = "binary"
    URL = "url"
    FILE_ID = "file_id"
    DATA_DICT = "data_dict"
    DATA_VALUE = "data_value"


def inputs_to_a2a_parts(inputs: Iterable[Input]) -> list[Part]:
    return [input_to_part(inp) for inp in inputs]


def a2a_parts_to_inputs(parts: Iterable[Part]) -> list[Input]:
    return [part_to_input(p) for p in parts]


def input_to_part(inp: Input) -> Part:
    if isinstance(inp, TextInput):
        return Part(
            text=inp.content,
            metadata=dict_to_struct(merge_metadata({PART_KIND_KEY: PartKind.TEXT.value}, inp.metadata)),
        )
    if isinstance(inp, BinaryInput):
        meta = merge_metadata(
            {
                PART_KIND_KEY: PartKind.BINARY.value,
                BINARY_TYPE_KEY: inp.kind.value,
                VENDOR_METADATA_KEY: dict(inp.vendor_metadata) or None,
            },
            inp.metadata,
        )
        # ``BinaryInput.data`` is bytes; the wire stores the raw payload directly
        # rather than base64-encoding it, since proto bytes fields are binary-safe.
        # Some upstreams pre-encode; we accept either by detecting non-bytes input.
        raw = inp.data if isinstance(inp.data, (bytes, bytearray)) else b64decode(inp.data)
        return Part(raw=bytes(raw), media_type=str(inp.media_type), metadata=dict_to_struct(meta))
    if isinstance(inp, FileIdInput):
        return Part(
            data=dict_to_value({"file_id": inp.file_id, "filename": inp.filename}),
            filename=inp.filename or None,
            metadata=dict_to_struct(
                merge_metadata({PART_KIND_KEY: PartKind.FILE_ID.value, FILENAME_KEY: inp.filename}, inp.metadata)
            ),
        )
    if isinstance(inp, UrlInput):
        return Part(
            url=inp.url,
            metadata=dict_to_struct(
                merge_metadata(
                    {PART_KIND_KEY: PartKind.URL.value, BINARY_TYPE_KEY: inp.kind.value},
                    inp.metadata,
                )
            ),
        )
    if isinstance(inp, DataInput):
        if isinstance(inp.data, dict):
            return Part(
                data=dict_to_value(cast(dict[str, Any], inp.data)),
                metadata=dict_to_struct(merge_metadata({PART_KIND_KEY: PartKind.DATA_DICT.value}, inp.metadata)),
            )
        return Part(
            data=dict_to_value({"value": inp.data}),
            metadata=dict_to_struct(merge_metadata({PART_KIND_KEY: PartKind.DATA_VALUE.value}, inp.metadata)),
        )
    raise TypeError(f"Unsupported Input type: {type(inp).__name__}")


def part_to_input(part: Part) -> Input:
    raw_metadata = struct_to_dict(part.metadata) if part.HasField("metadata") else {}
    kind_marker = raw_metadata.get(PART_KIND_KEY)
    user_metadata = strip_internal_metadata(raw_metadata)
    content_field = part.WhichOneof("content")

    if content_field == "text":
        return TextInput(part.text, metadata=user_metadata)

    if content_field == "raw":
        return BinaryInput(
            bytes(part.raw),
            media_type=part.media_type or "application/octet-stream",
            kind=coerce_binary_type(raw_metadata.get(BINARY_TYPE_KEY)),
            vendor_metadata=dict(raw_metadata.get(VENDOR_METADATA_KEY) or {}),
            metadata=user_metadata,
        )

    if content_field == "url":
        return UrlInput(
            part.url,
            kind=coerce_binary_type(raw_metadata.get(BINARY_TYPE_KEY)),
            metadata=user_metadata,
        )

    if content_field == "data":
        decoded = value_to_python(part.data)
        if kind_marker == PartKind.FILE_ID.value and isinstance(decoded, dict):
            return FileIdInput(
                str(decoded.get("file_id")),
                filename=cast(str | None, decoded.get("filename") or part.filename or None),
                metadata=user_metadata,
            )
        if kind_marker == PartKind.DATA_VALUE.value and isinstance(decoded, dict):
            return DataInput(decoded.get("value"), metadata=user_metadata)
        return DataInput(decoded, metadata=user_metadata)

    raise TypeError(f"Part has no content set (oneof empty): {part!r}")


def merge_metadata(internal: dict[str, Any], user: dict[str, Any] | None) -> dict[str, Any] | None:
    """Merge AG2-internal markers with caller-supplied user metadata. Drops ``None`` values."""
    merged: dict[str, Any] = {k: v for k, v in internal.items() if v is not None}
    if user:
        merged.update(user)
    return merged or None


def strip_internal_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove all AG2-internal markers from a metadata dict."""
    return {k: v for k, v in metadata.items() if not k.startswith(METADATA_PREFIX)}


def coerce_binary_type(value: Any) -> BinaryType:
    if isinstance(value, BinaryType):
        return value
    if isinstance(value, str):
        try:
            return BinaryType(value)
        except ValueError:
            return BinaryType.BINARY
    return BinaryType.BINARY
