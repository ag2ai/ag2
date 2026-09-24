# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from a2a.types import Part
from google.protobuf import json_format, struct_pb2

from ag2.events import (
    BinaryInput,
    BinaryType,
    DataInput,
    FileIdInput,
    Input,
    TextInput,
    ToolResult,
    UrlInput,
)

_FILE_ID_METADATA_KEY = "ag2.file_id"
_BINARY_KIND_METADATA_KEY = "ag2.binary_kind"
_FILENAME_METADATA_KEY = "filename"


def input_to_part(inp: Input) -> Part:
    """Convert an AG2 ``Input`` event to an A2A ``Part`` (protobuf, v1.x).

    A ``Part`` in A2A 1.x is a flat protobuf message — exactly one of
    ``text`` / ``raw`` / ``url`` / ``data`` is meaningful, plus optional
    ``filename``, ``media_type``, ``metadata``.
    """
    if isinstance(inp, TextInput):
        return Part(text=inp.content)

    if isinstance(inp, BinaryInput):
        filename = str(inp.vendor_metadata.get("filename", "")) if inp.vendor_metadata else ""
        metadata = struct_from_dict({_BINARY_KIND_METADATA_KEY: inp.kind.value})
        return Part(
            raw=inp.data,
            media_type=str(inp.media_type),
            filename=filename,
            metadata=metadata,
        )

    if isinstance(inp, UrlInput):
        return Part(
            url=inp.url,
            metadata=struct_from_dict({_BINARY_KIND_METADATA_KEY: inp.kind.value}),
        )

    if isinstance(inp, FileIdInput):
        return Part(
            filename=inp.filename or "",
            metadata=struct_from_dict({_FILE_ID_METADATA_KEY: inp.file_id}),
        )

    if isinstance(inp, DataInput):
        return Part(data=_value_from(inp.data), media_type="application/json")

    raise TypeError(f"Cannot map {type(inp).__name__} to A2A Part")


def part_to_input(part: Part) -> Input:
    """Convert an A2A ``Part`` to an AG2 ``Input`` event.

    Picks the field that is populated. For data parts whose ``media_type``
    is one of our extension MIME types, the caller (mappers.tools) is
    expected to handle the part *before* falling through to this function.
    """
    if part.text:
        return TextInput(part.text)

    metadata = struct_to_dict(part.metadata)
    file_id = metadata.get(_FILE_ID_METADATA_KEY)
    if file_id:
        return FileIdInput(str(file_id), filename=part.filename or None)

    kind = _binary_kind(metadata)

    if part.raw:
        return BinaryInput(
            part.raw,
            media_type=part.media_type or "application/octet-stream",
            vendor_metadata={_FILENAME_METADATA_KEY: part.filename} if part.filename else {},
            kind=kind,
        )

    if part.url:
        return UrlInput(part.url, kind=kind)

    if part.HasField("data"):
        return DataInput(_value_to_python(part.data))

    raise ValueError("A2A Part has no populated content field")


def data_part(payload: Any, *, media_type: str) -> Part:
    """Build a Part carrying structured data with the given MIME type.

    Used by mappers.tools to wrap our extension payloads
    (tool-schemas+json, tool-call+json, tool-result+json).
    """
    return Part(data=_value_from(payload), media_type=media_type)


def is_data_part_with_mime(part: Part, media_type: str) -> bool:
    return part.HasField("data") and part.media_type == media_type


def part_data_to_python(part: Part) -> Any:
    """Decode a data Part's ``data`` field into a native Python value."""
    return _value_to_python(part.data)


def tool_result_to_text(result: ToolResult) -> str:
    """Render a ``ToolResult`` to a flat text payload.

    Used by every wire shape that needs the textual content of a tool
    result (``tool-result+json`` Part, ``ag2.history+json`` events).
    Non-text parts fall back to ``str()`` because the wire layer carries
    text only — binary results must be re-attached at the message level
    if ever needed.
    """
    return "".join(part.content if isinstance(part, TextInput) else str(part) for part in result.parts)


def _value_from(value: Any) -> struct_pb2.Value:
    target = struct_pb2.Value()
    json_format.Parse(json.dumps(value, default=_json_default), target)
    return target


def _value_to_python(v: struct_pb2.Value) -> Any:
    return json_format.MessageToDict(v, preserving_proto_field_name=True)


def struct_from_dict(payload: dict[str, Any]) -> struct_pb2.Struct:
    s = struct_pb2.Struct()
    json_format.ParseDict(payload, s)
    return s


def struct_to_dict(s: struct_pb2.Struct) -> dict[str, Any]:
    if not s or not s.fields:
        return {}
    return json_format.MessageToDict(s, preserving_proto_field_name=True)


def _binary_kind(metadata: dict[str, Any]) -> BinaryType:
    raw = metadata.get(_BINARY_KIND_METADATA_KEY, BinaryType.BINARY.value)
    try:
        return BinaryType(raw)
    except ValueError:
        return BinaryType.BINARY


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime, Decimal)):
        return str(obj)
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("ascii")
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"Cannot JSON-serialize {type(obj).__name__}")
