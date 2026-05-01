# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict, ParseDict


def dict_to_struct(value: dict[str, Any] | None) -> struct_pb2.Struct | None:
    """Convert a JSON-friendly dict to ``google.protobuf.Struct``.

    Returns ``None`` for empty / falsy inputs so callers can pass the result
    straight to a proto constructor (proto fields are optional by default in
    A2A 1.0; ``None`` means "leave unset").
    """
    if not value:
        return None
    out = struct_pb2.Struct()
    ParseDict(value, out)
    return out


def struct_to_dict(value: struct_pb2.Struct | None) -> dict[str, Any]:
    """Convert ``google.protobuf.Struct`` (or absent field) to a plain dict."""
    if value is None or not value.fields:
        return {}
    return MessageToDict(value, preserving_proto_field_name=True)


def dict_to_value(value: Any) -> struct_pb2.Value:
    """Convert any JSON-shaped Python value to ``google.protobuf.Value``."""
    out = struct_pb2.Value()
    ParseDict(value, out)
    return out


def value_to_python(value: struct_pb2.Value | None) -> Any:
    """Convert ``google.protobuf.Value`` (or absent) back to a Python object."""
    if value is None:
        return None
    return MessageToDict(value, preserving_proto_field_name=True)
