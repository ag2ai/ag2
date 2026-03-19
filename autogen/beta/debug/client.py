# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Serialization utilities for converting live BaseEvent objects into
JSON-compatible dicts, used by the HTTP API when returning session state.
"""

from dataclasses import fields as dataclass_fields, is_dataclass
from typing import Any

from ..events.base import BaseEvent


def _serialize_value(value: Any) -> Any:
    """Recursively convert a value to a JSON-compatible type."""
    if isinstance(value, BaseEvent):
        return {
            "type": type(value).__name__,
            **{k: _serialize_value(v) for k, v in value.__dict__.items() if not k.startswith("_")},
        }
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Exception):
        return {"type": type(value).__name__, "message": str(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _serialize_value(getattr(value, f.name)) for f in dataclass_fields(value)}
    return repr(value)


def serialize_event(event: BaseEvent) -> dict[str, Any]:
    """Return ``{type: str, data: dict}`` for a single BaseEvent."""
    data = {k: _serialize_value(v) for k, v in event.__dict__.items() if not k.startswith("_")}
    return {"type": type(event).__name__, "data": data}
