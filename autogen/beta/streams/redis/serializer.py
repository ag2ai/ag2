# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import pickle
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from autogen.beta.events import BaseEvent
from autogen.beta.events.base import EventMeta


class Serializer(Enum):
    """Serialization format for Redis storage and pub/sub transport."""

    JSON = "json"  # default
    PICKLE = "pickle"


def serialize(obj: Any, fmt: Serializer) -> bytes:
    """Serialize an event to bytes using the specified format."""
    if fmt is Serializer.PICKLE:
        return pickle.dumps(obj)
    return json.dumps(_to_json(obj)).encode()


def deserialize(data: bytes, fmt: Serializer) -> Any:
    """Deserialize bytes back to an event using the specified format."""
    if fmt is Serializer.PICKLE:
        return pickle.loads(data)  # noqa: S301
    return _from_json(json.loads(data))


def _to_json(obj: Any) -> Any:
    """Recursively serialize an object to JSON-compatible types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, BaseEvent):
        cls = type(obj)
        data: dict[str, Any] = {"__type__": f"{cls.__module__}.{cls.__qualname__}"}
        for key, value in obj.__dict__.items():
            if not key.startswith("_"):
                data[key] = _to_json(value)
        return data

    if is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        data = {"__type__": f"{cls.__module__}.{cls.__qualname__}"}
        for f in fields(obj):
            data[f.name] = _to_json(getattr(obj, f.name))
        return data

    if isinstance(obj, Exception):
        return {
            "__type__": "exception",
            "exc_type": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "message": str(obj),
        }

    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_json(item) for item in obj]

    return str(obj)


def _resolve_class(type_path: str) -> type:
    """Import and return the class from a dotted path."""
    module_path, _, class_name = type_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _from_json(data: Any) -> Any:
    """Recursively deserialize JSON data back to event objects."""
    if data is None or isinstance(data, (str, int, float, bool)):
        return data

    if isinstance(data, list):
        return [_from_json(item) for item in data]

    if isinstance(data, dict):
        type_path = data.get("__type__")
        if not type_path:
            return {k: _from_json(v) for k, v in data.items()}

        if type_path == "exception":
            try:
                exc_cls = _resolve_class(data["exc_type"])
            except (ImportError, AttributeError):
                exc_cls = Exception
            return exc_cls(data.get("message", ""))

        cls = _resolve_class(type_path)
        fields_data = {k: _from_json(v) for k, v in data.items() if k != "__type__"}

        if isinstance(cls, EventMeta):
            return cls(**fields_data)

        if is_dataclass(cls):
            return cls(**fields_data)

        return fields_data

    return data
