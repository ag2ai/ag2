# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Event serialization utilities."""

import base64
import importlib
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeGuard
from uuid import UUID

from pydantic import BaseModel

if TYPE_CHECKING:
    from .base import BaseEvent

# Wraps a plain dict whose keys collide with a wire marker, so data supplied by
# users, tools or remote peers is restored as a dict and never read as a type tag.
_DICT_MARKER = "__ag2_dict__"
_RESERVED_KEYS = frozenset({
    "__event__",
    "__enum__",
    "__exception__",
    "__bytes__",
    "__uuid__",
    "__dataclass__",
    "__pydantic__",
    _DICT_MARKER,
})


def _is_event_instance(value: Any) -> bool:
    return hasattr(type(value), "_event_fields_")


def _is_event_class(obj: Any) -> "TypeGuard[type[BaseEvent]]":
    return isinstance(obj, type) and hasattr(obj, "_event_fields_")


def event_field_names(cls: type) -> set[str]:
    """Names of the fields an event class declares across its MRO."""
    names: set[str] = set()
    for klass in cls.__mro__:
        names.update(getattr(klass, "_event_fields_", {}))
    return names


def qualified_name(event: "BaseEvent") -> str:
    """Get the fully qualified name of an event instance's class."""
    return qualified_name_from_class(type(event))


def qualified_name_from_class(cls: type) -> str:
    """Get the fully qualified name of an event class."""
    return f"{cls.__module__}.{cls.__qualname__}"


def event_to_dict(event: "BaseEvent") -> dict[str, Any]:
    """Serialize an event to a dictionary.

    Uses the event's __dict__ which contains all field values set by
    the EventMeta-generated __init__.
    """
    result: dict[str, Any] = {}
    for key, value in event.__dict__.items():
        if key.startswith("_"):
            continue
        result[key] = serialize_value(value)
    return result


def serialize_value(value: Any) -> Any:
    """Recursively serialize a value for JSON compatibility."""
    if _is_event_instance(value):
        return {"__event__": qualified_name(value), **event_to_dict(value)}
    if isinstance(value, Enum):
        return {"__enum__": qualified_name_from_class(type(value)), "value": value.value}
    if isinstance(value, Exception):
        return {"__exception__": type(value).__name__, "message": str(value)}
    if isinstance(value, dict):
        serialized = {k: serialize_value(v) for k, v in value.items()}
        if _RESERVED_KEYS.intersection(serialized):
            return {_DICT_MARKER: serialized}
        return serialized
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, UUID):
        return {"__uuid__": str(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": qualified_name_from_class(type(value)),
            "data": {f.name: serialize_value(getattr(value, f.name)) for f in fields(value)},
        }
    if isinstance(value, BaseModel):
        return {
            "__pydantic__": qualified_name_from_class(type(value)),
            "data": serialize_value(value.model_dump(mode="python")),
        }
    # Primitives (str, int, float, bool, None) pass through
    return value


def deserialize_payload(
    payload: dict[str, Any],
    event_registry: Any | None = None,
) -> dict[str, Any]:
    """Recursively reconstruct nested events and special types in a payload."""
    result: dict[str, Any] = {}
    for key, value in payload.items():
        result[key] = deserialize_value(value, event_registry)
    return result


def deserialize_value(value: Any, event_registry: Any | None = None) -> Any:
    """Recursively deserialize a value from wire format."""
    if isinstance(value, dict):
        if _DICT_MARKER in value:
            # Escaped plain dict: restore its items without reading their keys as markers.
            inner = value[_DICT_MARKER]
            if not isinstance(inner, dict):
                raise ValueError(f"{_DICT_MARKER} must wrap an object")
            return {k: deserialize_value(v, event_registry) for k, v in inner.items()}
        if "__event__" in value:
            # Nested event
            event_type_name = value["__event__"]
            event_cls = _resolve_event_type(event_type_name, event_registry)
            if event_cls is not None:
                known = event_field_names(event_cls)
                nested_data = {k: deserialize_value(v, event_registry) for k, v in value.items() if k in known}
                return event_cls(**nested_data)
        if "__bytes__" in value:
            return base64.b64decode(value["__bytes__"])
        if "__uuid__" in value:
            return UUID(value["__uuid__"])
        if "__enum__" in value:
            # Fail open: an enum whose class can no longer be imported, or whose
            # member has since been removed, degrades to its raw value rather
            # than failing the load and taking the whole event with it.
            try:
                enum_cls = _resolve_class(value["__enum__"])
                if issubclass(enum_cls, Enum):
                    return enum_cls(value["value"])
            except (ImportError, ValueError):
                pass
            return value["value"]
        if "__exception__" in value:
            # Reconstruct as a generic Exception with the original message
            return Exception(value.get("message", ""))
        if "__dataclass__" in value:
            cls = _resolve_class(value["__dataclass__"])
            # Check the type before calling it: the tag names a class to construct.
            if not is_dataclass(cls):
                raise TypeError(f"{value['__dataclass__']!r} is not a dataclass")
            raw_data = value.get("data", {})
            if not isinstance(raw_data, dict):
                raise ValueError("__dataclass__ data must be an object")
            data = {k: deserialize_value(v, event_registry) for k, v in raw_data.items()}
            return cls(**data)
        if "__pydantic__" in value:
            cls = _resolve_class(value["__pydantic__"])
            if not issubclass(cls, BaseModel):
                raise TypeError(f"{value['__pydantic__']!r} is not a pydantic model")
            raw = deserialize_value(value.get("data", {}), event_registry)
            return cls.model_validate(raw)
        return {k: deserialize_value(v, event_registry) for k, v in value.items()}
    if isinstance(value, list):
        return [deserialize_value(v, event_registry) for v in value]
    return value


def _resolve_class(type_path: str) -> type:
    parts = type_path.split(".")
    for i in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:i])
        attr_chain = parts[i:]
        try:
            module = importlib.import_module(module_path)
            obj: Any = module
            for attr in attr_chain:
                obj = getattr(obj, attr)
            if isinstance(obj, type):
                return obj
        except (ImportError, AttributeError):
            continue
    raise ImportError(f"Could not resolve class {type_path!r}")


def _resolve_event_type(type_name: str, event_registry: Any | None = None) -> "type[BaseEvent] | None":
    """Resolve an event type name to a class.

    Tries the registry first (if provided), then falls back to import-based resolution.
    """
    if event_registry is not None:
        cls: type[BaseEvent] | None = event_registry.resolve(type_name)
        if cls is not None:
            return cls
    return import_event_class(type_name)


def import_event_class(type_name: str) -> "type[BaseEvent] | None":
    """Import an event class by its fully qualified name.

    Handles nested qualnames (e.g. ``module.path.Outer.Inner``) by walking
    attribute chains after importing the module.
    """
    # Try progressively shorter module paths to handle nested qualnames.
    parts = type_name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:i])
        attr_chain = parts[i:]
        try:
            module = importlib.import_module(module_path)
            obj: Any = module
            for attr in attr_chain:
                obj = getattr(obj, attr)
            if _is_event_class(obj):
                return obj
        except (ImportError, AttributeError):
            continue
    return None
