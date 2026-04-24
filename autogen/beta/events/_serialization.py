# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Event serialization utilities.

Shared by BaseEvent.to_dict/from_dict and Envelope wire format.
Placed in the events package to avoid circular imports between
Layer 1 (events) and Layer 2 (network primitives).
"""

import base64
import importlib
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BaseEvent


def _is_event_instance(value: Any) -> bool:
    return hasattr(type(value), "_event_fields_")


def _is_event_class(obj: Any) -> bool:
    return isinstance(obj, type) and hasattr(obj, "_event_fields_")


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
        return value.value
    if isinstance(value, Exception):
        return {"__exception__": type(value).__name__, "message": str(value)}
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
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
        if "__event__" in value:
            # Nested event
            event_type_name = value["__event__"]
            event_cls = _resolve_event_type(event_type_name, event_registry)
            if event_cls is not None:
                nested_data = {k: deserialize_value(v, event_registry) for k, v in value.items() if k != "__event__"}
                return event_cls(**nested_data)
        if "__bytes__" in value:
            return base64.b64decode(value["__bytes__"])
        if "__exception__" in value:
            # Reconstruct as a generic Exception with the original message
            return Exception(value.get("message", ""))
        return {k: deserialize_value(v, event_registry) for k, v in value.items()}
    if isinstance(value, list):
        return [deserialize_value(v, event_registry) for v in value]
    return value


def _resolve_event_type(type_name: str, event_registry: Any | None = None) -> "type[BaseEvent] | None":
    """Resolve an event type name to a class.

    Tries the registry first (if provided), then falls back to import-based resolution.
    """
    if event_registry is not None:
        cls = event_registry.resolve(type_name)
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
