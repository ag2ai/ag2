# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pickle
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from autogen.beta.events import BaseEvent

# Pickle deserialization on network data (Redis pub/sub) allows RCE for any
# actor with write access to the Redis channel.  JSON is the safe default.
# Set AG2_ALLOW_PICKLE_DESERIALIZATION=1 only in isolated, trusted environments.
_PICKLE_ENABLED: bool = os.environ.get("AG2_ALLOW_PICKLE_DESERIALIZATION") == "1"

_PICKLE_DISABLED_MSG = (
    "Pickle deserialization is disabled for security. "
    "Set AG2_ALLOW_PICKLE_DESERIALIZATION=1 to opt in. "
    "See docs/security/deserialization.md"
)


class Serializer(Enum):
    """Serialization format for Redis storage and pub/sub transport."""

    JSON = "json"  # default
    PICKLE = "pickle"  # requires AG2_ALLOW_PICKLE_DESERIALIZATION=1


def serialize(obj: Any, fmt: Serializer) -> bytes:
    """Serialize an event to bytes using the specified format."""
    if fmt is Serializer.PICKLE:
        # Guard: pickle on network data is an RCE vector; require explicit opt-in.
        if not _PICKLE_ENABLED:
            raise ValueError(_PICKLE_DISABLED_MSG)
        return pickle.dumps(obj)
    return json.dumps(_to_json(obj)).encode()


def deserialize(data: bytes, fmt: Serializer) -> Any:
    """Deserialize bytes back to an event using the specified format."""
    if fmt is Serializer.PICKLE:
        # Guard: pickle.loads on untrusted Redis bytes allows arbitrary code execution.
        if not _PICKLE_ENABLED:
            raise ValueError(_PICKLE_DISABLED_MSG)
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


# Registry of allowed event/dataclass types for JSON deserialization.
# importlib.import_module on attacker-controlled __type__ strings allows RCE;
# a registry lookup restricts deserialization to pre-approved classes only.
_EVENT_REGISTRY: dict[str, type] = {}


def register_event_class(cls: type) -> type:
    """Register a class so it can be deserialized from JSON.

    Use as a decorator or call explicitly after class definition:

        @register_event_class
        class MyEvent(BaseEvent): ...

    All BaseEvent subclasses are auto-registered at import time via
    _auto_register_base_event_subclasses().
    """
    key = f"{cls.__module__}.{cls.__qualname__}"
    _EVENT_REGISTRY[key] = cls
    return cls


def _auto_register_base_event_subclasses() -> None:
    """Walk all already-loaded BaseEvent subclasses and register them.

    Called once at module import.  New subclasses defined afterwards must
    use @register_event_class explicitly.
    """
    stack = list(BaseEvent.__subclasses__())
    while stack:
        sub = stack.pop()
        register_event_class(sub)
        stack.extend(sub.__subclasses__())


_auto_register_base_event_subclasses()


def _resolve_class(type_path: str) -> type:
    """Return the class for type_path using the safe registry.

    Raises ValueError for unregistered type paths instead of dynamically
    importing attacker-controlled module names.
    """
    cls = _EVENT_REGISTRY.get(type_path)
    if cls is None:
        raise ValueError(
            f"Unregistered event type: {type_path!r}. Decorate the class with @register_event_class before publishing."
        )
    return cls


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
            # Exception type resolution: fall back to base Exception on unknown types
            # to avoid dynamic import of attacker-controlled exc_type strings.
            exc_cls = _EVENT_REGISTRY.get(data.get("exc_type", ""), Exception)
            return exc_cls(data.get("message", ""))

        cls = _resolve_class(type_path)
        fields_data = {k: _from_json(v) for k, v in data.items() if k != "__type__"}

        if isinstance(cls, type) and issubclass(cls, BaseEvent):
            return cls(**fields_data)

        if is_dataclass(cls):
            return cls(**fields_data)

        return fields_data

    return data
