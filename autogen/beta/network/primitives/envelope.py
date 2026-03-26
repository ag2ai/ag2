# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Envelope — event wrapper with network metadata.

An Envelope wraps a BaseEvent with addressing, tracing, priority, and delivery
requirements. It is what flows through Channels. Core agents that don't need
network features continue using Stream.send(event). Network-aware code uses
Channel.send(envelope).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from autogen.beta.events import BaseEvent


def _uuid4_hex() -> str:
    return uuid4().hex


@dataclass(slots=True)
class Envelope:
    """Event wrapper with network metadata."""

    event: BaseEvent

    # Addressing
    sender: str
    recipient: str | None = None  # None = broadcast

    # Tracing
    trace_id: str = field(default_factory=_uuid4_hex)  # Groups entire workflow
    correlation_id: str = field(default_factory=_uuid4_hex)  # Groups request-response
    causation_id: str | None = None  # Points to parent envelope

    # Priority & delivery
    priority: Any = None  # Interpreted by PriorityScheme
    timestamp: float = field(default_factory=time.time)
    ttl: float | None = None  # Time-to-live in seconds
    requires_ack: bool = False  # Whether delivery must be acknowledged

    # -----------------------------------------------------------------------
    # Wire format
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Canonical JSON-serializable representation.

        Wire format contract:
        - ``"v": 1`` — schema version for backward-compatible evolution
        - Event type is a qualified Python name for deserialization
        - Transports choose their own encoding (JSON, MessagePack, Protobuf)
        - LocalChannel skips serialization entirely
        """
        return {
            "v": 1,
            "event": {
                "type": _qualified_name(self.event),
                "data": _event_to_dict(self.event),
            },
            "sender": self.sender,
            "recipient": self.recipient,
            "trace_id": self.trace_id,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "requires_ack": self.requires_ack,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        event_registry: EventRegistry | None = None,
    ) -> Envelope:
        """Reconstruct from wire format.

        Uses event_registry for type resolution. Falls back to import-based
        resolution for standard AG2 event types.
        """
        version = data.get("v", 1)
        if version != 1:
            raise ValueError(f"Unsupported envelope wire format version: {version}")

        event_data = data["event"]
        event_type_name = event_data["type"]
        event_payload = event_data["data"]

        # Resolve event class
        registry = event_registry or _default_registry
        event_cls = registry.resolve(event_type_name)
        if event_cls is None:
            event_cls = _import_event_class(event_type_name)
        if event_cls is None:
            raise ValueError(f"Cannot resolve event type: {event_type_name}")

        # Reconstruct nested events marked with __event__ key
        event_payload = _deserialize_payload(event_payload, event_registry)
        try:
            event = event_cls(**event_payload)
        except TypeError as exc:
            raise ValueError(
                f"Failed to construct {event_type_name} from payload: {exc}. "
                f"Payload keys: {sorted(event_payload.keys())}"
            ) from exc

        return cls(
            event=event,
            sender=data["sender"],
            recipient=data.get("recipient"),
            trace_id=data["trace_id"],
            correlation_id=data["correlation_id"],
            causation_id=data.get("causation_id"),
            priority=data.get("priority"),
            timestamp=data.get("timestamp", 0.0),
            ttl=data.get("ttl"),
            requires_ack=data.get("requires_ack", False),
        )

    _UNSET = object()  # Sentinel for distinguishing None from "not provided"

    def child(
        self,
        event: BaseEvent,
        *,
        sender: str | None = None,
        recipient: Any = _UNSET,
        priority: Any = None,
    ) -> Envelope:
        """Create a child envelope inheriting trace lineage.

        Pass ``recipient=None`` explicitly to create a broadcast child.
        Omit ``recipient`` to inherit from parent.
        """
        resolved_recipient = self.recipient if recipient is Envelope._UNSET else recipient
        resolved_sender = sender if sender is not None else self.sender
        return Envelope(
            event=event,
            sender=resolved_sender,
            recipient=resolved_recipient,
            trace_id=self.trace_id,  # Same workflow
            correlation_id=_uuid4_hex(),  # New request-response group
            causation_id=self.correlation_id,  # Points to parent
            priority=priority if priority is not None else self.priority,
        )

    @property
    def is_expired(self) -> bool:
        """Check if the envelope has exceeded its TTL."""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl


# ---------------------------------------------------------------------------
# Event Registry
# ---------------------------------------------------------------------------


class EventRegistry:
    """Registry for custom event type deserialization.

    Standard AG2 event types are resolved via import. Custom event types
    (from plugins, domain code) register here for wire format support.
    """

    def __init__(self) -> None:
        self._types: dict[str, type[BaseEvent]] = {}

    def register(self, event_cls: type[BaseEvent]) -> None:
        """Register a custom event type for wire format deserialization."""
        name = _qualified_name_from_class(event_cls)
        self._types[name] = event_cls

    def resolve(self, type_name: str) -> type[BaseEvent] | None:
        """Resolve a type name to an event class."""
        return self._types.get(type_name)


_default_registry = EventRegistry()


def register_event(event_cls: type[BaseEvent]) -> type[BaseEvent]:
    """Decorator to register a custom event type for wire format support.

    Example::

        @register_event
        class MyCustomEvent(BaseEvent):
            data: str
    """
    _default_registry.register(event_cls)
    return event_cls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qualified_name(event: BaseEvent) -> str:
    """Get the fully qualified name of an event instance's class."""
    cls = type(event)
    return _qualified_name_from_class(cls)


def _qualified_name_from_class(cls: type) -> str:
    """Get the fully qualified name of an event class."""
    module = cls.__module__
    qualname = cls.__qualname__
    return f"{module}.{qualname}"


def _event_to_dict(event: BaseEvent) -> dict[str, Any]:
    """Serialize an event to a dictionary.

    Uses the event's __dict__ which contains all field values set by
    the EventMeta-generated __init__.
    """
    result: dict[str, Any] = {}
    for key, value in event.__dict__.items():
        if key.startswith("_"):
            continue
        result[key] = _serialize_value(value)
    return result


def _serialize_value(value: Any) -> Any:
    """Recursively serialize a value for JSON compatibility."""
    from enum import Enum

    if isinstance(value, BaseEvent):
        return {"__event__": _qualified_name(value), **_event_to_dict(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        import base64

        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    # Primitives (str, int, float, bool, None) pass through
    return value


def _deserialize_payload(
    payload: dict[str, Any],
    event_registry: EventRegistry | None = None,
) -> dict[str, Any]:
    """Recursively reconstruct nested events and special types in a payload."""
    result: dict[str, Any] = {}
    for key, value in payload.items():
        result[key] = _deserialize_value(value, event_registry)
    return result


def _deserialize_value(value: Any, event_registry: EventRegistry | None = None) -> Any:
    """Recursively deserialize a value from wire format."""
    if isinstance(value, dict):
        if "__event__" in value:
            # Nested event
            event_type_name = value["__event__"]
            registry = event_registry or _default_registry
            event_cls = registry.resolve(event_type_name)
            if event_cls is None:
                event_cls = _import_event_class(event_type_name)
            if event_cls is not None:
                nested_data = {k: _deserialize_value(v, event_registry) for k, v in value.items() if k != "__event__"}
                return event_cls(**nested_data)
        if "__bytes__" in value:
            import base64

            return base64.b64decode(value["__bytes__"])
        return {k: _deserialize_value(v, event_registry) for k, v in value.items()}
    if isinstance(value, list):
        return [_deserialize_value(v, event_registry) for v in value]
    return value


def _import_event_class(type_name: str) -> type[BaseEvent] | None:
    """Import an event class by its fully qualified name.

    Handles nested qualnames (e.g. ``module.path.Outer.Inner``) by walking
    attribute chains after importing the module.
    """
    import importlib

    # Try progressively shorter module paths to handle nested qualnames.
    # For "a.b.C.D" we try: import "a.b.C" getattr "D",
    # then import "a.b" getattr "C.D", then import "a" getattr "b.C.D".
    parts = type_name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:i])
        attr_chain = parts[i:]
        try:
            module = importlib.import_module(module_path)
            obj: Any = module
            for attr in attr_chain:
                obj = getattr(obj, attr)
            if isinstance(obj, type) and issubclass(obj, BaseEvent):
                return obj
        except (ImportError, AttributeError):
            continue
    return None
