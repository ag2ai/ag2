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
from autogen.beta.events._serialization import (
    deserialize_payload,
    deserialize_value,
    event_to_dict,
    import_event_class,
    qualified_name,
    qualified_name_from_class,
    serialize_value,
)


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
                "type": qualified_name(self.event),
                "data": event_to_dict(self.event),
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
            event_cls = import_event_class(event_type_name)
        if event_cls is None:
            raise ValueError(f"Cannot resolve event type: {event_type_name}")

        # Reconstruct nested events marked with __event__ key
        event_payload = deserialize_payload(event_payload, event_registry)
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
        name = qualified_name_from_class(event_cls)
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
# Backward-compatible aliases for private helpers.
# New code should import from autogen.beta.events._serialization directly.
# ---------------------------------------------------------------------------

_qualified_name = qualified_name
_qualified_name_from_class = qualified_name_from_class
_event_to_dict = event_to_dict
_serialize_value = serialize_value
_deserialize_payload = deserialize_payload
_deserialize_value = deserialize_value
_import_event_class = import_event_class
