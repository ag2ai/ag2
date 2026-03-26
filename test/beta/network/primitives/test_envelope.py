# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.events import ModelMessage
from autogen.beta.events.base import BaseEvent
from autogen.beta.network.primitives.envelope import Envelope, EventRegistry, register_event
from autogen.beta.network.primitives.priority import DefaultPriority


class TestEnvelopeCreation:
    def test_basic_creation(self) -> None:
        event = ModelMessage(content="hello")
        env = Envelope(event=event, sender="actor-a")
        assert env.event is event
        assert env.sender == "actor-a"
        assert env.recipient is None
        assert env.trace_id  # Auto-generated
        assert env.correlation_id
        assert env.causation_id is None
        assert env.priority is None
        assert env.ttl is None
        assert env.requires_ack is False

    def test_with_all_fields(self) -> None:
        event = ModelMessage(content="hello")
        env = Envelope(
            event=event,
            sender="a",
            recipient="b",
            trace_id="trace-1",
            correlation_id="corr-1",
            causation_id="cause-1",
            priority=DefaultPriority.URGENT,
            ttl=30.0,
            requires_ack=True,
        )
        assert env.recipient == "b"
        assert env.trace_id == "trace-1"
        assert env.priority == DefaultPriority.URGENT
        assert env.ttl == 30.0
        assert env.requires_ack is True


class TestEnvelopeWireFormat:
    def test_round_trip(self) -> None:
        event = ModelMessage(content="hello world")
        env = Envelope(
            event=event,
            sender="actor-a",
            recipient="actor-b",
            priority=2,
            ttl=60.0,
        )

        data = env.to_dict()
        assert data["v"] == 1
        assert data["sender"] == "actor-a"
        assert data["recipient"] == "actor-b"
        assert data["event"]["data"]["content"] == "hello world"

        restored = Envelope.from_dict(data)
        assert restored.sender == env.sender
        assert restored.recipient == env.recipient
        assert restored.trace_id == env.trace_id
        assert restored.correlation_id == env.correlation_id
        assert restored.priority == env.priority
        assert restored.ttl == env.ttl
        assert restored.event.content == "hello world"

    def test_unsupported_version_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            Envelope.from_dict({"v": 99, "event": {}, "sender": "a"})

    def test_unknown_event_type_raises(self) -> None:
        data = {
            "v": 1,
            "event": {"type": "nonexistent.module.FakeEvent", "data": {}},
            "sender": "a",
            "trace_id": "t",
            "correlation_id": "c",
        }
        with pytest.raises(ValueError, match="Cannot resolve"):
            Envelope.from_dict(data)


class TestEnvelopeChild:
    def test_child_inherits_trace(self) -> None:
        parent = Envelope(
            event=ModelMessage(content="parent"),
            sender="a",
            trace_id="trace-abc",
        )
        child_event = ModelMessage(content="child")
        child = parent.child(child_event, sender="b", recipient="c")

        assert child.trace_id == "trace-abc"  # Inherited
        assert child.causation_id == parent.correlation_id  # Points to parent
        assert child.correlation_id != parent.correlation_id  # New
        assert child.sender == "b"
        assert child.recipient == "c"

    def test_child_inherits_priority(self) -> None:
        parent = Envelope(
            event=ModelMessage(content="p"),
            sender="a",
            priority=DefaultPriority.URGENT,
        )
        child = parent.child(ModelMessage(content="c"))
        assert child.priority == DefaultPriority.URGENT

    def test_child_overrides_priority(self) -> None:
        parent = Envelope(
            event=ModelMessage(content="p"),
            sender="a",
            priority=DefaultPriority.URGENT,
        )
        child = parent.child(ModelMessage(content="c"), priority=DefaultPriority.BACKGROUND)
        assert child.priority == DefaultPriority.BACKGROUND


class TestEnvelopeTTL:
    def test_not_expired_without_ttl(self) -> None:
        env = Envelope(event=ModelMessage(content="m"), sender="a")
        assert not env.is_expired

    def test_not_expired_within_ttl(self) -> None:
        env = Envelope(event=ModelMessage(content="m"), sender="a", ttl=999.0)
        assert not env.is_expired

    def test_expired_after_ttl_elapsed(self) -> None:
        import time

        env = Envelope(
            event=ModelMessage(content="m"),
            sender="a",
            timestamp=time.time() - 5.0,  # 5 seconds in the past
            ttl=2.0,  # 2 second TTL
        )
        assert env.is_expired

    def test_uses_wall_clock_not_monotonic(self) -> None:
        """Timestamp should use time.time() for cross-process compatibility."""
        import time

        before = time.time()
        env = Envelope(event=ModelMessage(content="m"), sender="a")
        after = time.time()

        # Envelope timestamp should be a wall-clock value (epoch seconds)
        assert before <= env.timestamp <= after


class TestEventRegistry:
    def test_register_and_resolve(self) -> None:
        registry = EventRegistry()

        class CustomEvent(BaseEvent):
            value: str

        registry.register(CustomEvent)
        resolved = registry.resolve(
            f"{CustomEvent.__module__}.{CustomEvent.__qualname__}"
        )
        assert resolved is CustomEvent

    def test_resolve_unknown_returns_none(self) -> None:
        registry = EventRegistry()
        assert registry.resolve("nonexistent.Event") is None

    def test_register_event_decorator(self) -> None:
        @register_event
        class DecoratedEvent(BaseEvent):
            data: str

        # Should be resolvable from default registry
        from autogen.beta.network.primitives.envelope import _default_registry

        resolved = _default_registry.resolve(
            f"{DecoratedEvent.__module__}.{DecoratedEvent.__qualname__}"
        )
        assert resolved is DecoratedEvent
