# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared event-serialization layer."""

import datetime
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from autogen.beta.events import BaseEvent, ModelMessage, ToolCallEvent
from autogen.beta.events._serialization import (
    deserialize_value,
    import_event_class,
    serialize_value,
)


class _Outer:
    """Container for a nested event class used by the import test."""

    class NestedEvent(BaseEvent):
        value: str


class TestImportEventClass:
    def test_resolves_module_level_event(self) -> None:
        cls = import_event_class(f"{ModelMessage.__module__}.{ModelMessage.__qualname__}")
        assert cls is ModelMessage

    def test_resolves_nested_event(self) -> None:
        qualname = f"{_Outer.NestedEvent.__module__}.{_Outer.NestedEvent.__qualname__}"
        cls = import_event_class(qualname)
        assert cls is _Outer.NestedEvent

    def test_returns_none_for_missing_dotted_path(self) -> None:
        assert import_event_class("nonexistent.module.FakeEvent") is None

    def test_returns_none_for_non_event_class(self) -> None:
        assert import_event_class("builtins.int") is None


def _round_trip(value: object) -> object:
    return deserialize_value(serialize_value(value))


class TestPrimitives:
    @pytest.mark.parametrize("value", [None, "hello", 42, 3.14, True, False])
    def test_primitive_round_trip(self, value: object) -> None:
        assert _round_trip(value) == value


class TestBytes:
    def test_bytes_round_trip(self) -> None:
        raw = b"\x12\x34\x00\x99\xff\xab"
        result = _round_trip(raw)
        assert result == raw
        assert isinstance(result, bytes)

    def test_bytearray_round_trip(self) -> None:
        raw = bytearray(b"\x01\x02\x03")
        # bytearray serializes as bytes — round-trip yields bytes (acceptable)
        result = _round_trip(raw)
        assert result == bytes(raw)

    def test_bytes_inside_dict(self) -> None:
        payload = {"signature": b"\x12\x34", "name": "weather"}
        assert _round_trip(payload) == payload

    def test_bytes_inside_list(self) -> None:
        payload = [b"\x01", b"\x02"]
        assert _round_trip(payload) == payload

    def test_bytes_inside_event_provider_data(self) -> None:
        """The original Gemini thought_signature bug — bytes in provider_data."""
        sig = b"\x12\x34\n\x32\x01\x0c\x39\xd6\xc7\xa3"
        event = ToolCallEvent(
            id="call-1",
            name="get_weather",
            arguments='{"city": "Paris"}',
            provider_data={"thought_signature": sig},
        )
        round_tripped = _round_trip(event)
        assert isinstance(round_tripped, ToolCallEvent)
        assert round_tripped.provider_data["thought_signature"] == sig
        assert isinstance(round_tripped.provider_data["thought_signature"], bytes)


class TestUUID:
    def test_uuid_round_trip(self) -> None:
        u = uuid4()
        result = _round_trip(u)
        assert result == u
        assert isinstance(result, UUID)


@dataclass
class _SamplePoint:
    x: int
    y: int
    label: str = "origin"


class TestDataclass:
    def test_dataclass_round_trip(self) -> None:
        point = _SamplePoint(1, 2, "p")
        result = _round_trip(point)
        assert result == point
        assert isinstance(result, _SamplePoint)


class _SamplePydanticModel(BaseModel):
    name: str
    count: int
    payload: bytes | None = None


class TestPydantic:
    def test_pydantic_round_trip(self) -> None:
        model = _SamplePydanticModel(name="hello", count=3)
        result = _round_trip(model)
        assert result == model
        assert isinstance(result, _SamplePydanticModel)

    def test_pydantic_with_bytes_field_round_trip(self) -> None:
        """SDK objects (e.g. OpenAI ImageGenerationCall) embed bytes; pydantic mode='json' handles them."""
        model = _SamplePydanticModel(name="img", count=1, payload=b"\xff\xd8\xff")
        result = _round_trip(model)
        assert result == model


class TestUnknownTypePassthrough:
    """Unknown types pass through ``serialize_value`` unchanged.

    The wire-format boundary (e.g. ``json.dumps`` in the Redis serializer)
    is what surfaces a ``TypeError`` if the value can't be encoded — keeping
    ``serialize_value`` itself permissive preserves backwards compatibility
    for callers like ``BaseEvent.to_dict`` that pair it with their own
    JSON fallback (e.g. ``json.dumps(..., default=str)``).
    """

    def test_unknown_type_passes_through(self) -> None:
        when = datetime.datetime(2026, 4, 29)
        assert serialize_value(when) is when

    def test_unknown_type_inside_dict_passes_through(self) -> None:
        when = datetime.datetime(2026, 4, 29)
        assert serialize_value({"when": when}) == {"when": when}


class TestEventRoundTrip:
    def test_simple_event(self) -> None:
        event = ModelMessage("hello world")
        result = _round_trip(event)
        assert isinstance(result, ModelMessage)
        assert result.content == "hello world"

    def test_tool_call_event(self) -> None:
        event = ToolCallEvent(id="abc", name="search", arguments='{"q": "ag2"}')
        result = _round_trip(event)
        assert isinstance(result, ToolCallEvent)
        assert result.name == "search"
        assert result.arguments == '{"q": "ag2"}'
