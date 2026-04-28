# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for event-class import utilities used by ``EventLogWriter``."""

from autogen.beta.events import BaseEvent, ModelMessage
from autogen.beta.events._serialization import import_event_class


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
