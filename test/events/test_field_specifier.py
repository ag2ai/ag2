# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The field specifier is a type-time function and a runtime class under one name.

Type checkers read ``Field`` as the function declared under ``TYPE_CHECKING``;
at runtime it is an alias for ``FieldInfo``. Only the alias is exercised here —
the aliasing is what an import order nobody exercises could break.
"""

from ag2.events import BaseEvent, Condition, Field, FieldInfo


class TestFieldSpecifierAlias:
    def test_exported_name_is_the_runtime_class(self):
        assert Field is FieldInfo

    def test_calling_the_exported_name_yields_a_descriptor(self):
        field = Field(default="x")

        assert isinstance(field, FieldInfo)
        assert field.get_default() == "x"

    def test_declaring_with_the_exported_name_installs_a_descriptor(self):
        class Event(BaseEvent):
            a: str = Field(default="x")

        assert isinstance(Event.a, FieldInfo)
        assert Event.a.name == "a"
        assert Event().a == "x"

    def test_class_level_comparison_still_builds_a_condition(self):
        class Event(BaseEvent):
            a: str = Field(default="x")

        assert isinstance(Event.a == "x", Condition)
