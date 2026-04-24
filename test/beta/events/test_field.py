# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.events import BaseEvent, Field


class TestFieldBasics:
    def test_event_with_field(self):
        class Event(BaseEvent):
            a: str = Field()
            b: int

        assert Event.a.name == "a"
        assert Event.b.name == "b"

        obj = Event(a="1", b=1)
        assert obj.a == "1"
        assert obj.b == 1

    def test_event_with_value_field(self):
        class Event(BaseEvent):
            a: str = "1"

        obj = Event()
        assert obj.a == "1"

    def test_event_with_default_field(self):
        class Event(BaseEvent):
            a: str = Field("1")

        obj = Event()
        assert obj.a == "1"

    def test_event_with_default_factory(self):
        class Event(BaseEvent):
            a: str = Field(default_factory=lambda: "1")

        obj = Event()
        assert obj.a == "1"


class TestFieldInit:
    def test_init_true_accepts_value(self):
        class Event(BaseEvent):
            a: str = Field(init=True)

        obj = Event(a="hello")
        assert obj.a == "hello"

    def test_init_false_applies_default(self):
        class Event(BaseEvent):
            a: str
            _internal: int = Field(default=0, init=False)

        obj = Event(a="hello")
        assert obj.a == "hello"
        assert obj._internal == 0

    def test_init_false_with_default_factory(self):
        class Event(BaseEvent):
            items: list = Field(default_factory=list, init=False)

        obj = Event()
        assert obj.items == []

    def test_init_false_separate_instances(self):
        class Event(BaseEvent):
            items: list = Field(default_factory=list, init=False)

        a = Event()
        b = Event()
        a.items.append(1)
        assert b.items == []


class TestFieldRepr:
    def test_repr_true_shows_field(self):
        class Event(BaseEvent):
            a: str

        obj = Event(a="hello")
        assert "a='hello'" in repr(obj)

    def test_repr_false_hides_field(self):
        class Event(BaseEvent):
            a: str
            secret: str = Field(default="hidden", repr=False)

        obj = Event(a="visible")
        assert "a='visible'" in repr(obj)
        assert "secret" not in repr(obj)

    def test_repr_false_field_still_accessible(self):
        class Event(BaseEvent):
            secret: str = Field(default="hidden", repr=False)

        obj = Event()
        assert obj.secret == "hidden"

    def test_repr_mixed_fields(self):
        class Event(BaseEvent):
            public: str
            internal: str = Field(default="x", repr=False)
            also_public: int = 42

        obj = Event(public="hello")
        r = repr(obj)
        assert "public='hello'" in r
        assert "also_public=42" in r
        assert "internal" not in r


class TestFieldCompare:
    def test_compare_true_includes_field(self):
        class Event(BaseEvent):
            a: str
            b: int

        assert Event(a="x", b=1) == Event(a="x", b=1)
        assert Event(a="x", b=1) != Event(a="x", b=2)

    def test_compare_false_excludes_field(self):
        class Event(BaseEvent):
            a: str
            timestamp: int = Field(default=0, compare=False)

        assert Event(a="x", timestamp=1) == Event(a="x", timestamp=2)
        assert Event(a="x", timestamp=1) != Event(a="y", timestamp=1)

    def test_compare_false_all_fields(self):
        class Event(BaseEvent):
            a: str = Field(compare=False)
            b: int = Field(compare=False)

        assert Event(a="x", b=1) == Event(a="y", b=2)

    def test_compare_different_types(self):
        class EventA(BaseEvent):
            a: str

        class EventB(BaseEvent):
            a: str

        assert EventA(a="x") != EventB(a="x")

    def test_compare_with_custom_eq_override(self):
        class Event(BaseEvent):
            a: str
            b: int

            def __eq__(self, other: object) -> bool:
                if not isinstance(other, Event):
                    return NotImplemented
                return self.a == other.a

        assert Event(a="x", b=1) == Event(a="x", b=99)
        assert Event(a="x", b=1) != Event(a="y", b=1)

    def test_compare_inherited_fields(self):
        class Parent(BaseEvent):
            a: str

        class Child(Parent):
            b: int = Field(compare=False)

        assert Child(a="x", b=1) == Child(a="x", b=2)
        assert Child(a="x", b=1) != Child(a="y", b=1)


class TestFieldPositionalArgs:
    def test_positional_arg(self):
        class Event(BaseEvent):
            a: str = Field(kw_only=False)

        obj = Event("hello")
        assert obj.a == "hello"

    def test_positional_and_kwarg(self):
        class Event(BaseEvent):
            a: str = Field(kw_only=False)
            b: int

        obj = Event("hello", b=1)
        assert obj.a == "hello"
        assert obj.b == 1

    def test_multiple_positional(self):
        class Event(BaseEvent):
            a: str = Field(kw_only=False)
            b: int = Field(kw_only=False)

        obj = Event("hello", 42)
        assert obj.a == "hello"
        assert obj.b == 42

    def test_positional_still_accepts_kwarg(self):
        class Event(BaseEvent):
            a: str = Field(kw_only=False)

        obj = Event(a="hello")
        assert obj.a == "hello"

    def test_too_many_positional_raises(self):
        class Event(BaseEvent):
            a: str = Field(kw_only=False)

        with pytest.raises(TypeError, match="1 positional argument"):
            Event("hello", "extra")

    def test_duplicate_positional_and_kwarg_raises(self):
        class Event(BaseEvent):
            a: str = Field(kw_only=False)

        with pytest.raises(TypeError, match="multiple values"):
            Event("hello", a="world")

    def test_inherited_positional(self):
        class Parent(BaseEvent):
            a: str = Field(kw_only=False)

        class Child(Parent):
            b: int

        obj = Child("hello", b=1)
        assert obj.a == "hello"
        assert obj.b == 1
