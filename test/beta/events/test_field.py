# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.events import BaseEvent, Field


def test_event_with_field():
    class Event(BaseEvent):
        a: str = Field()
        b: int

    assert Event.a.name == "a"
    assert Event.b.name == "b"

    obj = Event(a="1", b=1)
    assert obj.a == "1"
    assert obj.b == 1


def test_event_with_value_field():
    class Event(BaseEvent):
        a: str = "1"

    obj = Event()
    assert obj.a == "1"


def test_event_with_default_field():
    class Event(BaseEvent):
        a: str = Field("1")

    obj = Event()
    assert obj.a == "1"


def test_event_with_default_factory():
    class Event(BaseEvent):
        a: str = Field(default_factory=lambda: "1")

    obj = Event()
    assert obj.a == "1"
