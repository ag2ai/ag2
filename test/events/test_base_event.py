# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from uuid import UUID

import pytest
from pydantic import BaseModel

from autogen.events.base_event import (
    BaseEvent,
    _event_classes,
    wrap_event,
)


@pytest.fixture
def TestEvent() -> Generator[type[BaseEvent], None, None]:  # noqa: N802
    org_event_classes = _event_classes.copy()
    try:

        @wrap_event
        class TestEvent(BaseEvent):
            sender: str
            receiver: str
            content: str

        yield TestEvent
    finally:
        _event_classes.clear()
        _event_classes.update(org_event_classes)


class TestBaseEvent:
    def test_model_dump_validate(self, TestEvent: type[BaseModel], uuid: UUID) -> None:  # noqa: N803
        # print(f"{TestMessage=}")

        event = TestEvent(uuid=uuid, sender="sender", receiver="receiver", content="Hello, World!")

        expected = {
            "type": "test",
            "content": {
                "uuid": uuid,
                "sender": "sender",
                "receiver": "receiver",
                "content": "Hello, World!",
            },
        }
        actual = event.model_dump()
        assert actual == expected

        model = TestEvent.model_validate(expected)
        assert model.model_dump() == expected

        model = TestEvent(**expected)
        assert model.model_dump() == expected

    def test_single_content_parameter_event(self, uuid: UUID) -> None:
        @wrap_event
        class TestSingleContentParameterEvent(BaseEvent):
            content: str

        message = TestSingleContentParameterEvent(uuid=uuid, content="Hello, World!")

        expected = {"type": "test_single_content_parameter", "content": {"content": "Hello, World!", "uuid": uuid}}
        assert message.model_dump() == expected

        model = TestSingleContentParameterEvent.model_validate(expected)
        assert model.model_dump() == expected

        model = TestSingleContentParameterEvent(**expected)
        assert model.model_dump() == expected
