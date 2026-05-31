# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from uuid import UUID

import pytest
from pydantic import BaseModel

from autogen.messages.base_message import (
    BaseMessage,
    _message_classes,
    wrap_message,
)


@pytest.fixture
def TestMessage() -> Generator[type[BaseMessage], None, None]:  # noqa: N802
    org_message_classes = _message_classes.copy()
    try:

        @wrap_message
        class TestMessage(BaseMessage):
            sender: str
            receiver: str
            content: str

        yield TestMessage
    finally:
        _message_classes.clear()
        _message_classes.update(org_message_classes)


class TestBaseMessage:
    def test_model_dump_validate(self, TestMessage: type[BaseModel], uuid: UUID) -> None:  # noqa: N803
        # print(f"{TestMessage=}")

        message = TestMessage(uuid=uuid, sender="sender", receiver="receiver", content="Hello, World!")

        expected = {
            "type": "test",
            "content": {
                "uuid": uuid,
                "sender": "sender",
                "receiver": "receiver",
                "content": "Hello, World!",
            },
        }
        actual = message.model_dump()
        assert actual == expected

        model = TestMessage.model_validate(expected)
        assert model.model_dump() == expected

        model = TestMessage(**expected)
        assert model.model_dump() == expected

    def test_single_content_parameter_message(self, uuid: UUID) -> None:
        @wrap_message
        class TestSingleContentParameterMessage(BaseMessage):
            content: str

        message = TestSingleContentParameterMessage(uuid=uuid, content="Hello, World!")

        expected = {"type": "test_single_content_parameter", "content": {"content": "Hello, World!", "uuid": uuid}}
        assert message.model_dump() == expected

        model = TestSingleContentParameterMessage.model_validate(expected)
        assert model.model_dump() == expected

        model = TestSingleContentParameterMessage(**expected)
        assert model.model_dump() == expected
