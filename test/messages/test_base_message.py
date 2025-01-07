# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from typing import Generator, Type
from uuid import uuid4

import pytest
from pydantic import BaseModel

from autogen.messages.base_message import (
    BaseMessage,
    _message_classes,
    get_annotated_type_for_message_classes,
    wrap_message,
)


@pytest.fixture()
def TestMessage() -> Generator[Type[BaseMessage], None, None]:
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
    def test_model_dump_validate(self, TestMessage: Type[BaseModel]) -> None:
        uuid = uuid4()

        print(f"{TestMessage=}")

        message = TestMessage(uuid=uuid, sender="sender", receiver="receiver", content="Hello, World!")

        expected = {
            "type": "test_message",
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
