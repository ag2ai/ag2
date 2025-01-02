# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

from autogen.base_message import BaseMessage


def test_BaseMessage():
    uuid = uuid4()

    actual = BaseMessage(uuid=uuid)
    expected_model_dump = {"uuid": uuid}

    assert actual.model_dump() == expected_model_dump
