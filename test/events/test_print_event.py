# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, call
from uuid import uuid4

from autogen.events.print_event import PrintEvent


class TestPrintEvent:
    def test_print(self) -> None:
        uuid = uuid4()
        print_message = PrintEvent("Hello, World!", "How are you", sep=" ", end="\n", flush=False, uuid=uuid)
        assert isinstance(print_message, PrintEvent)

        expected_model_dump = {
            "type": "print",
            "content": {"uuid": uuid, "objects": ["Hello, World!", "How are you"], "sep": " ", "end": "\n"},
        }
        assert print_message.model_dump() == expected_model_dump

        mock = MagicMock()
        print_message.print(f=mock)
        # print(mock.call_args_list)
        expected_call_args_list = [call("Hello, World!", "How are you", sep=" ", end="\n", flush=True)]
        assert mock.call_args_list == expected_call_args_list
