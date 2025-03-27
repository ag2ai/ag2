# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from autogen.events.print_event import PrintEvent
from autogen.messages.print_message import PrintMessage


class TestPrintMessage:
    def test_deprecation(self) -> None:
        print_message = PrintMessage("Hello, World!", "How are you", sep=" ", end="\n", flush=False)
        assert isinstance(print_message, PrintEvent)
