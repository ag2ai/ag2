# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from threading import Thread
from typing import Any

from autogen.events.base_event import BaseEvent
from autogen.io import IOConsole, IOStream


class TestIOStream:
    def test_initial_default_io_stream(self) -> None:
        assert isinstance(IOStream.get_default(), IOConsole)

    def test_set_default_io_stream(self) -> None:
        class MyIOStream(IOStream):
            def print(self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
                pass

            def send(self, message: BaseEvent) -> None:
                pass

            def input(self, prompt: str = "", *, password: bool = False) -> str:
                return "Hello, World!"

        assert isinstance(IOStream.get_default(), IOConsole)

        with IOStream.set_default(MyIOStream()):
            assert isinstance(IOStream.get_default(), MyIOStream)

            with IOStream.set_default(IOConsole()):
                assert isinstance(IOStream.get_default(), IOConsole)

            assert isinstance(IOStream.get_default(), MyIOStream)

        assert isinstance(IOStream.get_default(), IOConsole)

    def test_get_default_on_new_thread(self) -> None:
        exceptions: list[Exception] = []

        def on_new_thread(exceptions: list[Exception] = exceptions) -> None:
            try:
                assert isinstance(IOStream.get_default(), IOConsole)
            except Exception as e:
                exceptions.append(e)

        # create a new thread and run the function
        thread = Thread(target=on_new_thread)

        thread.start()

        # get exception from the thread
        thread.join()

        if exceptions:
            raise exceptions[0]
