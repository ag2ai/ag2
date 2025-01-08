# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import getpass
from typing import Any

from autogen.messages.base_message import BaseMessage
from autogen.messages.print_message import PrintMessage

from .base import IOStream

__all__ = ("IOConsole",)


class IOConsole(IOStream):
    """A console input/output stream."""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        """Print data to the output stream.

        Args:
            objects (any): The data to print.
            sep (str, optional): The separator between objects. Defaults to " ".
            end (str, optional): The end of the output. Defaults to "\n".
            flush (bool, optional): Whether to flush the output. Defaults to False.
        """
        print_message = PrintMessage(*objects, sep=sep, end=end)
        self.send(print_message)
        # print(*objects, sep=sep, end=end, flush=flush)

    def send(self, message: BaseMessage) -> None:
        """Send a message to the output stream.

        Args:
            message (Any): The message to send.
        """
        message.print()

    def input(self, prompt: str = "", *, password: bool = False) -> str:
        """Read a line from the input stream.

        Args:
            prompt (str, optional): The prompt to display. Defaults to "".
            password (bool, optional): Whether to read a password. Defaults to False.

        Returns:
            str: The line read from the input stream.

        """

        if password:
            return getpass.getpass(prompt if prompt != "" else "Password: ")
        return input(prompt)
