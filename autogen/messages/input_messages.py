# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Any, Callable, Optional

from .base_message import BaseMessage, wrap_message

__all__ = [
    "InputRequestMessage",
    "InputResponseMessage",
    "MultipleChoiceInputRequestMessage",
    "MultipleChoiceInputResponseMessage",
    "PasswordInputRequestMessage",
    "PasswordInputResponseMessage",
    "SingleChoiceInputRequestMessage",
    "SingleChoiceInputResponseMessage",
    "TextInputRequestMessage",
    "TextInputResponsetMessage",
]


class InputRequestMessage(BaseMessage, ABC):
    prompt: str

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        f(self.prompt, flush=True)


@wrap_message
class TextInputRequestMessage(InputRequestMessage):
    pass


@wrap_message
class PasswordInputRequestMessage(InputRequestMessage):
    pass


@wrap_message
class SingleChoiceInputRequestMessage(InputRequestMessage):
    choices: list[str]


@wrap_message
class MultipleChoiceInputRequestMessage(InputRequestMessage):
    choices: list[str]


class InputResponseMessage(BaseMessage, ABC):
    request: InputRequestMessage


@wrap_message
class TextInputResponsetMessage(InputResponseMessage):
    text: str
    request: TextInputRequestMessage

    def __str__(self) -> str:
        return self.text


@wrap_message
class PasswordInputResponseMessage(InputResponseMessage):
    password: str
    request: PasswordInputRequestMessage

    def __str__(self) -> str:
        return self.password


@wrap_message
class SingleChoiceInputResponseMessage(InputResponseMessage):
    choice: str
    request: SingleChoiceInputRequestMessage

    def __str__(self) -> str:
        return self.choice


@wrap_message
class MultipleChoiceInputResponseMessage(InputResponseMessage):
    choices: list[str]
    request: MultipleChoiceInputRequestMessage

    def __str__(self) -> str:
        return str(self.choices)
