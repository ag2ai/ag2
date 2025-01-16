# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from typing import Any

__all__ = [
    "AgentNameConflict",
    "InvalidCarryOverType",
    "NoEligibleSpeaker",
    "SenderRequired",
    "UndefinedNextAgent",
]


class AgentNameConflict(Exception):  # noqa: N818
    def __init__(self, msg: str = "Found multiple agents with the same name.", *args: Any, **kwargs: Any):
        super().__init__(msg, *args, **kwargs)


class NoEligibleSpeaker(Exception):  # noqa: N818
    """Exception raised for early termination of a GroupChat."""

    def __init__(self, message: str = "No eligible speakers."):
        self.message = message
        super().__init__(self.message)


class SenderRequired(Exception):  # noqa: N818
    """Exception raised when the sender is required but not provided."""

    def __init__(self, message: str = "Sender is required but not provided."):
        self.message = message
        super().__init__(self.message)


class InvalidCarryOverType(Exception):  # noqa: N818
    """Exception raised when the carryover type is invalid."""

    def __init__(
        self, message: str = "Carryover should be a string or a list of strings. Not adding carryover to the message."
    ):
        self.message = message
        super().__init__(self.message)


class UndefinedNextAgent(Exception):  # noqa: N818
    """Exception raised when the provided next agents list does not overlap with agents in the group."""

    def __init__(self, message: str = "The provided agents list does not overlap with agents in the group."):
        self.message = message
        super().__init__(self.message)


class ModelToolNotSupportedError(Exception):
    """
    Exception raised when attempting to use tools with models that do not support them.
    """

    def __init__(
        self,
        model: str,
    ):
        self.message = f"Tools are not supported with {model} models. Refer to the documentation at https://platform.openai.com/docs/guides/reasoning#limitations"
        super().__init__(self.message)
