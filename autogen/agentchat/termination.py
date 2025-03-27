# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from typing import Any, Protocol

from ..code_utils import content_str


class TerminateProtocol(Protocol):
    def is_termination_message(self, message: dict[str, Any]) -> bool: ...


class TerminationCondition(TerminateProtocol):
    def __or__(self, other: "TerminationCondition") -> "OrCondition":
        return OrCondition(self, other)

    def __and__(self, other: "TerminationCondition") -> "AndCondition":
        return AndCondition(self, other)

    def __invert__(self) -> "NotCondition":
        return NotCondition(self)


class OrCondition(TerminationCondition):
    def __init__(self, cond1: TerminationCondition, cond2: TerminationCondition):
        self.cond1 = cond1
        self.cond2 = cond2

    def is_termination_message(self, message: dict[str, Any]) -> bool:
        return self.cond1.is_termination_message(message) or self.cond2.is_termination_message(message)


class AndCondition(TerminationCondition):
    def __init__(self, cond1: TerminationCondition, cond2: TerminationCondition):
        self.cond1 = cond1
        self.cond2 = cond2

    def is_termination_message(self, message: dict[str, Any]) -> bool:
        return self.cond1.is_termination_message(message) and self.cond2.is_termination_message(message)


class NotCondition(TerminationCondition):
    def __init__(self, condition: TerminationCondition):
        self.condition = condition

    def is_termination_message(self, message: dict[str, Any]) -> bool:
        return not self.condition.is_termination_message(message)


class Keyword(TerminationCondition):
    def __init__(self, keyword: str):
        self.keyword = keyword

    def is_termination_message(self, message: dict[str, Any]) -> bool:
        return content_str(message.get("content")) == self.keyword


class MaxTurns(TerminationCondition):
    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        self._turns = 0

    def is_termination_message(self, message: dict[str, Any]) -> bool:
        self._turns += 1
        return self._turns >= self.max_turns
