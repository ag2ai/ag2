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


class Keyword:
    def __init__(self, keyword: str):
        self.keyword = keyword

    def is_termination_message(self, message: dict[str, Any]) -> bool:
        return content_str(message.get("content")) == self.keyword
