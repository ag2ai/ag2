# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol, runtime_checkable

from ..tools import Tool

__all__ = ["Interoperable"]


@runtime_checkable
class Interoperable(Protocol):
    def convert_tool(self, tool: Any, **kwargs: Any) -> Tool: ...
