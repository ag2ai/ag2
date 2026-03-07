# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack
from typing import Protocol, runtime_checkable

from autogen.beta.annotations import Context
from autogen.beta.events import ToolCall

from .schemas import FunctionToolSchema


@runtime_checkable
class Tool(Protocol):
    name: str
    schema: FunctionToolSchema

    def register(self, stack: "ExitStack", ctx: "Context") -> None:
        pass

    async def execute(self, event: "ToolCall", ctx: "Context") -> None:
        """ToolCall event interruptor to execute tool.

        Returns None to suppress futher event processing.
        """
        ...
