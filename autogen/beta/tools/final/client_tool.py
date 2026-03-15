# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from functools import partial
from typing import Any

from autogen.beta.annotations import Context
from autogen.beta.events import ClientToolCall, ToolCall
from autogen.beta.middleware import BaseMiddleware, ToolExecution
from autogen.beta.tools.tool import Tool

from .function_tool import FunctionToolSchema


class ClientTool(Tool):
    __slots__ = ("schema",)

    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = FunctionToolSchema.from_dict(schema)

    async def schemas(self, context: "Context") -> list[FunctionToolSchema]:
        return [self.schema]

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        execution: ToolExecution = self
        for mw in middleware:
            execution = partial(mw.on_tool_execution, execution)

        async def execute(event: "ToolCall", context: "Context") -> None:
            return await execution(event, context)

        stack.enter_context(
            context.stream.where((ToolCall.name == self.schema.function.name) & ClientToolCall.not_()).sub_scope(
                execute
            ),
        )

    async def __call__(self, event: "ToolCall", context: "Context") -> "ClientToolCall":
        return ClientToolCall.from_call(event)
