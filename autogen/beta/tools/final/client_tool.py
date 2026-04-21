# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from functools import partial
from typing import Any

from autogen.beta.annotations import Context
from autogen.beta.events import ClientToolCallEvent, ToolCallEvent
from autogen.beta.middleware import BaseMiddleware, ToolExecution
from autogen.beta.tools.tool import Tool

from .function_tool import FunctionToolSchema


class ClientTool(Tool):
    __slots__ = (
        "schema",
        "name",
    )

    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = FunctionToolSchema.from_dict(schema)
        self.name = self.schema.function.name

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

        async def execute(event: "ToolCallEvent", context: "Context") -> None:
            result = await execution(event, context)
            await context.send(result)

        stack.enter_context(
            context.stream.where(
                (ToolCallEvent.name == self.schema.function.name) & ClientToolCallEvent.not_()
            ).sub_scope(execute),
        )

    async def __call__(self, event: "ToolCallEvent", context: "Context") -> "ClientToolCallEvent":
        return ClientToolCallEvent.from_call(event)
