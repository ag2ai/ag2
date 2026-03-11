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

from .schemas import FunctionToolSchema
from .tool import Tool


class ClientTool(Tool):
    __slots__ = (
        "schema",
        "name",
    )

    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = FunctionToolSchema.model_validate(schema)
        self.name = self.schema.function.name

    def register(
        self,
        stack: "ExitStack",
        ctx: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        execution: ToolExecution = self
        for mw in middleware:
            execution = partial(mw.on_tool_execution, execution)

        async def execute(event: "ToolCall", ctx: "Context") -> None:
            return await execution(event, ctx)

        stack.enter_context(
            ctx.stream.where((ToolCall.name == self.name) & ClientToolCall.not_()).sub_scope(execute),
        )

    async def __call__(self, event: "ToolCall", ctx: "Context") -> "ClientToolCall":
        return ClientToolCall.from_call(event)
