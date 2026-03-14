# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from functools import partial
from typing import Any

from mcphero import MCPToolAdapterOpenAI

from autogen.beta.annotations import Context
from autogen.beta.events import ToolCall, ToolError, ToolResult
from autogen.beta.middleware import BaseMiddleware, ToolExecution

from .schemas import FunctionToolSchema
from .tool import Tool


class MCPTool(Tool):
    def __init__(self, schema: dict[str, Any], adapter: MCPToolAdapterOpenAI) -> None:
        self.schema = FunctionToolSchema.model_validate(schema)
        self.name = self.schema.function.name
        self._adapter = adapter

    def register(
        self,
        stack: ExitStack,
        context: Context,
        *,
        middleware: Iterable[BaseMiddleware] = (),
    ) -> None:
        execution: ToolExecution = self
        for mw in middleware:
            execution = partial(mw.on_tool_execution, execution)

        async def execute(event: ToolCall, context: Context) -> None:
            result = await execution(event, context)
            await context.send(result)

        stack.enter_context(context.stream.where(ToolCall.name == self.name).sub_scope(execute))

    async def __call__(self, event: ToolCall, context: Context) -> ToolResult:
        try:
            raw = await self._adapter.call_tool(event.name, event.serialized_arguments)
            return ToolResult(parent_id=event.id, name=event.name, raw_content=raw)
        except Exception as e:
            return ToolError(parent_id=event.id, name=event.name, error=e)
