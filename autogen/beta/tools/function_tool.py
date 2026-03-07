# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable
from contextlib import AsyncExitStack, ExitStack
from traceback import format_exc
from typing import Any, overload

from fast_depends import Provider
from fast_depends.core import CallModel
from fast_depends.pydantic import PydanticSerializer
from fast_depends.pydantic.schema import get_schema

from autogen.beta.annotations import Context
from autogen.beta.events import ToolCall, ToolError, ToolResult
from autogen.beta.utils import CONTEXT_OPTION_NAME, build_model

from .schemas import FunctionDefinition, FunctionParameters, FunctionToolSchema
from .tool import Tool


class FunctionTool(Tool):
    def __init__(
        self,
        model: CallModel,
        *,
        name: str,
        description: str,
        schema: FunctionParameters,
    ) -> None:
        self.model = model

        self.name = name
        self.description = description
        self.schema = FunctionToolSchema(
            function=FunctionDefinition(
                name=self.name,
                description=self.description,
                parameters=schema,
            )
        )

        self.provider: Provider | None = None

    @staticmethod
    def ensure_tool(
        func: "Tool | Callable[..., Any]",
        *,
        provider: Provider | None = None,
    ) -> "FunctionTool":
        t = func if isinstance(func, Tool) else tool(func)
        t.provider = provider
        return t

    def register(self, stack: "ExitStack", ctx: "Context") -> None:
        stack.enter_context(ctx.stream.where(ToolCall.name == self.name).sub_scope(self.execute))

    async def execute(self, event: ToolCall, ctx: "Context") -> None:
        try:
            async with AsyncExitStack() as stack:
                result = await self.model.asolve(
                    **(json.loads(event.arguments) | {CONTEXT_OPTION_NAME: ctx}),
                    stack=stack,
                    cache_dependencies={},
                    dependency_provider=self.provider,
                )
                result = PydanticSerializer.encode(result)

        except Exception as e:
            await ctx.send(
                ToolError(
                    parent_id=event.id,
                    name=event.name,
                    content=format_exc(limit=3),
                    error=e,
                )
            )

        else:
            await ctx.send(
                ToolResult(
                    parent_id=event.id,
                    name=event.name,
                    content=result.decode(),
                )
            )


@overload
def tool(
    function: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    schema: FunctionParameters | None = None,
    sync_to_thread: bool = True,
) -> FunctionTool: ...


@overload
def tool(
    function: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: FunctionParameters | None = None,
    sync_to_thread: bool = True,
) -> Callable[[Callable[..., Any]], FunctionTool]: ...


def tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: FunctionParameters | None = None,
    sync_to_thread: bool = True,
) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
    def make_tool(f: Callable[..., Any]) -> FunctionTool:
        call_model = build_model(f, sync_to_thread=sync_to_thread)

        return FunctionTool(
            call_model,
            name=name or f.__name__,
            description=description or f.__doc__ or "",
            schema=schema
            or get_schema(
                call_model,
                exclude=(CONTEXT_OPTION_NAME,),
            ),
        )

    if function:
        return make_tool(function)
    return make_tool
