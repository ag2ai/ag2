# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from contextlib import ExitStack
from typing import Any, overload

from autogen.beta.annotations import Context
from autogen.beta.middleware import BaseMiddleware, ToolMiddleware
from autogen.beta.tools.schemas import ToolSchema
from autogen.beta.tools.tool import Tool

from .function_tool import FunctionParameters, FunctionTool, tool


class Toolkit(Tool):
    def __init__(
        self,
        *tools: Tool | Callable[..., Any],
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self._middleware: tuple[ToolMiddleware, ...] = tuple(middleware)
        self.tools: list[FunctionTool] = [FunctionTool.ensure_tool(t).with_middleware(*self._middleware) for t in tools]

    @overload
    def tool(
        self,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool: ...

    @overload
    def tool(
        self,
        function: None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> Callable[[Callable[..., Any]], FunctionTool]: ...

    def tool(
        self,
        function: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
        def make_tool(f: Callable[..., Any]) -> FunctionTool:
            t = FunctionTool.ensure_tool(
                tool(
                    f,
                    name=name,
                    description=description,
                    schema=schema,
                    sync_to_thread=sync_to_thread,
                    middleware=middleware,
                )
            ).with_middleware(*self._middleware)
            self.tools.append(t)
            return t

        if function:
            return make_tool(function)

        return make_tool

    async def schemas(self, context: "Context") -> Iterable[ToolSchema]:
        schemas: list[ToolSchema] = []
        for t in self.tools:
            schemas.extend(await t.schemas(context))
        return schemas

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        for t in self.tools:
            t.register(stack, context, middleware=middleware)
