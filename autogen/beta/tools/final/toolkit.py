# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from contextlib import ExitStack
from typing import Any, overload

from fast_depends import Provider

from autogen.beta.annotations import Context
from autogen.beta.exceptions import ToolConflictError
from autogen.beta.middleware import BaseMiddleware, ToolMiddleware
from autogen.beta.tools.schemas import ToolSchema
from autogen.beta.tools.tool import Tool

from .function_tool import FunctionParameters, FunctionTool, tool


class Toolkit(Tool):
    __slots__ = (
        "name",
        "_tools",
        "_middleware",
    )

    def __init__(
        self,
        *tools: Tool | Callable[..., Any],
        name: str | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self._middleware: tuple[ToolMiddleware, ...] = tuple(middleware)
        self.name = name or self.__class__.__name__

        self._tools: dict[str, Tool] = {}
        for t in tools:
            self._add_tool(t)

    @property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools.values())

    def set_provider(self, provider: Provider) -> None:
        for t in self.tools:
            t.set_provider(provider)

    def _add_tool(self, tool: Tool | Callable[..., Any], *, unsafe: bool = False) -> None:
        t = FunctionTool.ensure_tool(tool).with_middleware(*self._middleware)

        if not unsafe and t.name in self._tools:
            raise ToolConflictError(t.name)

        self._tools[t.name] = t

    def __or__(self, other: Any) -> "Toolkit":
        if isinstance(other, Toolkit):
            tools = self._tools | other._tools
        else:
            tool = FunctionTool.ensure_tool(other)
            tools = self._tools | {tool.name: tool}

        return Toolkit(*tools.values(), name=self.name, middleware=self._middleware)

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
            t = tool(
                f,
                name=name,
                description=description,
                schema=schema,
                sync_to_thread=sync_to_thread,
                middleware=middleware,
            )
            self._add_tool(t)
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
