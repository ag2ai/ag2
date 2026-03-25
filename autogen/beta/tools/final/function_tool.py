# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from contextlib import AsyncExitStack, ExitStack
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Any, TypeAlias, overload

from fast_depends import Provider
from fast_depends.core import CallModel
from fast_depends.pydantic.schema import get_schema

from autogen.beta.annotations import Context
from autogen.beta.events.tool_events import ToolCallEvent, ToolErrorEvent, ToolResult, ToolResultEvent
from autogen.beta.middleware import BaseMiddleware, ToolExecution
from autogen.beta.tools.schemas import ToolSchema
from autogen.beta.tools.tool import Tool
from autogen.beta.utils import CONTEXT_OPTION_NAME, build_model

FunctionParameters: TypeAlias = dict[str, Any]


@dataclass(slots=True)
class FunctionDefinition:
    name: str
    description: str = ""
    parameters: FunctionParameters = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.parameters.pop("title", None)


@dataclass(slots=True)
class FunctionToolSchema(ToolSchema):
    type: str = field(default="function", init=False)
    function: FunctionDefinition = field(default_factory=lambda: FunctionDefinition(name=""))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunctionToolSchema":
        func_data = data.get("function", {})
        return cls(function=FunctionDefinition(**func_data))


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

        self.schema = FunctionToolSchema(
            function=FunctionDefinition(
                name=name,
                description=description,
                parameters=schema,
            )
        )

        self.provider: Provider | None = None

    async def schemas(self, context: "Context") -> list[FunctionToolSchema]:
        return [self.schema]

    @staticmethod
    def ensure_tool(
        func: "Tool | Callable[..., Any]",
        *,
        provider: Provider | None = None,
    ) -> "FunctionTool":
        t = deepcopy(func) if isinstance(func, Tool) else tool(func)
        t.provider = provider
        return t

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

        stack.enter_context(context.stream.where(ToolCallEvent.name == self.schema.function.name).sub_scope(execute))

    async def __call__(self, event: "ToolCallEvent", context: "Context") -> "ToolResultEvent":
        try:
            async with AsyncExitStack() as stack:
                result = await self.model.asolve(
                    **(event.serialized_arguments | {CONTEXT_OPTION_NAME: context}),
                    stack=stack,
                    cache_dependencies={},
                    dependency_provider=self.provider,
                )

            return ToolResultEvent(
                parent_id=event.id,
                name=event.name,
                result=ToolResult.ensure_result(result),
            )

        except Exception as e:
            return ToolErrorEvent(
                parent_id=event.id,
                name=event.name,
                error=e,
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
