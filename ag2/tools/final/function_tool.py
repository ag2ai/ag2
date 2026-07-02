# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from contextlib import AsyncExitStack, ExitStack
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, TypeAlias, overload

from fast_depends.core import CallModel
from fast_depends.pydantic.schema import get_schema

from ag2.annotations import Context
from ag2.events import ToolCallEvent, ToolErrorEvent, ToolResultEvent
from ag2.middleware import BaseMiddleware, ToolExecution, ToolMiddleware, ToolResultType
from ag2.tools.schemas import ToolSchema
from ag2.tools.tool import Tool
from ag2.utils import CONTEXT_OPTION_NAME, build_model

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
    defer_loading: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunctionToolSchema":
        func_data = data.get("function", {})
        return cls(function=FunctionDefinition(**func_data), defer_loading=data.get("defer_loading", False))


class FunctionTool(Tool):
    __slots__ = (
        "model",
        "name",
        "schema",
        "_middleware",
    )

    def __init__(
        self,
        model: CallModel,
        *,
        name: str,
        description: str,
        schema: FunctionParameters,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self.model = model
        self._middleware: tuple[ToolMiddleware, ...] = tuple(middleware)

        self.schema = FunctionToolSchema(
            function=FunctionDefinition(
                name=name,
                description=description,
                parameters=schema,
            ),
        )

        self.name = name

    def with_middleware(self, *middleware: ToolMiddleware) -> "FunctionTool":
        """Return a new FunctionTool with additional middleware appended.

        Does not modify the original tool.
        """
        cloned = deepcopy(self)
        cloned._middleware = tuple(middleware) + self._middleware
        return cloned

    async def schemas(self, context: "Context") -> list[FunctionToolSchema]:
        return [self.schema]

    @staticmethod
    def ensure_tool(func: "Tool | Callable[..., Any]") -> "Tool":
        return deepcopy(func) if isinstance(func, Tool) else tool(func)

    def register(
        self,
        stack: "ExitStack | AsyncExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        execution: ToolExecution = self
        for hook in reversed(self._middleware):
            execution = _wrap_middleware(hook, execution)
        for mw in middleware:
            execution = _wrap_middleware(mw.on_tool_execution, execution)

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
                    dependency_provider=context.dependency_provider,
                )

            return ToolResultEvent.from_call(event, result=result)

        except Exception as e:
            return ToolErrorEvent.from_call(event, error=e)


@overload
def tool(
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
    function: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: FunctionParameters | None = None,
    sync_to_thread: bool = True,
    middleware: Iterable[ToolMiddleware] = (),
) -> Callable[[Callable[..., Any]], FunctionTool]: ...


def tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: FunctionParameters | None = None,
    sync_to_thread: bool = True,
    middleware: Iterable[ToolMiddleware] = (),
) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
    def make_tool(f: Callable[..., Any]) -> FunctionTool:
        call_model = build_model(
            f,
            sync_to_thread=sync_to_thread,
            serialize_result=False,
        )

        return FunctionTool(
            call_model,
            name=name or f.__name__,
            description=description or f.__doc__ or "",
            schema=_normalize_parameters(schema, call_model),
            middleware=middleware,
        )

    if function:
        return make_tool(function)
    return make_tool


def _normalize_parameters(schema: FunctionParameters | None, call_model: CallModel) -> FunctionParameters:
    if schema is not None:
        return schema
    # A no-arg callable yields {"type": "null"}, invalid as a JSON Schema object for tool
    # `parameters` for some LLM providers.
    generated = get_schema(call_model, exclude=(CONTEXT_OPTION_NAME,))
    if generated.get("type") != "object":
        return {"type": "object", "properties": {}}
    return generated


def _wrap_middleware(hook: "ToolMiddleware", inner: "ToolExecution") -> "ToolExecution":
    async def call(event: "ToolCallEvent", context: "Context") -> "ToolResultType":
        return await hook(inner, event, context)

    return call
