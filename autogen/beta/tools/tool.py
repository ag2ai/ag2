import json
from collections.abc import Callable
from contextlib import AsyncExitStack
from functools import wraps
from typing import Any, overload

from fast_depends import Provider, dependency_provider
from fast_depends.core import CallModel, build_call_model
from fast_depends.pydantic import PydanticSerializer
from fast_depends.pydantic.schema import get_schema
from fast_depends.utils import is_coroutine_callable, run_in_threadpool

from autogen.beta.stream import Context

from .schemas import FunctionDefinition, FunctionParameters, FunctionTool

CONTEXT_OPTION_NAME = "ctx"


class Tool:
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
        self.schema = FunctionTool(
            function=FunctionDefinition(
                name=self.name,
                description=self.description,
                parameters=schema,
            )
        )

    @staticmethod
    def ensure_tool(
        func: "Tool | Callable[..., Any]",
        *,
        provider: Provider | None = None,
    ) -> "Tool":
        t = func if isinstance(func, Tool) else tool(func)
        if provider:
            t.model.dependency_provider = provider
        return t

    async def execute(self, arguments: str, ctx: Context) -> bytes:
        async with AsyncExitStack() as stack:
            result = await self.model.asolve(
                **(json.loads(arguments) | {CONTEXT_OPTION_NAME: ctx}),
                stack=stack,
                cache_dependencies={},
            )
            return PydanticSerializer.encode(result)


@overload
def tool(
    function: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    schema: FunctionParameters | None = None,
    sync_to_thread: bool = True,
) -> Tool: ...


@overload
def tool(
    function: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: FunctionParameters | None = None,
    sync_to_thread: bool = True,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: FunctionParameters | None = None,
    sync_to_thread: bool = True,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    def make_tool(f: Callable[..., Any]) -> Tool:
        call_model = build_call_model(
            _to_async(f, sync_to_thread=sync_to_thread),
            dependency_provider=dependency_provider,
            serializer_cls=PydanticSerializer(
                pydantic_config={"arbitrary_types_allowed": True},
                use_fastdepends_errors=False,
            ),
        )

        return Tool(
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


def _to_async(
    func: Callable[..., Any],
    *,
    sync_to_thread: bool = True,
) -> Callable[..., Any]:
    if is_coroutine_callable(func):
        return func

    if sync_to_thread:

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await run_in_threadpool(func, *args, **kwargs)

    else:

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

    return async_wrapper
