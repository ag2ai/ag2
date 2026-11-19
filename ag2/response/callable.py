# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import AsyncExitStack
from functools import wraps
from typing import Any, TypeAlias, cast, overload

from fast_depends import Provider
from fast_depends.utils import is_coroutine_callable
from pydantic import create_model
from typing_extensions import TypeVar as TypeVar313

from ag2.annotations import Context
from ag2.utils import CONTEXT_OPTION_NAME, build_model

from .proto import ResponseProto
from .schema import ResponseSchema

T = TypeVar313("T", default=str)

ResponseValidator: TypeAlias = Callable[
    [str, "Context", "Provider | None"],
    Awaitable[T],
]

ResponseHook: TypeAlias = Callable[..., Coroutine[Any, Any, T]] | Callable[..., T]


class CallableResponse(ResponseProto[T]):
    def __init__(
        self,
        validator: ResponseValidator[T],
        *,
        name: str,
        schema: dict[str, Any] | None,
        description: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.json_schema = schema
        self.system_prompt = None
        self.__execute = validator

    async def validate(
        self,
        content: str,
        context: "Context",
        provider: "Provider | None" = None,
    ) -> T:
        return await self.__execute(content, context, provider)


@overload
def response_schema(
    func: Callable[..., Coroutine[Any, Any, T]],
    *,
    name: str | None = None,
    description: str | None = None,
    schema: dict[str, Any] | None = None,
    sync_to_thread: bool = True,
    embed: bool = True,
) -> CallableResponse[T]: ...


@overload
def response_schema(
    func: Callable[..., T],
    *,
    name: str | None = None,
    description: str | None = None,
    schema: dict[str, Any] | None = None,
    sync_to_thread: bool = True,
    embed: bool = True,
) -> CallableResponse[T]: ...


@overload
def response_schema(
    func: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: dict[str, Any] | None = None,
    sync_to_thread: bool = True,
    embed: bool = True,
) -> Callable[[ResponseHook[T]], CallableResponse[T]]: ...


def response_schema(
    func: ResponseHook[T] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: dict[str, Any] | None = None,
    sync_to_thread: bool = True,
    embed: bool = True,
) -> Callable[[ResponseHook[T]], CallableResponse[T]] | CallableResponse[T]:
    def make_callable_schema(f: ResponseHook[T]) -> CallableResponse[T]:
        final_name = name or f.__name__

        validator, response_schema = _unwrap_message_to_fast_depends_decorator(
            f,
            final_name,
            sync_to_thread,
            embed,
        )

        return CallableResponse[T](
            validator,
            name=final_name,
            description=description or f.__doc__,
            schema=schema or response_schema.json_schema,
        )

    if func is not None:
        return make_callable_schema(func)
    return make_callable_schema


def _unwrap_message_to_fast_depends_decorator(
    func: ResponseHook[T],
    name: str,
    sync_to_thread: bool = True,
    embed: bool = True,
) -> tuple[ResponseValidator[T], ResponseSchema[Any]]:
    """Wrap ``func`` as a response validator. The returned schema describes the
    model's reply, which is ``func``'s input — not its ``T`` result."""
    model = build_model(_adapt_positional_only(func), sync_to_thread=sync_to_thread)

    async def execute(*args: Any, _dep_provider_: "Provider | None", **kwargs: Any) -> T:
        async with AsyncExitStack() as stack:
            # A `ResponseHook[T]` returns `T`; `asolve` itself is untyped.
            result: T = await model.asolve(
                *args,
                stack=stack,
                cache_dependencies={},
                dependency_provider=_dep_provider_,
                **kwargs,
            )
        return result

    dependant_params = model.flat_params

    if len(dependant_params) <= 1:
        if option := next(iter(dependant_params), None):
            is_multi_params = option.kind is inspect.Parameter.VAR_POSITIONAL
        else:
            is_multi_params = False
    else:
        is_multi_params = True

    schema: ResponseSchema[Any]
    if is_multi_params:
        fields: dict[str, Any] = {i.field_name: (i.field_type, i.default_value) for i in dependant_params}
        model_schema = ResponseSchema(create_model(name, **fields), embed=embed)
        schema = model_schema

        async def decode_wrapper(
            message: str,
            context: "Context",
            provider: "Provider | None",
        ) -> T:
            content = await model_schema.validate(message, context, provider)
            return await execute(
                _dep_provider_=provider,
                **(content.model_dump() | {CONTEXT_OPTION_NAME: context}),
            )

    else:
        schema = ResponseSchema(dependant_params[0].field_type, embed=embed)

        async def decode_wrapper(
            message: str,
            context: "Context",
            provider: "Provider | None",
        ) -> T:
            return await execute(
                await schema.validate(message, context, provider),
                _dep_provider_=provider,
                **{CONTEXT_OPTION_NAME: context},
            )

    return decode_wrapper, schema


def _adapt_positional_only(func: ResponseHook[T]) -> ResponseHook[T]:
    """Adapt Python's positional-only arguments to the dependency solver's keyword calls."""
    signature = inspect.signature(func)
    positional_names = tuple(
        param.name for param in signature.parameters.values() if param.kind is inspect.Parameter.POSITIONAL_ONLY
    )
    if not positional_names:
        return func

    @wraps(func)
    def sync_adapter(*extra: Any, **kwargs: Any) -> T:
        args = [kwargs.pop(name) for name in positional_names]
        return cast(Callable[..., T], func)(*args, *extra, **kwargs)

    @wraps(func)
    async def async_adapter(*extra: Any, **kwargs: Any) -> T:
        args = [kwargs.pop(name) for name in positional_names]
        return await cast(Callable[..., Coroutine[Any, Any, T]], func)(*args, *extra, **kwargs)

    adapter: ResponseHook[T] = async_adapter if is_coroutine_callable(func) else sync_adapter

    setattr(
        adapter,
        "__signature__",
        signature.replace(
            parameters=[
                param.replace(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD) if param.name in positional_names else param
                for param in signature.parameters.values()
            ]
        ),
    )
    return adapter
