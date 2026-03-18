# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import AsyncExitStack
from typing import Any, TypeAlias, cast, overload

from fast_depends import Provider
from pydantic import BaseModel, TypeAdapter, create_model
from typing_extensions import TypeVar as TypeVar313

from autogen.beta.annotations import Context
from autogen.beta.utils import CONTEXT_OPTION_NAME, build_model

from .proto import ResponseProto

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
) -> CallableResponse[T]: ...


@overload
def response_schema(
    func: Callable[..., T],
    *,
    name: str | None = None,
    description: str | None = None,
    schema: dict[str, Any] | None = None,
    sync_to_thread: bool = True,
) -> CallableResponse[T]: ...


@overload
def response_schema(
    func: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: dict[str, Any] | None = None,
    sync_to_thread: bool = True,
) -> Callable[[ResponseHook[T]], CallableResponse[T]]: ...


def response_schema(
    func: ResponseHook[T] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: dict[str, Any] | None = None,
    sync_to_thread: bool = True,
) -> Callable[[ResponseHook[T]], CallableResponse[T]] | CallableResponse[T]:
    def make_callable_schema(f: ResponseHook[T]) -> CallableResponse[T]:
        final_name = name or f.__name__

        validator, json_schema = _unwrap_message_to_fast_depends_decorator(f, final_name, sync_to_thread)

        return CallableResponse[T](
            validator,
            name=final_name,
            description=description or f.__doc__ or "",
            schema=schema or json_schema,
        )

    if func:
        return make_callable_schema(func)
    return make_callable_schema


def _unwrap_message_to_fast_depends_decorator(
    func: ResponseHook[T],
    name: str,
    sync_to_thread: bool = True,
) -> tuple[ResponseValidator[T], dict[str, Any] | None]:
    model = build_model(func, sync_to_thread=sync_to_thread)

    async def execute(*args: Any, _dep_provider_: "Provider | None", **kwargs: Any) -> T:
        async with AsyncExitStack() as stack:
            return await model.asolve(
                *args,
                stack=stack,
                cache_dependencies={},
                dependency_provider=_dep_provider_,
                **kwargs,
            )

    dependant_params = model.flat_params

    if len(dependant_params) <= 1:
        if option := next(iter(dependant_params), None):
            is_multi_params = option.kind is inspect.Parameter.VAR_POSITIONAL
        else:
            is_multi_params = False
    else:
        is_multi_params = True

    schema: dict[str, Any] | None = None
    if is_multi_params:
        pydantic_model = cast(
            type[BaseModel],
            create_model(name, **{i.field_name: (i.field_type, i.default_value) for i in dependant_params}),
        )

        schema = pydantic_model.model_json_schema()

        async def decode_wrapper(
            message: str,
            context: "Context",
            provider: "Provider | None",
        ) -> T:
            content = pydantic_model.model_validate_json(message)
            return await execute(
                _dep_provider_=provider,
                **(content.model_dump() | {CONTEXT_OPTION_NAME: context}),
            )

    else:
        type_ = dependant_params[0].field_type

        if type_ is str:

            async def decode_wrapper(
                message: str,
                context: "Context",
                provider: "Provider | None",
            ) -> T:
                return await execute(
                    message,
                    _dep_provider_=provider,
                    **{CONTEXT_OPTION_NAME: context},
                )

        else:
            adapter = TypeAdapter(type_)
            schema = adapter.json_schema()

            async def decode_wrapper(
                message: str,
                context: "Context",
                provider: "Provider | None",
            ) -> T:
                return await execute(
                    adapter.validate_json(message),
                    _dep_provider_=provider,
                    **{CONTEXT_OPTION_NAME: context},
                )

    return decode_wrapper, schema
