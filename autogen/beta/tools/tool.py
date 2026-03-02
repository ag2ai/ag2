# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any, overload

from fast_depends import Provider
from fast_depends.core import CallModel
from fast_depends.pydantic import PydanticSerializer
from fast_depends.pydantic.schema import get_schema

from autogen.beta.context import Context
from autogen.beta.utils import CONTEXT_OPTION_NAME, build_model

from .schemas import FunctionDefinition, FunctionParameters, FunctionTool


class Tool:
    def __init__(
        self,
        model: CallModel,
        *,
        name: str,
        description: str,
        schema: FunctionParameters,
        strict: bool | None,
    ) -> None:
        self.model = model

        self.name = name
        self.description = description
        self.schema = FunctionTool(
            function=FunctionDefinition(
                name=self.name,
                description=self.description,
                parameters=schema,
                strict=strict,
            )
        )

        self.provider: Provider | None = None

    @staticmethod
    def ensure_tool(
        func: "Tool | Callable[..., Any]",
        *,
        provider: Provider | None = None,
    ) -> "Tool":
        t = func if isinstance(func, Tool) else tool(func)
        t.provider = provider
        return t

    async def execute(self, arguments: str, ctx: Context) -> bytes:
        async with AsyncExitStack() as stack:
            result = await self.model.asolve(
                **(json.loads(arguments) | {CONTEXT_OPTION_NAME: ctx}),
                stack=stack,
                cache_dependencies={},
                dependency_provider=self.provider,
            )
            return PydanticSerializer.encode(result)


@overload
def tool(
    function: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    schema: FunctionParameters | None = None,
    strict: bool | None = True,
    sync_to_thread: bool = True,
) -> Tool: ...


@overload
def tool(
    function: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: FunctionParameters | None = None,
    strict: bool | None = True,
    sync_to_thread: bool = True,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: FunctionParameters | None = None,
    strict: bool | None = True,
    sync_to_thread: bool = True,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    def make_tool(f: Callable[..., Any]) -> Tool:
        call_model = build_model(f, sync_to_thread=sync_to_thread)

        return Tool(
            call_model,
            name=name or f.__name__,
            description=description or f.__doc__ or "",
            schema=schema
            or get_schema(
                call_model,
                exclude=(CONTEXT_OPTION_NAME,),
            ),
            strict=strict,
        )

    if function:
        return make_tool(function)
    return make_tool
