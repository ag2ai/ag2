# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from typing import TypeAlias

from .annotations import Context
from .events import HumanInputRequest, HumanMessage
from .exceptions import HumanInputNotProvidedError
from .utils import CONTEXT_OPTION_NAME, build_model

HumanHook: TypeAlias = Callable[..., HumanMessage] | Callable[..., Awaitable[HumanMessage]]


def wrap_hitl(func: HumanHook) -> None:
    call_model = build_model(func)

    async def wrapper(event: HumanInputRequest, context: Context) -> None:
        async with AsyncExitStack() as stack:
            event = await call_model.asolve(
                event,
                stack=stack,
                cache_dependencies={},
                dependency_provider=context.dependency_provider,
                **{CONTEXT_OPTION_NAME: context},
            )
        await context.send(event)

    return wrapper


def default_hitl_hook() -> HumanMessage:
    raise HumanInputNotProvidedError
