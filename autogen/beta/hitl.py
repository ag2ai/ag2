# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable, Iterable
from contextlib import AsyncExitStack
from functools import partial
from typing import TypeAlias

from .annotations import Context
from .events import HumanInputRequest, HumanMessage
from .exceptions import HumanInputNotProvidedError
from .middleware.base import BaseMiddleware, HumanInputHook
from .utils import CONTEXT_OPTION_NAME, build_model

HumanHook: TypeAlias = Callable[..., HumanMessage] | Callable[..., Awaitable[HumanMessage]]

HitlExecution: TypeAlias = Callable[[HumanInputRequest, Context], Awaitable[None]]


def wrap_hitl(
    func: HumanHook,
) -> Callable[[Iterable["BaseMiddleware"]], HitlExecution]:
    call_model = build_model(func)

    async def _call_model(event: HumanInputRequest, context: Context) -> HumanMessage:
        async with AsyncExitStack() as stack:
            event = await call_model.asolve(
                event,
                stack=stack,
                cache_dependencies={},
                dependency_provider=context.dependency_provider,
                **{CONTEXT_OPTION_NAME: context},
            )
        return event

    def make_hook(middlewares: Iterable["BaseMiddleware"]) -> HitlExecution:
        ask_user: HumanInputHook = _call_model
        for middleware in middlewares:
            ask_user = partial(middleware.on_human_input, ask_user)

        async def wrapper(event: HumanInputRequest, context: Context) -> None:
            event = await ask_user(event, context)
            await context.send(event)

        return wrapper

    return make_hook


def default_hitl_hook(middlewares: Iterable["BaseMiddleware"]) -> HitlExecution:
    async def _call_model(event: HumanInputRequest, context: Context) -> None:
        raise HumanInputNotProvidedError

    return _call_model
