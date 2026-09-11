# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable, Iterable
from contextlib import AsyncExitStack
from functools import partial
from typing import Literal, TypeAlias

from .annotations import Context
from .events import HumanInputRequest, HumanMessage
from .exceptions import HumanInputNotProvidedError
from .middleware.base import BaseMiddleware, HumanInputHook
from .utils import CONTEXT_OPTION_NAME, build_model

HumanHook: TypeAlias = (
    Callable[..., HumanMessage]
    | Callable[..., Awaitable[HumanMessage]]
    | Callable[..., str]
    | Callable[..., Awaitable[str]]
)

HitlExecution: TypeAlias = Callable[[HumanInputRequest, Context], Awaitable[None]]

# Whether a protocol peer that can put a question to *our* human may be asked, or
# is refused outright. Deliberately two-valued, unlike a permission policy: a
# permission request carries an allow option a peer can pick blind, whereas an
# arbitrary elicitation form has no answer AG2 could invent without fabricating
# data on the user's behalf. So there is no ``"auto"`` — only ask a human, or
# decline.
#
# Defined here, in core, because both protocol integrations need the same word:
# ``ag2.acp`` answering an ACP agent's ``elicitation/create``, and ``ag2.mcp``
# deciding whether a served agent may ask its MCP client. One vocabulary, learnt
# once — a second alias with the same two values would be free to drift.
ElicitationPolicy = Literal["ask", "decline"]


def wrap_hitl(
    func: HumanHook,
) -> Callable[[Iterable["BaseMiddleware"]], HitlExecution]:
    call_model = build_model(func)

    async def _call_model(event: HumanInputRequest, context: Context) -> HumanMessage:
        # Nothing is caught here: a hook that raises is the channel failing, and
        # ``Context.input`` — the only way in — turns that into a
        # ``HumanInputError`` on the way back out.
        async with AsyncExitStack() as stack:
            result = await call_model.asolve(
                event,
                stack=stack,
                cache_dependencies={},
                dependency_provider=context.dependency_provider,
                **{CONTEXT_OPTION_NAME: context},
            )

        return HumanMessage.ensure_message(result, parent_id=event.id)

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
