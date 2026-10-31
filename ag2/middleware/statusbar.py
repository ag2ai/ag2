# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from copy import deepcopy
from functools import partial
from inspect import isroutine
from typing import Any, TypeAlias, overload

from fast_depends.core import CallModel

from ag2.annotations import Context
from ag2.events import BaseEvent, Input, ModelRequest, ModelResponse
from ag2.utils import CONTEXT_OPTION_NAME, build_model

from .base import BaseMiddleware, LLMCall, MiddlewareFactory
from .describe import MiddlewareDescription

__all__ = ("statusbar_middleware",)

_StatusResult: TypeAlias = Input | Sequence[Input] | None
_StatusCallback: TypeAlias = Callable[..., _StatusResult | Awaitable[_StatusResult]]


@overload
def statusbar_middleware(func: _StatusCallback, *, persist: bool = False) -> MiddlewareFactory: ...


@overload
def statusbar_middleware(
    func: None = None, *, persist: bool = False
) -> Callable[[_StatusCallback], MiddlewareFactory]: ...


def statusbar_middleware(
    func: _StatusCallback | None = None, *, persist: bool = False
) -> MiddlewareFactory | Callable[[_StatusCallback], MiddlewareFactory]:
    """Append freshly computed Input parts to each model request using AG2 DI.

    The callback can be sync or async and return one Input, an ordered sequence
    of Inputs, or None. Dynamic parts are never published to conversation history.
    Register this middleware before RetryMiddleware to reuse one snapshot across
    its provider attempts. Dependencies stay open through the wrapped model call.
    """
    if persist:
        raise ValueError("Persistent status middleware is not available yet")
    if func is None:
        return partial(statusbar_middleware, persist=persist)
    return _StatusBarFactory(func)


class _StatusBarFactory:
    def __init__(self, func: _StatusCallback) -> None:
        self.model = build_model(func, serialize_result=False)
        source = func if isroutine(func) else type(func)
        self.source = f"{source.__module__}.{source.__qualname__}"

    def describe(self) -> MiddlewareDescription:
        return MiddlewareDescription(kind="statusbar_middleware", config={"persist": False, "callback": self.source})

    def __call__(self, event: BaseEvent, context: Context) -> BaseMiddleware:
        return _StatusBarMiddleware(event, context, model=self.model)


class _StatusBarMiddleware(BaseMiddleware):
    def __init__(self, event: BaseEvent, context: Context, *, model: CallModel) -> None:
        super().__init__(event, context)
        self.model = model

    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        async with AsyncExitStack() as stack:
            injected: dict[str, Any] = {CONTEXT_OPTION_NAME: context}
            result = await self.model.asolve(
                stack=stack,
                cache_dependencies={},
                dependency_provider=context.dependency_provider,
                **injected,
            )
            prepared = list(events)
            parts = _status_parts(result)
            if parts:
                prepared.append(ModelRequest(parts))
            return await call_next(prepared, context)


def _status_parts(result: Any) -> list[Input]:
    if result is None:
        return []
    if isinstance(result, Input):
        return [deepcopy(result)]
    if (
        isinstance(result, Sequence)
        and not isinstance(result, (str, bytes, bytearray))
        and all(isinstance(part, Input) for part in result)
    ):
        return deepcopy(list(result))
    raise TypeError("Status callback must return Input, Sequence[Input], or None")
