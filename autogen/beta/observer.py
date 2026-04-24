# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Protocol, overload, runtime_checkable

from autogen.beta.types import ClassInfo

from .annotations import Context
from .events.conditions import Condition, TypeCondition

__all__ = (
    "Observer",
    "observer",
)


@runtime_checkable
class Observer(Protocol):
    def register(self, stack: ExitStack, context: Context) -> None: ...


@dataclass(slots=True, kw_only=True)
class BaseObserver:
    callback: Callable[..., Any]
    interrupt: bool = False
    sync_to_thread: bool = True

    def register(self, stack: ExitStack, context: Context) -> None:
        stack.enter_context(
            context.stream.sub_scope(
                self.callback,
                interrupt=self.interrupt,
                sync_to_thread=self.sync_to_thread,
            )
        )


@dataclass(slots=True)
class ConditionalObserver(BaseObserver):
    condition: Condition

    def register(self, stack: ExitStack, context: Context) -> None:
        stack.enter_context(
            context.stream.where(self.condition).sub_scope(
                self.callback,
                interrupt=self.interrupt,
                sync_to_thread=self.sync_to_thread,
            )
        )


@overload
def observer(
    condition: ClassInfo | Condition | None = None,
    callback: None = None,
    *,
    interrupt: bool = False,
    sync_to_thread: bool = True,
) -> Callable[[Callable[..., Any]], Observer]: ...


@overload
def observer(
    condition: ClassInfo | Condition | None = None,
    callback: Callable[..., Any] = ...,
    *,
    interrupt: bool = False,
    sync_to_thread: bool = True,
) -> Observer: ...


def observer(
    condition: ClassInfo | Condition | None = None,
    callback: Callable[..., Any] | None = None,
    *,
    interrupt: bool = False,
    sync_to_thread: bool = True,
) -> Observer | Callable[[Callable[..., Any]], Observer]:
    if condition is None:
        cond: Condition | None = None
    elif isinstance(condition, Condition):
        cond = condition
    else:
        cond = TypeCondition(condition)

    def decorator(func: Callable[..., Any]) -> Observer:
        if cond:
            return ConditionalObserver(
                condition=cond,
                callback=func,
                interrupt=interrupt,
                sync_to_thread=sync_to_thread,
            )

        return BaseObserver(
            callback=func,
            interrupt=interrupt,
            sync_to_thread=sync_to_thread,
        )

    if callback is not None:
        return decorator(callback)
    return decorator
