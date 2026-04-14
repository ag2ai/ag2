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

__all__ = ("Observer", "observer")


@runtime_checkable
class Observer(Protocol):
    def register(self, stack: ExitStack, context: Context) -> None: ...


@dataclass(slots=True)
class StreamObserver:
    condition: Condition
    callback: Callable[..., Any]
    interrupt: bool = False
    sync_to_thread: bool = True

    def register(self, stack: ExitStack, context: Context) -> None:
        stack.enter_context(
            context.stream.where(self.condition).sub_scope(
                self.callback,
                interrupt=self.interrupt,
                sync_to_thread=self.sync_to_thread,
            )
        )


def _ensure_condition(condition: ClassInfo | Condition) -> Condition:
    if isinstance(condition, Condition):
        return condition
    return TypeCondition(condition)


@overload
def observer(
    condition: ClassInfo | Condition,
    callback: Callable[..., Any],
    *,
    interrupt: bool = False,
    sync_to_thread: bool = True,
) -> StreamObserver: ...


@overload
def observer(
    condition: ClassInfo | Condition,
    callback: None = None,
    *,
    interrupt: bool = False,
    sync_to_thread: bool = True,
) -> Callable[[Callable[..., Any]], StreamObserver]: ...


def observer(
    condition: ClassInfo | Condition,
    callback: Callable[..., Any] | None = None,
    *,
    interrupt: bool = False,
    sync_to_thread: bool = True,
) -> StreamObserver | Callable[[Callable[..., Any]], StreamObserver]:
    cond = _ensure_condition(condition)

    if callback is not None:
        return StreamObserver(condition=cond, callback=callback, interrupt=interrupt, sync_to_thread=sync_to_thread)

    def decorator(func: Callable[..., Any]) -> StreamObserver:
        return StreamObserver(condition=cond, callback=func, interrupt=interrupt, sync_to_thread=sync_to_thread)

    return decorator
