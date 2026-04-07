# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, overload

from autogen.beta.types import ClassInfo

from .events.conditions import Condition, TypeCondition

__all__ = ("Observer", "observer")


@dataclass(slots=True)
class Observer:
    condition: Condition
    callback: Callable[..., Any]
    interrupt: bool = False
    sync_to_thread: bool = True


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
) -> Observer: ...


@overload
def observer(
    condition: ClassInfo | Condition,
    callback: None = None,
    *,
    interrupt: bool = False,
    sync_to_thread: bool = True,
) -> Callable[[Callable[..., Any]], Observer]: ...


def observer(
    condition: ClassInfo | Condition,
    callback: Callable[..., Any] | None = None,
    *,
    interrupt: bool = False,
    sync_to_thread: bool = True,
) -> Observer | Callable[[Callable[..., Any]], Observer]:
    cond = _ensure_condition(condition)

    if callback is not None:
        return Observer(condition=cond, callback=callback, interrupt=interrupt, sync_to_thread=sync_to_thread)

    def decorator(func: Callable[..., Any]) -> Observer:
        return Observer(condition=cond, callback=func, interrupt=interrupt, sync_to_thread=sync_to_thread)

    return decorator
