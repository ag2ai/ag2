# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from collections.abc import Iterable, Iterator
from contextlib import AsyncExitStack, ExitStack
from copy import deepcopy
from typing import Any

from ag2.annotations import Context
from ag2.middleware import BaseMiddleware

from .schemas import ToolSchema

# Carried by reference on copy: middleware is behaviour attached to a tool, not state
# belonging to it, so a shared limiter/budget/cache stays one object across every copy.
_SHARED_ATTRS = frozenset({"_middleware"})


def _iter_slots(cls: type) -> Iterator[str]:
    for klass in cls.__mro__:
        slots = klass.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        yield from slots


class Tool(ABC):
    name: str

    def __deepcopy__(self, memo: dict[int, Any]) -> "Tool":
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        names = list(getattr(self, "__dict__", {}))
        names.extend(name for name in _iter_slots(cls) if name not in ("__dict__", "__weakref__"))

        for name in names:
            try:
                value = getattr(self, name)
            except AttributeError:  # an unset slot
                continue
            setattr(new, name, value if name in _SHARED_ATTRS else deepcopy(value, memo))

        return new

    async def schemas(self, context: "Context") -> Iterable[ToolSchema]: ...

    def register(
        self,
        stack: "ExitStack | AsyncExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None: ...
