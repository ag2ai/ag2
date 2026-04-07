# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from typing import Protocol, runtime_checkable

from autogen.beta.annotations import Context
from autogen.beta.middleware import BaseMiddleware

from .schemas import ToolSchema


@runtime_checkable
class Tool(Protocol):
    async def schemas(self, context: "Context") -> Iterable[ToolSchema]: ...

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None: ...
