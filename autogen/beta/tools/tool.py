# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from collections.abc import Iterable
from contextlib import AsyncExitStack, ExitStack

from fast_depends import Provider

from autogen.beta.annotations import Context
from autogen.beta.middleware import BaseMiddleware

from .schemas import ToolSchema


class Tool(ABC):
    name: str

    def set_provider(self, provider: Provider) -> None:
        pass

    async def schemas(self, context: "Context") -> Iterable[ToolSchema]: ...

    def register(
        self,
        stack: "ExitStack | AsyncExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None: ...
