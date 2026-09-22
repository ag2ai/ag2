# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import AsyncExitStack, ExitStack

from ag2.annotations import Context
from ag2.middleware import BaseMiddleware

from .schemas import ToolSchema


class Tool(ABC):
    name: str

    @abstractmethod
    async def schemas(self, context: "Context") -> Iterable[ToolSchema]:
        """What capability this tool contributes to the model request.

        Every kind implements it; see ADR 0002.
        """
        ...

    def register(
        self,
        stack: "ExitStack | AsyncExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None: ...
