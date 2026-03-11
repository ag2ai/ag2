# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from typing import Protocol, runtime_checkable

from autogen.beta.annotations import Context
from autogen.beta.middleware import BaseMiddleware, ToolExecution

from .schemas import FunctionToolSchema


@runtime_checkable
class Tool(Protocol, ToolExecution):
    name: str
    schema: FunctionToolSchema

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        pass
