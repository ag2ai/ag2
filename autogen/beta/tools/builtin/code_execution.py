# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field

from autogen.beta.annotations import Context
from autogen.beta.middleware import BaseMiddleware
from autogen.beta.tools.schemas import ToolSchema
from autogen.beta.tools.tool import Tool


@dataclass(slots=True)
class CodeExecutionToolSchema(ToolSchema):
    """Provider-neutral capability flag for code execution."""

    type: str = field(default="code_execution", init=False)


class CodeExecutionTool(Tool):
    """Provider-neutral code execution capability.

    Each LLM client's mapper is responsible for converting this schema
    into the correct provider-specific API format.
    """

    async def schemas(self, context: "Context") -> list[ToolSchema]:
        return [CodeExecutionToolSchema()]

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        pass
