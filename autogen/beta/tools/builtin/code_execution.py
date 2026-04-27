# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from autogen.beta.annotations import Context
from autogen.beta.events import BuiltinToolCallEvent, ToolCallEvent
from autogen.beta.middleware import BaseMiddleware
from autogen.beta.tools.schemas import ToolSchema
from autogen.beta.tools.tool import Tool

CODE_EXECUTION_TOOL_NAME = "code_execution"

CodeExecutionVersions: TypeAlias = Literal["code_execution_20250825", "code_execution_20260120"]


@dataclass(slots=True)
class CodeExecutionToolSchema(ToolSchema):
    """Provider-neutral capability flag for code execution."""

    type: str = field(default=CODE_EXECUTION_TOOL_NAME, init=False)
    version: CodeExecutionVersions = "code_execution_20250825"


class CodeExecutionTool(Tool):
    """Provider-neutral code execution capability.

    Each LLM client's mapper is responsible for converting this schema
    into the correct provider-specific API format.
    """

    __slots__ = (
        "name",
        "_schema",
    )

    def __init__(
        self,
        *,
        version: CodeExecutionVersions = "code_execution_20250825",
    ) -> None:
        self._schema = CodeExecutionToolSchema(version=version)
        self.name = CODE_EXECUTION_TOOL_NAME

    async def schemas(self, context: "Context") -> list[ToolSchema]:
        return [self._schema]

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        async def execute(event: "ToolCallEvent", context: "Context") -> None:
            pass

        stack.enter_context(
            context.stream.where(BuiltinToolCallEvent.name == CODE_EXECUTION_TOOL_NAME).sub_scope(execute),
        )
