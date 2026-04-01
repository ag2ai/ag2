# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Literal

from autogen.beta.annotations import Context
from autogen.beta.middleware import BaseMiddleware
from autogen.beta.tools.schemas import ToolSchema
from autogen.beta.tools.tool import Tool


@dataclass(slots=True)
class MemoryToolSchema(ToolSchema):
    """Provider-neutral capability flag for the memory tool."""

    type: str = field(default="memory", init=False)
    version: Literal["memory_20250818"] = "memory_20250818"


class MemoryTool(Tool):
    """Memory tool that enables Claude to store and retrieve information across conversations.

    Claude can create, read, update, and delete files in the /memories directory to store
    what it learns while working, then reference those memories in future conversations.

    This is a client-side tool: the application is responsible for implementing the
    handlers for each memory command (view, create, str_replace, insert, delete, rename).

    See: https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool
    """

    def __init__(
        self,
        *,
        version: Literal["memory_20250818"] = "memory_20250818",
    ) -> None:
        self._schema = MemoryToolSchema(version=version)

    async def schemas(self, context: "Context") -> list[ToolSchema]:
        return [self._schema]

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        pass
