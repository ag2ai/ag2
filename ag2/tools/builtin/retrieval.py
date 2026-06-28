# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import AsyncExitStack, ExitStack
from dataclasses import dataclass, field

from ag2.annotations import Context, Variable
from ag2.events import BuiltinToolCallEvent, ToolCallEvent
from ag2.middleware import BaseMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.schemas import ToolSchema
from ag2.tools.tool import Tool

RETRIEVAL_TOOL_NAME = "retrieval"


@dataclass(slots=True)
class RetrievalToolSchema(ToolSchema):
    type: str = field(default=RETRIEVAL_TOOL_NAME, init=False)
    knowledge_id: str = ""
    prompt_template: str | None = None


class RetrievalTool(Tool):
    """Z.AI knowledge-base retrieval tool (Z.AI-specific; unsupported by other providers)."""

    __slots__ = (
        "_knowledge_id",
        "_prompt_template",
        "name",
    )

    def __init__(
        self,
        knowledge_id: str | Variable,
        *,
        prompt_template: str | Variable | None = None,
    ) -> None:
        self._knowledge_id = knowledge_id
        self._prompt_template = prompt_template
        self.name = RETRIEVAL_TOOL_NAME

    async def schemas(self, context: "Context") -> list[RetrievalToolSchema]:
        return [
            RetrievalToolSchema(
                knowledge_id=resolve_variable(self._knowledge_id, context, param_name="knowledge_id"),
                prompt_template=resolve_variable(self._prompt_template, context, param_name="prompt_template"),
            )
        ]

    def register(
        self,
        stack: "ExitStack | AsyncExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        async def execute(event: "ToolCallEvent", context: "Context") -> None:
            pass

        stack.enter_context(
            context.stream.where(BuiltinToolCallEvent.name == RETRIEVAL_TOOL_NAME).sub_scope(execute),
        )
