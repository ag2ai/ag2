# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import AsyncExitStack, ExitStack
from dataclasses import dataclass, field
from typing import Literal

from ag2.annotations import Context, Variable
from ag2.events import BuiltinToolCallEvent, ToolCallEvent
from ag2.middleware import BaseMiddleware
from ag2.tools.schemas import ToolSchema
from ag2.tools.tool import Tool

from ._resolve import resolve_variable

COLLECTIONS_SEARCH_TOOL_NAME = "collections_search"


@dataclass(slots=True)
class CollectionsSearchToolSchema(ToolSchema):
    type: str = field(default=COLLECTIONS_SEARCH_TOOL_NAME, init=False)
    collection_ids: list[str] = field(default_factory=list)
    limit: int | None = None
    instructions: str | None = None
    retrieval_mode: Literal["hybrid", "semantic", "keyword"] | None = None


class CollectionsSearchTool(Tool):
    """Provider-executed RAG search over uploaded document collections (xAI ``collections_search``).

    Provider support:

    - **xAI (Grok)** — semantic/keyword/hybrid search over the given document
      collections. Grok issues the searches server-side and uses the retrieved
      chunks to formulate its answer; the tool call surfaces as a
      :class:`~ag2.events.BuiltinToolCallEvent` but the API returns no result
      payload.

    - All other providers raise
      :class:`~ag2.exceptions.UnsupportedToolError`.

    See:
    - https://docs.x.ai/developers/tools/collections-search
    """

    __slots__ = (
        "_params",
        "name",
    )

    def __init__(
        self,
        collection_ids: list[str] | Variable,
        *,
        limit: int | Variable | None = None,
        instructions: str | Variable | None = None,
        retrieval_mode: Literal["hybrid", "semantic", "keyword"] | Variable | None = None,
    ) -> None:
        self._params: dict[str, object] = {"collection_ids": collection_ids}
        if limit is not None:
            self._params["limit"] = limit
        if instructions is not None:
            self._params["instructions"] = instructions
        if retrieval_mode is not None:
            self._params["retrieval_mode"] = retrieval_mode

        self.name = COLLECTIONS_SEARCH_TOOL_NAME

    async def schemas(self, context: "Context") -> list[CollectionsSearchToolSchema]:
        resolved = {k: resolve_variable(v, context, param_name=k) for k, v in self._params.items()}
        return [CollectionsSearchToolSchema(**resolved)]

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
            context.stream.where(BuiltinToolCallEvent.name == COLLECTIONS_SEARCH_TOOL_NAME).sub_scope(execute),
        )
