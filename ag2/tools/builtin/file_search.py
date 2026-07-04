# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import AsyncExitStack, ExitStack
from dataclasses import dataclass, field
from typing import Any

from ag2.annotations import Context, Variable
from ag2.events import BuiltinToolCallEvent, ToolCallEvent
from ag2.middleware import BaseMiddleware
from ag2.tools.schemas import ToolSchema
from ag2.tools.tool import Tool

from ._resolve import resolve_variable

FILE_SEARCH_TOOL_NAME = "file_search"


@dataclass(slots=True)
class FileSearchToolSchema(ToolSchema):
    type: str = field(default=FILE_SEARCH_TOOL_NAME, init=False)
    vector_store_ids: list[str] = field(default_factory=list)
    max_num_results: int | None = None
    filters: dict[str, Any] | None = None
    include_results: bool = False


class FileSearchTool(Tool):
    """Provider-executed vector-store retrieval (OpenAI Responses ``file_search``).

    Provider support:

    - **OpenAI Responses API** — maps to ``file_search`` over the given vector
      stores. ``filters`` is passed through verbatim (OpenAI comparison /
      compound filter objects). ``include_results=True`` asks the API to return
      the raw retrieved chunks (``include=["file_search_call.results"]``).

    - All other providers raise
      :class:`~ag2.exceptions.UnsupportedToolError`.

    See:
    - https://developers.openai.com/api/docs/guides/tools-file-search
    """

    __slots__ = (
        "_params",
        "name",
    )

    def __init__(
        self,
        vector_store_ids: list[str] | Variable,
        *,
        max_num_results: int | Variable | None = None,
        filters: dict[str, Any] | Variable | None = None,
        include_results: bool = False,
    ) -> None:
        self._params: dict[str, object] = {"vector_store_ids": vector_store_ids}
        if max_num_results is not None:
            self._params["max_num_results"] = max_num_results
        if filters is not None:
            self._params["filters"] = filters
        if include_results:
            self._params["include_results"] = include_results

        self.name = FILE_SEARCH_TOOL_NAME

    async def schemas(self, context: "Context") -> list[FileSearchToolSchema]:
        resolved = {k: resolve_variable(v, context, param_name=k) for k, v in self._params.items()}
        return [FileSearchToolSchema(**resolved)]

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
            context.stream.where(BuiltinToolCallEvent.name == FILE_SEARCH_TOOL_NAME).sub_scope(execute),
        )
