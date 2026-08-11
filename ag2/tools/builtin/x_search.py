# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import AsyncExitStack, ExitStack
from dataclasses import dataclass, field
from datetime import datetime

from ag2.annotations import Context, Variable
from ag2.events import BuiltinToolCallEvent, ToolCallEvent
from ag2.middleware import BaseMiddleware
from ag2.tools.schemas import ToolSchema
from ag2.tools.tool import Tool

from ._resolve import resolve_variable

X_SEARCH_TOOL_NAME = "x_search"


@dataclass(slots=True)
class XSearchToolSchema(ToolSchema):
    type: str = field(default=X_SEARCH_TOOL_NAME, init=False)
    allowed_x_handles: list[str] | None = None
    excluded_x_handles: list[str] | None = None
    from_date: datetime | None = None
    to_date: datetime | None = None
    enable_image_understanding: bool | None = None
    enable_video_understanding: bool | None = None


class XSearchTool(Tool):
    __slots__ = (
        "_params",
        "name",
    )

    def __init__(
        self,
        *,
        allowed_x_handles: list[str] | Variable | None = None,
        excluded_x_handles: list[str] | Variable | None = None,
        from_date: datetime | Variable | None = None,
        to_date: datetime | Variable | None = None,
        enable_image_understanding: bool | Variable | None = None,
        enable_video_understanding: bool | Variable | None = None,
    ) -> None:
        self._params: dict[str, object] = {}
        if allowed_x_handles is not None:
            self._params["allowed_x_handles"] = allowed_x_handles
        if excluded_x_handles is not None:
            self._params["excluded_x_handles"] = excluded_x_handles
        if from_date is not None:
            self._params["from_date"] = from_date
        if to_date is not None:
            self._params["to_date"] = to_date
        if enable_image_understanding is not None:
            self._params["enable_image_understanding"] = enable_image_understanding
        if enable_video_understanding is not None:
            self._params["enable_video_understanding"] = enable_video_understanding

        self.name = X_SEARCH_TOOL_NAME

    async def schemas(self, context: "Context") -> list[XSearchToolSchema]:
        resolved = {k: resolve_variable(v, context, param_name=k) for k, v in self._params.items()}
        return [XSearchToolSchema(**resolved)]

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
            context.stream.where(BuiltinToolCallEvent.name == X_SEARCH_TOOL_NAME).sub_scope(execute),
        )
